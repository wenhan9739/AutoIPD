# -*- coding: utf-8 -*-
"""
kmfig.render — 把 PDF 页面统一加载为：高 DPI 光栅 + 文字（含颜色/方向）+ 矢量路径，
全部换算到同一套“光栅像素坐标”，后续模块只在这一套坐标下工作。
对于纯图片输入（jpg/png，如 docx/pptx 里的图或 MinerU 裁出的图）则只有光栅。
"""
import numpy as np
import cv2
import fitz  # PyMuPDF


def _int_to_rgb(c):
    """PyMuPDF 文本 span 的 color 是 sRGB 整数。"""
    if c is None:
        return None
    return ((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0


class PageData:
    """统一的页面数据容器。像素坐标系：原点左上，x 右，y 下。"""

    def __init__(self, img_bgr, zoom=1.0, words=None, spans=None, drawings=None,
                 source=None, page_no=None, has_vector=False):
        self.img = img_bgr
        self.zoom = zoom
        self.words = words or []        # dict(x0,y0,x1,y1,text)
        self.spans = spans or []        # dict(x0,y0,x1,y1,text,color(rgb 0-1),size,vertical)
        self.drawings = drawings or []  # dict(color, fill, width, dashed, segs=[(x0,y0,x1,y1)], rect)
        self.source = source
        self.page_no = page_no
        self.has_vector = has_vector

    @property
    def shape(self):
        return self.img.shape[:2]


def load_pdf_page(pdf_path, page_no, dpi=300, clip=None):
    """渲染 PDF 某页；clip 为 PDF 点坐标 fitz.Rect（可选，裁剪渲染区域）。"""
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    clip_rect = fitz.Rect(clip) if clip is not None else None
    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[..., :3]
    img_bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    ox, oy = (clip_rect.x0, clip_rect.y0) if clip_rect is not None else (0.0, 0.0)

    def tx(x):
        return (x - ox) * zoom

    def ty(y):
        return (y - oy) * zoom

    # 文字（word 级）
    words = []
    for w in page.get_text("words", clip=clip_rect):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        words.append(dict(x0=tx(x0), y0=ty(y0), x1=tx(x1), y1=ty(y1), text=text))
    # 文字（span 级，带颜色与方向）
    spans = []
    d = page.get_text("dict", clip=clip_rect)
    for blk in d.get("blocks", []):
        for line in blk.get("lines", []):
            direction = line.get("dir", (1, 0))
            vertical = abs(direction[0]) < 0.5
            for sp in line.get("spans", []):
                bb = sp["bbox"]
                spans.append(dict(x0=tx(bb[0]), y0=ty(bb[1]), x1=tx(bb[2]), y1=ty(bb[3]),
                                  text=sp.get("text", ""), color=_int_to_rgb(sp.get("color")),
                                  size=sp.get("size", 0) * zoom, vertical=vertical))
    # 矢量路径 → 只保留直线段（'l'）与矩形（'re'），换算到像素
    drawings = []
    for dr in page.get_drawings():
        segs = []
        n_curve = 0
        for it in dr["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                segs.append((tx(p1.x), ty(p1.y), tx(p2.x), ty(p2.y)))
            elif it[0] == "re":
                r = it[1]
                segs.append((tx(r.x0), ty(r.y0), tx(r.x1), ty(r.y0)))
                segs.append((tx(r.x1), ty(r.y0), tx(r.x1), ty(r.y1)))
                segs.append((tx(r.x1), ty(r.y1), tx(r.x0), ty(r.y1)))
                segs.append((tx(r.x0), ty(r.y1), tx(r.x0), ty(r.y0)))
            elif it[0] == "c":
                n_curve += 1
        if not segs and n_curve == 0:
            continue
        rect = dr.get("rect")
        dashes = dr.get("dashes")
        dashed = bool(dashes) and dashes not in ("[] 0", "[]", "")
        drawings.append(dict(color=dr.get("color"), fill=dr.get("fill"),
                             width=(dr.get("width") or 0) * zoom, dashed=dashed,
                             n_curve=n_curve, segs=segs,
                             rect=(tx(rect.x0), ty(rect.y0), tx(rect.x1), ty(rect.y1)) if rect else None))
    has_vector = len(words) > 0 or len(drawings) > 0
    return PageData(img_bgr, zoom=zoom, words=words, spans=spans, drawings=drawings,
                    source=pdf_path, page_no=page_no, has_vector=has_vector)


def load_image(img_path, min_width=1600):
    """加载普通图片；分辨率过低时放大到 min_width，方便 OCR 与线检测。"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {img_path}")
    scale = 1.0
    if img.shape[1] < min_width:
        scale = min_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return PageData(img, zoom=scale, source=img_path, page_no=None, has_vector=False)


def page_is_scanned(pdf_path, page_no):
    """页面几乎没有文字层且被一张大图覆盖 → 视为扫描件/位图页。"""
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    n_words = len(page.get_text("words"))
    imgs = page.get_image_info()
    big = any((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) > 0.5 * page.rect.width * page.rect.height
              for i in imgs)
    return n_words < 20 and big
