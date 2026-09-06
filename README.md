# IPDR-KM2IPD：从已发表RCT的PDF重建个体患者数据（IPD）

**版本 v1.0**

从已发表随机对照试验（RCT）的 PDF 文献出发，端到端地完成：
**文献解析 → KM 曲线发现 → 数字化提取 → IPD 重建（Guyot 算法）→ 质检**，
并输出带人群/干预标注的个体患者数据与三方一致性验证报告。

> 适用场景：meta 分析 / IPD 荟萃中，仅能获得发表 KM 曲线而无法获得原始数据时，
> 批量重建各试验各臂的伪 IPD 并进行生存分析。

## 方法总览（五级流水线）

```
RCT 文献 PDF / docx / pptx
   │  ① MinerU 批量解析（版面分析 + chart检测 + 文本/图片提取）
   ▼
② 候选页筛选（caption/关键词/矢量密集页） → 候选页清单 manifest
   ▼
③ KM 曲线数字化引擎
   │    面板检测（L形坐标轴配对 + 容器剔除）
   │    坐标标定（矢量文字层刻度 / 刻度线定位+小窗OCR，RANSAC稳健拟合）
   │    曲线提取（矢量描边路径优先；位图色相聚类+逐列跟踪；黑白图/点线兜底）
   │    KM先验修复（起点(0,1)、单调性、删失标记断口拼合、伪臂去重）
   ▼
④ IPD 重建（Guyot 2012 算法，双实现互相验证）
   │    Python 移植版（kmfig/guyot.py，逐行对齐 IPDfromKM）
   │    R 版（IPDfromKM::preprocess + getIPD，同输入对拍）
   ▼
⑤ 质检（四层）
      数据完整性 / 轨迹贴合度(precision) / 重建KM vs 原曲线 / 与文中报告值比对
      + 人群/干预标注总表 + 逐面板三方叠加核验图
```

## 验证结果（27 篇 RCT、477 条曲线实测）

- 曲线数字化：轨迹贴合 precision 中位数 1.000；与文中报告中位数比对，105/156 条相对差 ≤5%
- IPD 重建：78 面板 / 180 臂；**Python 与 R 同输入对拍 141/180 逐行一致**，
  其余仅删失时间放置微差（两实现重建 KM 曲线间 RMSE ≤0.007）
- 重建 KM vs 原数字化曲线：RMSE 中位 0.0092（0.92 个百分点）
- 重建中位 vs 数字化中位：差值中位数 0.000 个月
- 重建 IPD 的 Cox HR 与图中发表 HR 一致（例：KEYNOTE-189 PFS 2.036 ∈ 发表逆向 CI；AK105-302 2.344 vs 逆向 2.33）

## 目录结构

```
pipeline/
├── run_mineru_all.py        # ① MinerU 批量解析（断点续跑）
├── select_figures.py        # ② OS/PFS 候选页筛选 → work/manifest.json
├── kmfig/                   # ③④ 数字化+重建引擎
│   ├── render.py            #    PDF/图片统一渲染（文字层+矢量路径+光栅）
│   ├── panels.py            #    面板检测（L形坐标轴）
│   ├── calibrate.py         #    坐标轴标定（矢量/OCR 双路径）
│   ├── vector_extract.py    #    矢量曲线提取
│   ├── raster_extract.py    #    位图曲线提取（跟踪法）
│   ├── digitize.py          #    面板编排 + QA
│   └── guyot.py             #    Guyot IPD 重建（Python 移植版）
├── digitize_all.py          # ③④ 批量数字化驱动
├── ipd_reconstruct.py       # ④ 批量 IPD 重建（Python Guyot）
├── ipdfromkm_batch.R        # ④ R IPDfromKM 批量重建 + coxph
├── compare_ipd.py           # ⑤ Py/R/原曲线 三方对比
├── verify_all.py            # ⑤ 定量贴合核验（precision/完整性）
├── qa_ipd_panels.py         # ⑤ 逐面板三方叠加质检图
├── build_master.py          # ⑤ 人群/干预/角色 标注总表
├── trial_arms.json          #    27 试验 curated 干预/人群对照表
├── annotate_panels.py       #    人群(图注/字母映射)+干预+OCR风险表补齐
├── validate.py              #    与文中报告中位数比对
├── update_published_medians.py  # 发表中位双源抽取（图内标注+md）
└── redraw_plots.py          #    重建 vs 原曲线 叠加图重绘
```

## 运行

```bash
pip install -r requirements.txt
# R 侧需安装 IPDfromKM: install.packages("IPDfromKM")

export IPDR_ROOT=/path/to/workdir        # 数据与结果的根目录
mkdir -p $IPDR_ROOT/最终所纳入27个RCT     # 按试验名分文件夹放置 PDF

cd pipeline
python run_mineru_all.py                 # ① 批量 MinerU（可断点续跑）
python select_figures.py                 # ② 候选页清单
python digitize_all.py                   # ③ 全量数字化（QA 打分）
python ipd_reconstruct.py                # ④ Python Guyot 批量重建
Rscript ipdfromkm_batch.R \              # ④ R IPDfromKM（同输入）
    "$IPDR_ROOT/results/ipd/r_input" "$IPDR_ROOT/results/ipd/r_out"
python compare_ipd.py                    # ⑤ 三方对比报告
python verify_all.py                     # ⑤ 定量贴合核验
python qa_ipd_panels.py                  # ⑤ 逐面板肉眼质检图
python build_master.py                   # ⑤ 人群/干预标注总表 curves_master.csv
```

环境变量 `IPDR_ROOT` 为工作根目录（含文献 PDF、mineru_out/、work/、results/）。

## 输出结构

```
$IPDR_ROOT/
├── mineru_out/<试验>/<文档>/     # MinerU 解析结果
├── work/verify/                  # 逐臂轨迹核验图（红=原曲线未覆盖 绿=轨迹偏离 黄=重合）
├── results/
│   ├── curves_master.csv         # ★ 总表：试验/人群/干预/角色/中位/QA（每臂一行）
│   ├── QA_report.md              # 质检报告（需人工复核清单）
│   ├── verification.csv          # 定量核验（precision/完整性/视觉结论）
│   ├── overlays/                 # 数字化 vs 发表图 叠加核验图
│   └── ipd/
│       ├── <面板>__<臂>.ipd.csv  # ★ 重建伪IPD（time, event）
│       ├── plots/                # 重建KM vs 原曲线 叠加图
│       ├── key_values.csv        # 里程碑生存率/中位/风险数误差
│       ├── compare_report.csv    # Py/R/原曲线 三方对比
│       └── r_out/                # R IPDfromKM 结果 + coxph HR
```

`curves_master.csv` 关键列：**人群**（试验级人群+面板亚组）、**干预**（标准化药名+联合方案）、
**角色**（experimental/control）、中位_重建/中位_数字化、重建vs原曲线RMSE、QA通过、
重建IPD/数字化CSV/面板JSON 路径——可直接按"人群+干预+角色"筛选后读入 R `coxph` 或 Shiny 应用。

## 质检流程（四层）

1. **数据完整性**：CSV 与 JSON 逐行一致、时间单调、surv∈[0,1]、起点=(0,1)
2. **轨迹贴合**：沿提取轨迹每 3px 采样到最近"原曲线像素"距离 ≤2.5px 的比例（precision）
3. **三方曲线对比**：原数字化 vs Python重建KM vs R重建KM 叠加图 + RMSE/中位差
4. **与发表值闭环**：图中标注的中位/HR 与数字化中位、重建 Cox HR 比对

逐面板三方质检图（肉眼复核用）由 `qa_ipd_panels.py` 生成，
每图含：原图数字化曲线、Py/R 重建曲线、发表图裁剪、人群/干预标注。

## 已知边界

- 矢量 PDF 走矢量提取（精度≈0.05 月）；位图/扫描件走光栅跟踪（精度受分辨率限制）
- 双臂曲线长距离重叠段约有半个线宽（<1% 生存率）的偏差（QA 的 overlap 列给出提示）
- 纯黑双实线（无颜色/线型区分）的 KM 图无法自动分臂
- 随访末期小平台 + 少量患者的臂，Guyot 整数舍入可使中位偏移 >0.5 月（7/156 实测）
- 无风险表数字的臂无法做 Guyot 重建（本流水线已含 OCR 风险表补齐，覆盖率 231/237 面板）

## 引用与依赖

- Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data:
  reconstructing the data from published Kaplan-Meier survival curves.
  BMC Med Res Methodol. 2012;12:9.
- IPDfromKM R 包（算法参考实现，CRAN: IPDfromKM）
- MinerU（PDF 解析）：https://github.com/opendatalab/MinerU
- RapidOCR（刻度/风险表 OCR）
- 本项目代码基于上述公开算法与工具实现，IPD 重建结果经 R IPDfromKM 对拍验证。

## License

MIT
