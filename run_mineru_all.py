# -*- coding: utf-8 -*-
"""批量 MinerU 处理驱动：按试验分目录输出，跳过已完成，记录失败清单。"""
import os, sys, subprocess, time, json

ROOT = os.environ.get("IPDR_ROOT", r"D:\work\IPDR")
SRC = os.path.join(ROOT, "最终所纳入27个RCT", "最终所纳入27个RCT")
OUT = os.path.join(ROOT, "mineru_out")
LOG = os.path.join(ROOT, "pipeline", "mineru_batch.log")
STATE = os.path.join(ROOT, "pipeline", "mineru_batch_state.json")

EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg"}

def collect_jobs():
    jobs = []
    for trial in sorted(os.listdir(SRC)):
        d = os.path.join(SRC, trial)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTS:
                stem = os.path.splitext(f)[0]
                # 完成标志: mineru输出目录下存在 *_content_list.json（pdf→auto/，docx/pptx→office/）
                base = os.path.join(OUT, trial, stem)
                marker = False
                for sub in ("auto", "office"):
                    done = os.path.join(base, sub)
                    if os.path.isdir(done) and any(x.endswith("_content_list.json")
                                                   for x in os.listdir(done)):
                        marker = True
                        break
                jobs.append({"trial": trial, "file": os.path.join(d, f),
                             "outdir": os.path.join(OUT, trial), "done": marker})
    return jobs

def main():
    jobs = collect_jobs()
    state = {"started": time.strftime("%F %T"), "jobs": []}
    with open(LOG, "a", encoding="utf-8") as log:
        for j in jobs:
            if j["done"]:
                state["jobs"].append({**j, "status": "skipped_done"})
                continue
            t0 = time.time()
            log.write(f"[{time.strftime('%F %T')}] START {j['trial']} / {os.path.basename(j['file'])}\n")
            log.flush()
            cmd = ["mineru", "-p", j["file"], "-o", j["outdir"],
                   "-b", "pipeline", "-d", "cuda:0"]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            dt = time.time() - t0
            # 完成判定重新检查（pdf→auto/，docx/pptx→office/）
            base = os.path.join(j["outdir"], os.path.splitext(os.path.basename(j["file"]))[0])
            ok = False
            for sub in ("auto", "office"):
                done_dir = os.path.join(base, sub)
                if os.path.isdir(done_dir) and any(x.endswith("_content_list.json") for x in os.listdir(done_dir)):
                    ok = True
                    break
            status = "ok" if ok else "FAILED"
            log.write(f"[{time.strftime('%F %T')}] {status} {j['trial']} / {os.path.basename(j['file'])} "
                      f"({dt:.0f}s, rc={r.returncode})\n")
            if not ok:
                log.write("  STDERR tail: " + (r.stderr or "")[-800:] + "\n")
            log.flush()
            state["jobs"].append({**j, "status": status, "secs": round(dt)})
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    fails = [j for j in state["jobs"] if j.get("status") == "FAILED"]
    print(f"BATCH DONE. ok={sum(1 for j in state['jobs'] if j.get('status')=='ok')} "
          f"skip={sum(1 for j in state['jobs'] if j.get('status')=='skipped_done')} failed={len(fails)}")

if __name__ == "__main__":
    main()
