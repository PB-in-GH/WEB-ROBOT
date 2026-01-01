# qoe_transport_lab/src/qoe_lab/experiments/collect_summaries.py
from __future__ import annotations

import argparse
import glob
import json
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="/home/pb/web-robot/qoe_transport_lab/runs")
    ap.add_argument("--pattern", type=str, default="*")
    ap.add_argument("--out", type=str, default=None, help="output tsv path; default: stdout")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs_root, args.pattern)) if os.path.isdir(d))

    rows = []
    for d in run_dirs:
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp, "r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "run": os.path.basename(d),
            "Ns": s.get("network_score"),
            "ds": s.get("delay_score"),
            "cs": s.get("recv_score"),
            "ls": s.get("loss_score"),
            "p95_ms": s.get("delay_95th_ms"),
            "recv_mbps": s.get("avg_recv_mbps"),
            "bw_mbps": s.get("avg_trace_bw_mbps"),
            "loss_pct": s.get("avg_loss_pct"),
        })

    header = ["run","Ns","ds","cs","ls","p95_ms","recv_mbps","bw_mbps","loss_pct"]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join("" if r[h] is None else (f"{r[h]:.6g}" if isinstance(r[h], (int,float)) else str(r[h])) for h in header))

    out_text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"[ok] wrote {args.out}")
    else:
        print(out_text, end="")

if __name__ == "__main__":
    main()
