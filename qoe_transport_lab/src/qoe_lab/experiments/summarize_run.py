# qoe_transport_lab/src/qoe_lab/experiments/summarize_run.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from qoe_lab.metrics.network_score import (
    NetworkScoreConfig,
    NetworkScoreWeights,
    compute_network_score_from_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="runs/<name> directory")
    ap.add_argument("--max_delay_ms", type=float, default=400.0)
    ap.add_argument("--w_delay", type=float, default=0.1)
    ap.add_argument("--w_recv", type=float, default=0.1)
    ap.add_argument("--w_loss", type=float, default=0.5)
    args = ap.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    meta_path = os.path.join(args.run_dir, "meta.json")
    out_path = os.path.join(args.run_dir, "summary.json")

    cfg = NetworkScoreConfig(
        max_delay_ms=args.max_delay_ms,
        ground_truth_c_mbps=None,  # 使用 trace_bw_mbps 均值
        clip_0_100=True,
        weights=NetworkScoreWeights(args.w_delay, args.w_recv, args.w_loss),
    )

    ns = compute_network_score_from_jsonl(metrics_path, cfg)

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    summary = {
        "network_score": ns.network_score,
        "delay_score": ns.delay_score,
        "recv_score": ns.recv_score,
        "loss_score": ns.loss_score,
        "delay_95th_ms": ns.delay_95th_ms,
        "min_delay_ms": ns.min_delay_ms,
        "avg_recv_mbps": ns.avg_recv_mbps,
        "avg_trace_bw_mbps": ns.avg_bw_mbps,
        "avg_loss_pct": ns.avg_loss_ratio * 100.0,
        "meta": meta,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {out_path}")
    print(
        f"Ns={ns.network_score:.3f}  ds={ns.delay_score:.3f}  cs={ns.recv_score:.3f}  ls={ns.loss_score:.3f}  "
        f"p95={ns.delay_95th_ms:.1f}ms  recv={ns.avg_recv_mbps:.3f}Mbps  bw={ns.avg_bw_mbps:.3f}Mbps  loss={100*ns.avg_loss_ratio:.2f}%"
    )


if __name__ == "__main__":
    main()
