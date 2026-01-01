# qoe_transport_lab/src/qoe_lab/trace/csv_trace.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TracePoint:
    t_s: float
    bw_mbps: float
    rtt_ms: float
    loss_pct: float = 0.0


def load_trace_csv(path: str) -> List[TracePoint]:
    """
    CSV columns (required): t_s,bw_mbps,rtt_ms
    optional: loss_pct
    """
    pts: List[TracePoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"t_s", "bw_mbps", "rtt_ms"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"trace csv 缺少必要列 {need}，实际列={reader.fieldnames}")

        for row in reader:
            t_s = float(row["t_s"])
            bw = float(row["bw_mbps"])
            rtt = float(row["rtt_ms"])
            loss = float(row.get("loss_pct", "0") or 0.0)
            pts.append(TracePoint(t_s=t_s, bw_mbps=bw, rtt_ms=rtt, loss_pct=loss))

    if not pts:
        raise ValueError("trace csv 为空")

    pts = sorted(pts, key=lambda x: x.t_s)
    if pts[0].t_s != 0.0:
        raise ValueError(f"trace 必须从 t_s=0 开始，当前首行 t_s={pts[0].t_s}")

    # 基本合法性
    for p in pts:
        if p.bw_mbps < 0 or p.rtt_ms < 0 or p.loss_pct < 0:
            raise ValueError(f"trace 存在负值：{p}")

    return pts
