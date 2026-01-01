# qoe_transport_lab/src/qoe_lab/trace/loader.py
from __future__ import annotations

import csv
from typing import List, Optional

from qoe_lab.rtc.trace_emulator import TracePoint


def load_trace_csv_piecewise(path: str) -> List[TracePoint]:
    """
    读取分段常量 trace CSV，直接返回 TraceRtcEmulator 所需的 List[TracePoint].

    CSV columns (required):
      - start_ms
      - bandwidth_bps
      - rtt_ms
      - loss_rate

    loss_rate: 0~1 (按帧丢，与你 TraceRtcEmulator 一致)
    """
    pts: List[TracePoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"start_ms", "bandwidth_bps", "rtt_ms", "loss_rate"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"trace csv 缺少必要列 {required}，实际列={reader.fieldnames}")

        for row in reader:
            start_ms = int(float(row["start_ms"]))
            bw_bps = float(row["bandwidth_bps"])
            rtt_ms = float(row["rtt_ms"])
            loss_rate = float(row["loss_rate"])

            if start_ms < 0:
                raise ValueError(f"start_ms 不能为负：{start_ms}")
            if bw_bps < 0:
                raise ValueError(f"bandwidth_bps 不能为负：{bw_bps}")
            if rtt_ms < 0:
                raise ValueError(f"rtt_ms 不能为负：{rtt_ms}")
            if not (0.0 <= loss_rate <= 1.0):
                raise ValueError(f"loss_rate 必须在[0,1]：{loss_rate}")

            pts.append(TracePoint(start_ms=start_ms, bandwidth_bps=bw_bps, rtt_ms=rtt_ms, loss_rate=loss_rate))

    if not pts:
        raise ValueError("trace csv 为空")

    pts = sorted(pts, key=lambda p: p.start_ms)

    # 确保从 0ms 开始（TraceRtcEmulator 内部也会补，但我们在 loader 里做强约束更稳）
    if pts[0].start_ms != 0:
        # 也可以选择自动补齐；这里选择补齐并保持与 TraceRtcEmulator 逻辑一致
        first = pts[0]
        pts = [TracePoint(start_ms=0, bandwidth_bps=first.bandwidth_bps, rtt_ms=first.rtt_ms, loss_rate=first.loss_rate)] + pts

    # 合并重复 start_ms（后者覆盖前者），避免奇怪输入
    merged: List[TracePoint] = []
    for p in pts:
        if merged and p.start_ms == merged[-1].start_ms:
            merged[-1] = p
        else:
            merged.append(p)

    return merged
