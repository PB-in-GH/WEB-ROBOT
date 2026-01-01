# qoe_transport_lab/src/qoe_lab/metrics/network_score.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NetworkScoreWeights:
    w_delay: float = 0.1
    w_recv: float = 0.1
    w_loss: float = 0.5


@dataclass
class NetworkScoreConfig:
    # BoB 里 max_delay 固定为 400ms；这里做成可配置
    max_delay_ms: float = 400.0
    # ground_truth_c：理想环境下可达到的平均带宽（Mbps）
    # 若不提供，则从 trace 的 bw_mbps 做均值估计
    ground_truth_c_mbps: Optional[float] = None
    # 对异常值做裁剪，避免出现负分/超过 100
    clip_0_100: bool = True
    weights: NetworkScoreWeights = NetworkScoreWeights()


@dataclass
class NetworkScoreResult:
    # 总分与子分
    network_score: float
    delay_score: float
    recv_score: float
    loss_score: float

    # 用于复现/诊断的中间量
    delay_95th_ms: float
    min_delay_ms: float
    avg_recv_mbps: float
    avg_bw_mbps: float
    avg_loss_ratio: float  # 0~1


def _percentile(values: List[float], p: float) -> float:
    """
    线性插值分位数。p in [0,100].
    """
    if not values:
        raise ValueError("percentile: empty values")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    n = len(xs)
    # 位置定义：r = (n-1)*p/100
    r = (n - 1) * (p / 100.0)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        return xs[lo]
    frac = r - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_network_score_from_records(
    records: List[Dict[str, Any]],
    cfg: NetworkScoreConfig = NetworkScoreConfig(),
) -> NetworkScoreResult:
    """
    records: 从 metrics.jsonl 读出的 list[dict]
    需要字段（缺失会报错）：
      - e2e_ms: float
      - recv_rate_mbps: float
      - loss_pct: float (0~100)
      - bw_mbps: float （trace 设定带宽）
    """
    if not records:
        raise ValueError("records 为空，无法计算")

    def _get_float(r, key):
        if key not in r:
            raise KeyError(f"缺少字段 {key}，record keys={list(r.keys())}")
        return float(r[key])

    e2e = [_get_float(r, "e2e_ms") for r in records]
    recv = [_get_float(r, "rx_rate_mbps") for r in records]
    loss_ratio = [_get_float(r, "net_loss_pct") / 100.0 for r in records]
    bw = [_get_float(r, "trace_bw_mbps") for r in records]

    delay_95 = _percentile(e2e, 95.0)
    min_delay = min(e2e)

    avg_recv = sum(recv) / len(recv)
    avg_bw = sum(bw) / len(bw) if bw else 0.0

    # ground_truth_c：若没给，用 trace bw 均值近似
    if cfg.ground_truth_c_mbps is not None:
        gt_c = float(cfg.ground_truth_c_mbps)
    else:
        gt_c = avg_bw

    # 防止除零
    if gt_c <= 1e-9:
        # 如果 trace 里没有 bw_mbps 字段或全为 0，这里必须显式报错
        raise ValueError("ground_truth_c_mbps 无法确定（gt_c=0）。请在日志中记录 bw_mbps，或显式传 cfg.ground_truth_c_mbps。")

    # ---- 子分定义（按 BoB 论文结构实现） ----
    # d_s = 100 * (max_delay - delay_95th) / (max_delay - min_delay)
    denom = (cfg.max_delay_ms - min_delay)
    if abs(denom) < 1e-9:
        # min_delay == max_delay 时，d_s 不可定义；此时给满分（说明全程 delay 已经到上限，属于极端）
        d_s = 0.0
    else:
        d_s = 100.0 * (cfg.max_delay_ms - delay_95) / denom

    # c_s = 100 * c / ground_truth_c
    c_s = 100.0 * avg_recv / gt_c

    # l_s = 100 * (1 - l)，l 为 loss ratio
    avg_l = sum(loss_ratio) / len(loss_ratio)
    l_s = 100.0 * (1.0 - avg_l)

    if cfg.clip_0_100:
        d_s = _clip(d_s, 0.0, 100.0)
        c_s = _clip(c_s, 0.0, 100.0)
        l_s = _clip(l_s, 0.0, 100.0)

    w = cfg.weights
    ns = w.w_delay * d_s + w.w_recv * c_s + w.w_loss * l_s

    return NetworkScoreResult(
        network_score=float(ns),
        delay_score=float(d_s),
        recv_score=float(c_s),
        loss_score=float(l_s),
        delay_95th_ms=float(delay_95),
        min_delay_ms=float(min_delay),
        avg_recv_mbps=float(avg_recv),
        avg_bw_mbps=float(avg_bw),
        avg_loss_ratio=float(avg_l),
    )


def load_metrics_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_network_score_from_jsonl(
    metrics_jsonl_path: str,
    cfg: NetworkScoreConfig = NetworkScoreConfig(),
) -> NetworkScoreResult:
    return compute_network_score_from_records(load_metrics_jsonl(metrics_jsonl_path), cfg)
