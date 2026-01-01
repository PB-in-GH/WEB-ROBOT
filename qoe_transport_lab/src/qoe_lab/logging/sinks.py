# qoe_transport_lab/src/qoe_lab/logging/sinks.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RunPaths:
    out_dir: str
    meta_path: str
    metrics_path: str


def make_run_dir(root: str, run_name: str) -> RunPaths:
    os.makedirs(root, exist_ok=True)
    # 防止覆盖：run_name 下若存在则追加数字
    base = os.path.join(root, run_name)
    out_dir = base
    k = 1
    while os.path.exists(out_dir):
        out_dir = f"{base}_{k}"
        k += 1
    os.makedirs(out_dir, exist_ok=True)
    return RunPaths(
        out_dir=out_dir,
        meta_path=os.path.join(out_dir, "meta.json"),
        metrics_path=os.path.join(out_dir, "metrics.jsonl"),
    )


class JsonlSink:
    def __init__(self, metrics_path: str):
        self.metrics_path = metrics_path
        self._f = open(metrics_path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()


def write_meta(meta_path: str, meta: Dict[str, Any]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
