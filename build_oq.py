"""Builds an oQe quant of Ripple-65B, calibrating on an existing q8 copy.

usage: build_oq.py <level> <bf16 source> <q8 copy> <output folder>

oQ falls back to a uniform 4-bit copy when the source is too big to calibrate in memory. The q8 copy
is much closer to the source and needs about half the memory of BF16, so it stands in for that copy.
The imatrix is cached in the output folder, so later levels skip collecting it.
"""
import logging
import os
import sys
import tempfile
from pathlib import Path

from omlx import oq

level, source, q8, out = float(sys.argv[1]), *sys.argv[2:5]
level = int(level) if level == int(level) else level
out = Path(out)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)

budget = oq._calibration_memory_budget


def always_proxy(*args, **kwargs):
    result = dict(budget(*args, **kwargs))
    result["requires_proxy"] = True
    result["capacity_bytes"] = 0
    return result


def links_to_q8(*args, working_dir=None, **kwargs):
    """A folder of links, because oQ deletes its proxy when it finishes."""
    folder = Path(tempfile.mkdtemp(prefix="q8-links-", dir=working_dir))
    for item in Path(q8).iterdir():
        if item.is_file():
            (folder / item.name).symlink_to(item)
    return folder


oq._calibration_memory_budget = always_proxy
oq._build_proxy_for_sensitivity = links_to_q8

name = f"ripple-65b-oq{level:g}e"
oq.quantize_oq_streaming(
    source,
    str(out / name),
    level,
    enhanced=True,
    imatrix_cache_path=str(out / ".oqe_imatrix" / "ripple-65b-s128-l512.npz"),
    imatrix_strict=True,
    sensitivity_model_path=q8,
    stream_calibration=False,
    progress_callback=lambda phase, pct, *rest: print(f"[{pct:5.1f}%] {phase} {rest[0] if rest else ''}", flush=True),
)
print("finished", name, flush=True)
