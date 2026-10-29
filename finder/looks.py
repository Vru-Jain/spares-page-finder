"""How a page looks as a whole, as 768 numbers, for the picture model in learn.py.

The network is DINOv2-small (Meta, Apache-2.0), a general image model never trained on
manuals, converted for OpenVINO by scripts/convert_dinov2.py. It sees the page squeezed
to 448 x 448 pixels and gives its class token and the mean of its patches. Measured on
6,198 scanned pages of H1868 and BERGE DAISEN: on top of the picture reader's own
verdict it tells many "check" pages that are not spares pages from those that are,
which the reader alone cannot. 0.1 s a page inside the picture workers.

The squeeze goes through 1,024 pixels and then Pillow's bicubic, as in the experiment
that chose the network: one step straight from the 200-dpi page (OpenCV's area
average) lost thin table lines, and with them a spares page or two per vessel.

The numbers of every picture-judged page are kept in data/looks/<manual's sha256>.npz,
so people's marks can teach the model later (learn.py) without reading the page again.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from finder import store

NET = Path(__file__).parent.parent / "models" / "dinov2-small.xml"
SIDE = 448
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet's, which the network was trained with
STD = np.array([0.229, 0.224, 0.225], np.float32)

_NET = None   # this worker's compiled network


def start(threads: int) -> None:
    """Load the network into this worker process, if it is there."""
    global _NET
    if NET.exists():
        import openvino
        _NET = openvino.Core().compile_model(str(NET), "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads, "ENABLE_CPU_PINNING": False})


def vector(img: np.ndarray) -> np.ndarray | None:
    """The 768 numbers of a page image (RGB, any size), or None without the network."""
    if _NET is None:
        return None
    scale = 1024 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = np.asarray(Image.fromarray(img).resize((SIDE, SIDE), Image.BICUBIC), np.float32)
    x = ((small / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    return _NET(x)[0][0].astype(np.float16)


def features(vectors: np.ndarray, verdicts) -> np.ndarray:
    """What the picture model sees: the numbers, and what the picture reader said."""
    said = np.array([[v == "spares", v == "check"] for v in verdicts], np.float32).reshape(-1, 2)
    return np.hstack([np.asarray(vectors, np.float32).reshape(-1, 768), said])


def file_of(sha: str) -> Path:
    return store.DATA / "looks" / f"{sha}.npz"


def save(sha: str, got: dict[int, tuple[str, np.ndarray]]) -> None:
    """Keep {page: (picture verdict, numbers)} of one manual."""
    if not got:
        return
    file_of(sha).parent.mkdir(parents=True, exist_ok=True)
    pages = sorted(got)
    with open(file_of(sha), "wb") as fh:   # np.savez would add ".npz" to a name it does not end with
        np.savez(fh, pages=np.array(pages), verdicts=np.array([got[n][0] for n in pages]),
                 vectors=np.stack([got[n][1] for n in pages]))


def load(sha: str) -> dict[int, tuple[str, np.ndarray]]:
    """{page: (picture verdict, numbers)} kept for a manual; empty if none were."""
    if not sha or not file_of(sha).exists():
        return {}
    with np.load(file_of(sha)) as kept:
        return {int(n): (str(v), x) for n, v, x in zip(kept["pages"], kept["verdicts"], kept["vectors"])}
