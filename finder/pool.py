"""Worker processes for the reader, used by text.py and pictures.py.

A pool is started on first use and kept for every manual after, since starting it (and
the OCR models in it) takes seconds; a pool that breaks is dropped, and the next manual
starts a fresh one. Workers run on the PC's fast cores only, and keep open the PDF
they are reading.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf as fitz


def fast_cores() -> tuple[int, int]:
    """(affinity mask of the performance cores, how many such cores). On this PC the
    four efficiency cores pace any thread team that includes them: 8 threads took
    3.7 s a page unpinned and 2.1 s pinned. (0, 0) when Windows will not say."""
    try:
        import ctypes
        from ctypes import wintypes

        class CpuSet(ctypes.Structure):   # SYSTEM_CPU_SET_INFORMATION
            _fields_ = [("Size", wintypes.DWORD), ("Type", wintypes.DWORD), ("Id", wintypes.DWORD),
                        ("Group", wintypes.WORD), ("Logical", ctypes.c_ubyte), ("Core", ctypes.c_ubyte),
                        ("Cache", ctypes.c_ubyte), ("Numa", ctypes.c_ubyte), ("Efficiency", ctypes.c_ubyte)]

        kernel = ctypes.WinDLL("kernel32")
        size = wintypes.ULONG()
        kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
        raw = ctypes.create_string_buffer(size.value)
        if not kernel.GetSystemCpuSetInformation(raw, size, ctypes.byref(size), None, 0):
            return 0, 0
        cpus, at = [], 0
        while at < size.value:
            cpu = CpuSet.from_buffer_copy(raw, at)
            cpus.append(cpu)
            at += cpu.Size
        best = max(cpu.Efficiency for cpu in cpus)
        fast = [cpu for cpu in cpus if cpu.Efficiency == best and cpu.Group == 0]
        return sum(1 << cpu.Logical for cpu in fast), len({cpu.Core for cpu in fast})
    except Exception:
        return 0, 0


_cores = None


def pin() -> int:
    """Keep this process, and the workers it starts from now on, on the fast cores.
    Returns how many fast cores there are, 0 when Windows will not say."""
    global _cores
    if _cores is None:
        mask, _cores = fast_cores()
        if mask:
            import ctypes
            kernel = ctypes.WinDLL("kernel32")
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask)
    return _cores


class Pool:
    """`size` worker processes, each prepared by initializer(*initargs(fast cores))."""

    def __init__(self, size: int, initializer=None, initargs=lambda cores: ()):
        self.size, self.initializer, self.initargs = size, initializer, initargs
        self._executor = None

    def each(self, fn, calls):
        """fn(*call) for every call, in the workers; results come as they finish. A dead
        worker raises here, which fails the manual, and the pool is dropped."""
        if not calls:
            return   # nothing to do: no need to start the workers
        try:
            if self._executor is None:
                cores = pin()
                self._executor = ProcessPoolExecutor(self.size, initializer=self.initializer,
                                                     initargs=self.initargs(cores))
            for future in as_completed([self._executor.submit(fn, *call) for call in calls]):
                yield future.result()
        except Exception:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            raise


_open = (None, None)   # (path, document): the PDF this worker process is reading


def open_pdf(path: str):
    """The PDF at `path`, opened once per worker for all the pages it is given."""
    global _open
    if _open[0] != path:
        _open = (path, fitz.open(path))
    return _open[1]
