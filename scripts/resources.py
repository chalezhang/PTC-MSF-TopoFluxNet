"""Process-local CPU reservation for concurrent independent OpenFOAM cases.

Accounting is advisory: OpenFOAM serial executables may ignore OMP_NUM_THREADS;
BLAS/native libraries may also spawn extra threads. The semaphore prevents our
own dispatch from requesting more nominal threads than the configured budget.
"""
import threading
from contextlib import contextmanager
import config as c

_LOCK=threading.Lock()
_CONDITION=threading.Condition(_LOCK)
_BUSY=0
_BUDGET=None


@contextmanager
def cpu_reservation(n):
    global _BUSY,_BUDGET
    n=int(n)
    if n<1 or n>c.CPU_BUDGET:raise ValueError('Request exceeds configured CPU budget')
    with _CONDITION:
        if _BUDGET is None:_BUDGET=c.CPU_BUDGET
        if _BUDGET!=c.CPU_BUDGET:
            if _BUSY:raise RuntimeError('Cannot change CPU budget while jobs are active')
            _BUDGET=c.CPU_BUDGET
        while _BUSY+n>_BUDGET:
            _CONDITION.wait()
        _BUSY+=n
    try:
        yield
    finally:
        with _CONDITION:
            _BUSY-=n
            _CONDITION.notify_all()
