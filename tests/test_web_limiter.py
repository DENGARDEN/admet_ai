"""Unit tests for web rate limiting and FIFO prediction queue."""

import importlib.util
from pathlib import Path
import threading
import time

import pytest

LIMITER_PATH = Path(__file__).resolve().parents[1] / "admet_ai" / "web" / "app" / "limiter.py"
SPEC = importlib.util.spec_from_file_location("admet_ai_web_limiter", LIMITER_PATH)
assert SPEC is not None and SPEC.loader is not None
limiter_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(limiter_module)

PredictionQueue = limiter_module.PredictionQueue
QueueFullError = limiter_module.QueueFullError
QueueTimeoutError = limiter_module.QueueTimeoutError
RateLimiter = limiter_module.RateLimiter


def test_rate_limiter_enforces_per_client_window() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=1)

    allowed, retry = limiter.allow("user_a")
    assert allowed and retry == 0.0

    allowed, retry = limiter.allow("user_a")
    assert allowed and retry == 0.0

    allowed, retry = limiter.allow("user_a")
    assert not allowed
    assert retry > 0.0

    # Separate client should remain unaffected
    allowed, retry = limiter.allow("user_b")
    assert allowed and retry == 0.0

    time.sleep(1.05)
    allowed, retry = limiter.allow("user_a")
    assert allowed and retry == 0.0


def test_rate_limiter_disabled_when_none() -> None:
    limiter = RateLimiter(max_requests=None, window_seconds=60)
    for _ in range(20):
        allowed, retry = limiter.allow("user_a")
        assert allowed and retry == 0.0


def test_prediction_queue_serializes_concurrent_workers_in_fifo_order() -> None:
    queue = PredictionQueue(max_queue_size=10, timeout=5.0)
    active_count = 0
    max_active_seen = 0
    execution_order: list[int] = []
    state_lock = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()

    def run_worker(worker_id: int, wait_for_release: bool = False) -> None:
        nonlocal active_count, max_active_seen
        with queue.slot():
            with state_lock:
                active_count += 1
                max_active_seen = max(max_active_seen, active_count)
                execution_order.append(worker_id)
            if wait_for_release:
                first_entered.set()
                release_first.wait(timeout=2.0)
            else:
                time.sleep(0.02)
            with state_lock:
                active_count -= 1

    t0 = threading.Thread(target=run_worker, args=(0, True))
    t0.start()
    assert first_entered.wait(timeout=2.0)

    queued_threads = []
    for wid in range(1, 5):
        t = threading.Thread(target=run_worker, args=(wid,))
        t.start()
        queued_threads.append(t)
        time.sleep(0.01)

    release_first.set()
    t0.join(timeout=2.0)
    for t in queued_threads:
        t.join(timeout=2.0)

    assert max_active_seen == 1
    assert execution_order == [0, 1, 2, 3, 4]


def test_prediction_queue_rejects_when_full() -> None:
    queue = PredictionQueue(max_queue_size=1, timeout=5.0)
    first_entered = threading.Event()
    second_queued = threading.Event()
    release_first = threading.Event()

    def hold_slot() -> None:
        with queue.slot():
            first_entered.set()
            release_first.wait(timeout=2.0)

    def wait_slot() -> None:
        second_queued.set()
        with queue.slot():
            pass

    t0 = threading.Thread(target=hold_slot)
    t0.start()
    assert first_entered.wait(timeout=2.0)

    t1 = threading.Thread(target=wait_slot)
    t1.start()
    assert second_queued.wait(timeout=2.0)
    time.sleep(0.05)

    with pytest.raises(QueueFullError):
        with queue.slot():
            pass

    release_first.set()
    t0.join(timeout=2.0)
    t1.join(timeout=2.0)


def test_prediction_queue_times_out() -> None:
    queue = PredictionQueue(max_queue_size=2, timeout=0.1)
    first_entered = threading.Event()
    release_first = threading.Event()

    def hold_slot() -> None:
        with queue.slot():
            first_entered.set()
            release_first.wait(timeout=2.0)

    t0 = threading.Thread(target=hold_slot)
    t0.start()
    assert first_entered.wait(timeout=2.0)

    try:
        with pytest.raises(QueueTimeoutError):
            with queue.slot():
                pass
    finally:
        release_first.set()
        t0.join(timeout=2.0)
