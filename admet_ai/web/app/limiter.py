"""Thread-safe rate limiting and FIFO prediction queue for the ADMET-AI web server."""

from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import time
from typing import Iterator


class QueueFullError(Exception):
    """Raised when the prediction queue has reached its maximum capacity."""


class QueueTimeoutError(Exception):
    """Raised when a queued prediction request times out waiting for its turn."""


class RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by client identifier."""

    def __init__(self, max_requests: int | None = 10, window_seconds: int = 60) -> None:
        """Initialize the rate limiter.

        :param max_requests: Maximum number of requests allowed per client within the window.
                             If None or <= 0, rate limiting is disabled.
        :param window_seconds: Sliding time window in seconds. If <= 0, rate limiting is disabled.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._check_count = 0

    def configure(self, max_requests: int | None, window_seconds: int) -> None:
        """Update rate limiter settings and reset tracked request timestamps.

        :param max_requests: Maximum requests allowed per window, or None to disable.
        :param window_seconds: Sliding time window in seconds.
        """
        with self._lock:
            self.max_requests = max_requests
            self.window_seconds = window_seconds
            self._requests.clear()
            self._check_count = 0

    def allow(self, key: str) -> tuple[bool, float]:
        """Check whether a request from the given client key is allowed.

        :param key: Client identifier (e.g., IP address or session ID).
        :return: Tuple of (allowed, retry_after_seconds).
        """
        with self._lock:
            if self.max_requests is None or self.max_requests <= 0 or self.window_seconds <= 0:
                return True, 0.0

            now = time.monotonic()
            cutoff = now - self.window_seconds

            self._check_count += 1
            if self._check_count % 100 == 0:
                stale_keys = [
                    k for k, timestamps in self._requests.items() if not timestamps or timestamps[-1] <= cutoff
                ]
                for stale_key in stale_keys:
                    del self._requests[stale_key]

            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = max(0.0, timestamps[0] + self.window_seconds - now)
                return False, retry_after

            timestamps.append(now)
            return True, 0.0


class PredictionQueue:
    """Thread-safe strict FIFO queue ensuring single-request execution on shared model state."""

    def __init__(self, max_queue_size: int | None = 10, timeout: float | None = 120.0) -> None:
        """Initialize the prediction queue.

        :param max_queue_size: Maximum number of waiting requests allowed in the queue.
                               If None, queue size is unbounded.
        :param timeout: Maximum time in seconds a request may wait in the queue.
                        If None, waits indefinitely.
        """
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self._condition = threading.Condition()
        self._active = False
        self._wait_queue: deque[int] = deque()
        self._next_ticket = 0

    def configure(self, max_queue_size: int | None, timeout: float | None) -> None:
        """Update queue capacity and timeout settings.

        :param max_queue_size: Maximum waiting requests allowed, or None for unbounded.
        :param timeout: Maximum wait time in seconds, or None for no timeout.
        """
        with self._condition:
            self.max_queue_size = max_queue_size
            self.timeout = timeout

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Acquire exclusive access to run a prediction in strict FIFO order.

        :raises QueueFullError: If the queue is already at max_queue_size.
        :raises QueueTimeoutError: If waiting exceeds timeout seconds.
        """
        with self._condition:
            if not self._active and not self._wait_queue:
                self._active = True
            else:
                if self.max_queue_size is not None and len(self._wait_queue) >= self.max_queue_size:
                    raise QueueFullError("Prediction queue is full.")

                ticket = self._next_ticket
                self._next_ticket += 1
                self._wait_queue.append(ticket)
                deadline = None if self.timeout is None else time.monotonic() + self.timeout

                try:
                    while self._active or self._wait_queue[0] != ticket:
                        remaining: float | None = None
                        if deadline is not None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise QueueTimeoutError("Timed out waiting in prediction queue.")
                        self._condition.wait(timeout=remaining)
                except BaseException:
                    try:
                        self._wait_queue.remove(ticket)
                    except ValueError:
                        pass
                    self._condition.notify_all()
                    raise

                self._wait_queue.popleft()
                self._active = True

        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()
