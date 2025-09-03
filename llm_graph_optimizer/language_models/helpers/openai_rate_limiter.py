import asyncio
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class _Ticket:
    tokens_needed: int
    event:         asyncio.Event

class OpenAIRateLimiter:
    """
    A rate-limiter that enforces requests per minute (RPM) and tokens per minute (TPM) limits,
    while maintaining strict FIFO ordering for queued requests.

    This class ensures that API calls adhere to rate limits imposed by OpenAI,
    and provides mechanisms to synchronize local state with server-provided rate-limit headers.
    """

    def __init__(self, rpm: int, tpm: int, max_estimated_response_tokens: int = 1000):
        """
        Initialize the rate-limiter with specified RPM and TPM limits.

        :param rpm: Maximum number of requests per minute.
        :param tpm: Maximum number of tokens per minute.
        """
        self.rpm, self.tpm   = float(rpm), float(tpm)
        self._req_bucket     = self.rpm
        self._tok_bucket     = self.tpm
        self._last_ts        = time.perf_counter()

        self._lock    = asyncio.Lock()
        self._queue: deque[_Ticket] = deque()
        self._task = None
        self._max_estimated_response_tokens = max_estimated_response_tokens
    # ---------- public API -------------------------------------------------

    async def acquire(self, tokens_estimate: int = 1) -> None:
        """
        Block until the request reaches the head of the queue and sufficient capacity exists.

        This method ensures strict FIFO ordering for queued requests and checks
        whether the required tokens and request capacity are available before proceeding.

        :param tokens_estimate: Estimated number of tokens required for the request.
        """
        await self._maybe_start_refill_task()
        ticket = _Ticket(tokens_estimate, asyncio.Event())

        async with self._lock:
            self._queue.append(ticket)
            # If the ticket is at the head of the queue and capacity exists, signal readiness
            if ticket is self._queue[0] and self._fits_head_unlocked():
                ticket.event.set()

        await ticket.event.wait()  # Suspend until the ticket is processed

    async def adjust_tokens(self, delta: int) -> None:
        """
        Adjust the token bucket capacity based on actual usage.

        This method allows for refunding or borrowing token capacity after the
        real usage is known, ensuring accurate tracking of token limits.

        :param delta: The adjustment value. Positive values refund tokens, negative values borrow tokens.
        """
        await self._maybe_start_refill_task()
        async with self._lock:
            self._tok_bucket = max(-self.tpm, min(self.tpm, self._tok_bucket + delta))

    async def sync_from_headers(self, headers: dict[str, str]) -> None:
        """
        Synchronize local rate-limit buckets with server-provided headers.

        This method updates the RPM and TPM limits and current bucket levels
        based on the headers returned by the OpenAI API. It should be called
        after every successful response or after a 429 error if headers are present.

        :param headers: A dictionary containing rate-limit headers from the API response.
        """
        try:
            lim_r = int(headers.get("x-ratelimit-limit-requests", None))
            rem_r = int(headers.get("x-ratelimit-remaining-requests", None))
            lim_t = int(headers.get("x-ratelimit-limit-tokens", None))
            rem_t = int(headers.get("x-ratelimit-remaining-tokens", None))
        except Exception:
            return

        async with self._lock:
            # Update bucket size and current fill-level if limits have changed
            if lim_r and rem_r:
                self._req_bucket = rem_r if rem_r >= 0 else min(self._req_bucket, lim_r)
                self.rpm = float(lim_r)
            if lim_t and rem_t:
                self._tok_bucket = rem_t if rem_t >= 0 else min(self._tok_bucket, lim_t)
                self.tpm = float(lim_t)

            # Wake the head of the queue if it fits
            if self._queue and self._fits_head_unlocked():
                self._queue[0].event.set()

    # ---------- internals --------------------------------------------------

    async def _maybe_start_refill_task(self) -> None:
        """
        Start the refill loop if it is not already running.

        This method ensures that the refill loop is started exactly once
        within the current running event loop.
        """
        if self._task is None:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(
                self._refill_loop(), name="openai-rate-refill"
            )

    async def _refill_loop(self):
        """
        Continuously replenish the request and token buckets.

        This loop runs in the background and periodically refills the buckets
        based on the elapsed time and configured RPM and TPM limits.
        """
        step = 0.05  # Refill interval in seconds
        while True:
            await asyncio.sleep(step)
            async with self._lock:
                self._replenish_buckets_unlocked()
                if self._queue and self._fits_head_unlocked():
                    self._queue[0].event.set()  # Wake the head of the queue

    def _replenish_buckets_unlocked(self):
        """
        Replenish the request and token buckets based on elapsed time.

        This method calculates the refill amount for both buckets and updates
        their levels accordingly.
        """
        now = time.perf_counter()
        elapsed = now - self._last_ts
        self._last_ts = now
        self._req_bucket = min(self.rpm, self._req_bucket + self.rpm * elapsed / 60)
        self._tok_bucket = min(self.tpm, self._tok_bucket + self.tpm * elapsed / 60)

    def _fits_head_unlocked(self) -> bool:
        """
        Check if the head-of-queue request fits within the current bucket capacity.

        If the request fits, this method debits the required capacity and removes
        the ticket from the queue.

        :return: True if the request fits, False otherwise.
        """
        head = self._queue[0]
        if self._req_bucket >= 1 and self._tok_bucket >= head.tokens_needed:
            self._req_bucket -= 1
            self._tok_bucket -= head.tokens_needed
            self._queue.popleft()
            return True
        return False