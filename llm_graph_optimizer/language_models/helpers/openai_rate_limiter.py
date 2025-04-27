import asyncio
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class _Ticket:
    tokens_needed: int
    event:         asyncio.Event

class OpenAIRateLimiter:
    """ Rate-limiter that enforces RPM & TPM *and* strict FIFO ordering. """

    def __init__(self, rpm: int, tpm: int):
        self.rpm, self.tpm   = float(rpm), float(tpm)
        self._req_bucket     = self.rpm
        self._tok_bucket     = self.tpm
        self._last_ts        = time.perf_counter()

        self._lock    = asyncio.Lock()
        self._queue: deque[_Ticket] = deque()
        self._task = None

    # ---------- public API -------------------------------------------------

    async def acquire(self, tokens_estimate: int = 1) -> None:
        """
        Block until this call reaches the head of the queue *and*
        enough capacity exists.  FIFO is guaranteed.
        """
        await self._maybe_start_refill_task()
        ticket = _Ticket(tokens_estimate, asyncio.Event())

        async with self._lock:
            self._queue.append(ticket)
            # if we're already at head and capacity exists, poke event now
            if ticket is self._queue[0] and self._fits_head_unlocked():
                ticket.event.set()

        await ticket.event.wait()                    # suspend until woken

    async def adjust_tokens(self, delta: int) -> None:
        await self._maybe_start_refill_task()
        """Refund (+) or borrow (-) TPM capacity after the real usage is known."""
        async with self._lock:
            self._tok_bucket = max(-self.tpm, min(self.tpm, self._tok_bucket + delta))

    async def sync_from_headers(self, headers: dict[str, str]) -> None:
        """
        Bring the local buckets *exactly* in line with what the API just told us.

        Call this once, immediately after every successful response
        (or after a 429, if headers are present).
        """
        # grab numbers; int(None) would crash → guard with `or`
        lim_r   = int(headers.get("x-ratelimit-limit-requests"   , 0))
        rem_r   = int(headers.get("x-ratelimit-remaining-requests", -1))
        lim_t   = int(headers.get("x-ratelimit-limit-tokens"     , 0))
        rem_t   = int(headers.get("x-ratelimit-remaining-tokens" , -1))
        # reset_r = float(headers.get("x-ratelimit-reset-requests" , 0))
        # reset_t = float(headers.get("x-ratelimit-reset-tokens"   , 0))

        async with self._lock:
            # 1️⃣ If the account’s limit changed (promotion, downgrade, org-switch…)
            #    …update the bucket size *and* current fill-level.
            if lim_r:
                # scale old balance to new headroom so we don't “lose” credit
                self._req_bucket = rem_r if rem_r >= 0 else min(self._req_bucket, lim_r)
                self.rpm         = float(lim_r)
            if lim_t:
                self._tok_bucket = rem_t if rem_t >= 0 else min(self._tok_bucket, lim_t)
                self.tpm         = float(lim_t)

            # 3️⃣ If the new head-of-queue fits, wake it
            if self._queue and self._fits_head_unlocked():
                self._queue[0].event.set()

    # ---------- internals --------------------------------------------------

    async def _maybe_start_refill_task(self) -> None:
        """Start the refill loop exactly once, inside the current running loop."""
        if self._task is None:
            loop     = asyncio.get_running_loop()    # always succeeds inside async code
            self._task = loop.create_task(
                self._refill_loop(), name="openai-rate-refill"
            )

    async def _refill_loop(self):
        step = 0.05               # 50 ms
        while True:
            await asyncio.sleep(step)
            async with self._lock:
                self._replenish_buckets_unlocked()
                if self._queue and self._fits_head_unlocked():
                    self._queue[0].event.set()       # wake the head

    def _replenish_buckets_unlocked(self):
        now     = time.perf_counter()
        elapsed = now - self._last_ts
        self._last_ts = now
        self._req_bucket = min(self.rpm, self._req_bucket + self.rpm * elapsed / 60)
        self._tok_bucket = min(self.tpm, self._tok_bucket + self.tpm * elapsed / 60)

    def _fits_head_unlocked(self) -> bool:
        """Does the head-of-queue request fit?  If yes, debit & pop it."""
        head = self._queue[0]
        if self._req_bucket >= 1 and self._tok_bucket >= head.tokens_needed:
            # debit
            self._req_bucket -= 1
            self._tok_bucket -= head.tokens_needed
            # remove ticket so the next one becomes head
            self._queue.popleft()
            return True
        return False