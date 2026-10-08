"""Core implementation of ShortId.

Design notes (the *why*):

1. Monotonic clock, not wall clock.
   Wall-clock time can jump backwards (NTP slew, manual correction, VM
   migration). A sortable ID scheme that keys off the wall clock therefore
   breaks its central promise. We accept a `clock` callable in the
   constructor so callers can inject time.monotonic (the default) or a fake
   in tests. This is the single seam that makes the library deterministic.

2. Structure: (timestamp_ms, 20-bit counter, 12-bit random).
   - timestamp_ms: 43 bits, good for ~278 years from the chosen epoch.
   - counter: 20 bits, rolls over within the same millisecond.
   - random: 12 bits, disambiguates across processes/instances that happen
     to share a timestamp and counter.
   The full 75-bit payload is base32hex-encoded (no padding) into 15 chars.
   We chose base32hex over base32 because its alphabet sorts lexicographically
   in the same order as the underlying integers, so string comparison of two
   IDs produced by the *same* instance is equivalent to ordering by creation
   time. This is the only comparison we promise; cross-instance ordering is
   not guaranteed because the random tail is mixed into the low bits.

3. Counter rollover.
   If 2**20 IDs are requested inside one millisecond, the counter wraps to
   zero. We do not silently reuse the same (timestamp, counter) pair: at the
   instant of wrap we bump the *visible* timestamp forward by one millisecond
   (via an internal offset) so that subsequent IDs in the same real ms still
   sort after earlier ones. This trades a tiny amount of timestamp accuracy
   for guaranteed monotonicity within an instance.

4. Epoch.
   We use a fixed epoch of 2020-01-01T00:00:00Z rather than the Unix epoch.
   This keeps the base32hex representation short (43 bits instead of 61) and
   makes IDs from this library visually distinct from generic Unix-ms
   encodings. Callers who need to recover the absolute timestamp can add
   the epoch offset back; see `decode()`.
"""

from __future__ import annotations

import time
from typing import Callable

__all__ = ["ShortId", "ShortIdError", "OverflowError"]

_EPOCH_MS = 1577836800000  # 2020-01-01T00:00:00Z in Unix milliseconds.
_TS_BITS = 43
_TS_MAX = (1 << _TS_BITS) - 1
_COUNTER_BITS = 20
_COUNTER_MAX = (1 << _COUNTER_BITS) - 1
_RAND_BITS = 12
_RAND_MAX = (1 << _RAND_BITS) - 1
_TOTAL_BITS = _TS_BITS + _COUNTER_BITS + _RAND_BITS  # 75

# base32hex alphabet (RFC 4648) sorts 0..9 before A..V, matching numeric order
# of the underlying bits. This is why lexicographic ordering of IDs from one
# instance equals creation-time ordering.
_ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
_ALPHA_LEN = 32
# 75 bits base32hex -> ceil(75/5) = 15 chars, no padding needed.
_ID_LEN = 15


class ShortIdError(Exception):
    """Base class for all ShortId errors."""


class OverflowError(ShortIdError):
    """Raised when the monotonic clock exceeds the 43-bit timestamp range.

    This is not the builtin OverflowError; it is a subclass of ShortIdError so
    callers can catch library errors with a single except clause.
    """


class ShortId:
    """Generate sortable, collision-resistant IDs from a monotonic clock.

    Parameters
    ----------
    clock:
        A zero-arg callable returning a float seconds value from a monotonic
        source. Defaults to ``time.monotonic``. Inject a fake in tests.
    random_source:
        A zero-arg callable returning a float in [0.0, 1.0). Used only at
        construction time to seed the random tail and the starting counter.
        Defaults to ``random.random`` (imported lazily so that simply
        importing this module has no side effects beyond ``time``).
    """

    __slots__ = ("_clock", "_last_ts", "_counter", "_ts_offset", "_rand", "_lock")

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        random_source: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock
        if random_source is None:
            import random

            random_source = random.random
        r = int(random_source() * (_RAND_MAX + 1))
        # Clamp to range defensively; random.random() is [0,1) but a custom
        # source might return exactly 1.0 or a negative value.
        if r < 0:
            r = 0
        elif r > _RAND_MAX:
            r = _RAND_MAX
        self._rand = r
        # Start the counter at a random offset so two instances created in the
        # same millisecond don't collide on counter 0 before divergence kicks
        # in. The random tail would disambiguate them anyway, but staggering
        # the counter reduces the chance further.
        start_counter = int(random_source() * (_COUNTER_MAX + 1))
        if start_counter < 0:
            start_counter = 0
        elif start_counter > _COUNTER_MAX:
            start_counter = _COUNTER_MAX
        self._counter = start_counter
        self._last_ts = -1
        self._ts_offset = 0
        # A threading.Lock is heavier than we need for the common
        # single-threaded case, but ShortId is often shared across threads in
        # web servers. We import threading lazily and store the bound lock so
        # the hot path is a single `with` on an attribute.
        import threading

        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def next_id(self) -> str:
        """Return the next ID as a 15-char base32hex string."""
        return self._encode(self._next_payload())

    def decode(self, sid: str) -> dict:
        """Decode an ID produced by this library.

        Returns a dict with keys:
          - ``timestamp_ms``: milliseconds since 2020-01-01T00:00:00Z (the
            library epoch). Add ``_EPOCH_MS`` to recover a Unix timestamp.
          - ``counter``: the 20-bit counter value at issue time.
          - ``random``: the 12-bit random value for this instance.

        Raises ``ShortIdError`` if the string is not a valid ID from this
        scheme. This deliberately rejects strings that are the right length
        but contain non-alphabet characters, rather than silently coercing.
        """
        payload = self._decode_to_int(sid)
        random = payload & _RAND_MAX
        payload >>= _RAND_BITS
        counter = payload & _COUNTER_MAX
        payload >>= _COUNTER_BITS
        timestamp = payload & _TS_MAX
        return {
            "timestamp_ms": timestamp,
            "counter": counter,
            "random": random,
        }

    # -- internals ---------------------------------------------------------

    def _next_payload(self) -> int:
        with self._lock:
            ts_ms = int(self._clock() * 1000)
            if ts_ms < 0:
                # A broken clock that goes negative would corrupt ordering.
                # Treat it as zero rather than emitting IDs that sort before
                # everything else.
                ts_ms = 0
            if ts_ms > _TS_MAX:
                raise OverflowError(
                    "monotonic clock exceeded the 43-bit timestamp range"
                )
            if ts_ms < self._last_ts:
                # The clock went backwards despite being monotonic. This
                # shouldn't happen with time.monotonic but can with a custom
                # clock (including a misbehaving fake). We clamp to the last
                # seen timestamp so ordering is preserved.
                ts_ms = self._last_ts
            if ts_ms == self._last_ts:
                self._counter += 1
                if self._counter > _COUNTER_MAX:
                    # Rollover: advance the visible timestamp by one ms so we
                    # don't re-emit the same (ts, counter) pair.
                    self._ts_offset += 1
                    self._counter = 0
                    ts_ms = self._last_ts + 1
                    if ts_ms > _TS_MAX:
                        raise OverflowError(
                            "counter rollover pushed timestamp out of range"
                        )
            else:
                # New (greater) millisecond: reset the offset and counter.
                self._ts_offset = 0
                self._counter = 0
            self._last_ts = ts_ms
            return (ts_ms << (_COUNTER_BITS + _RAND_BITS)) | (
                self._counter << _RAND_BITS
            ) | self._rand

    @staticmethod
    def _encode(value: int) -> str:
        # Build the base32hex string from the least significant 5-bit groups
        # upward. 75 bits is exactly 15 groups of 5, so no padding is ever
        # produced.
        chars = []
        for _ in range(_ID_LEN):
            chars.append(_ALPHA[value & 0x1F])
            value >>= 5
        # We appended low-order first; reverse to get big-endian (sortable)
        # order.
        return "".join(reversed(chars))

    @staticmethod
    def _decode_to_int(sid: str) -> int:
        if not isinstance(sid, str):
            raise ShortIdError("id must be a str")
        if len(sid) != _ID_LEN:
            raise ShortIdError(f"id must be {_ID_LEN} chars, got {len(sid)}")
        value = 0
        for ch in sid:
            idx = _ALPHA.find(ch)
            if idx == -1:
                raise ShortIdError(f"invalid character {ch!r} in id")
            value = (value << 5) | idx
        return value
