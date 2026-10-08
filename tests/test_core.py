"""Tests for ShortId.

Every test is deterministic. We never call time.monotonic directly; we inject
a fake clock whose return value we control. No sleeps, no wall-clock asserts.
"""

import unittest

from short_id import ShortId, ShortIdError, OverflowError
from short_id.core import _ID_LEN, _COUNTER_MAX, _TS_MAX, _RAND_MAX


class _FakeClock:
    """A clock whose time only moves when we tell it to."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance_ms(self, ms: int) -> None:
        self.t += ms / 1000.0


class _SeqRandom:
    """Deterministic random source: returns a fixed sequence then repeats."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self) -> float:
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return v


class TestEncoding(unittest.TestCase):
    def test_id_length_is_constant(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        for _ in range(1000):
            self.assertEqual(len(gen.next_id()), _ID_LEN)

    def test_alphabet_is_base32hex(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUV")
        for _ in range(100):
            self.assertTrue(set(gen.next_id()) <= allowed)

    def test_zero_payload_encodes_to_all_zeros(self):
        # With a zero timestamp, zero counter, zero random, every 5-bit group
        # is zero -> the all-'0' string.
        self.assertEqual(ShortId._encode(0), "0" * _ID_LEN)

    def test_encode_decode_roundtrip(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        sid = gen.next_id()
        decoded = gen.decode(sid)
        self.assertEqual(decoded["timestamp_ms"], 0)
        self.assertEqual(decoded["counter"], 0)
        self.assertEqual(decoded["random"], 0)


class TestOrdering(unittest.TestCase):
    def test_ids_from_one_instance_sort_by_creation_time(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        ids = []
        for i in range(50):
            ids.append(gen.next_id())
            clk.advance_ms(1)
        # Lexicographic order == creation order, because the timestamp is the
        # high bits and base32hex sorts numerically.
        self.assertEqual(ids, sorted(ids))

    def test_ids_within_same_ms_still_increment(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        a = gen.next_id()
        b = gen.next_id()  # same ms
        c = gen.next_id()  # same ms
        self.assertEqual(gen.decode(a)["counter"], 0)
        self.assertEqual(gen.decode(b)["counter"], 1)
        self.assertEqual(gen.decode(c)["counter"], 2)
        self.assertLess(a, b)
        self.assertLess(b, c)


class TestCounterRollover(unittest.TestCase):
    def test_counter_wraps_and_timestamp_advances(self):
        clk = _FakeClock()
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        # Set up internal state as if we had already issued IDs up to
        # _COUNTER_MAX - 1 within this millisecond, so the next call lands on
        # _COUNTER_MAX and the one after triggers rollover. The constructor
        # resets the counter to 0 on each new millisecond, so seeding via
        # random_source alone cannot land mid-range; we set the state directly.
        gen._last_ts = 0
        gen._counter = _COUNTER_MAX - 1
        first = gen.next_id()
        self.assertEqual(gen.decode(first)["counter"], _COUNTER_MAX)
        # Next call rolls the counter over.
        second = gen.next_id()
        self.assertEqual(gen.decode(second)["counter"], 0)
        # Timestamp must have advanced by 1 ms to preserve ordering.
        self.assertEqual(
            gen.decode(second)["timestamp_ms"],
            gen.decode(first)["timestamp_ms"] + 1,
        )
        self.assertLess(first, second)


class TestClockEdges(unittest.TestCase):
    def test_clock_going_backwards_is_clamped(self):
        clk = _FakeClock(start=0.005)  # 5 ms
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        a = gen.next_id()  # ts=5
        clk.t = 0.003      # clock jumps back
        b = gen.next_id()  # should clamp to ts=5
        self.assertEqual(gen.decode(b)["timestamp_ms"], 5)
        self.assertEqual(gen.decode(b)["counter"], 1)
        self.assertLess(a, b)

    def test_negative_clock_is_treated_as_zero(self):
        clk = _FakeClock(start=-1.0)
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        sid = gen.next_id()
        self.assertEqual(gen.decode(sid)["timestamp_ms"], 0)

    def test_clock_overflow_raises(self):
        clk = _FakeClock(start=(_TS_MAX + 1) / 1000.0)
        gen = ShortId(clock=clk, random_source=_SeqRandom([0.0]))
        with self.assertRaises(OverflowError):
            gen.next_id()


class TestDecodeValidation(unittest.TestCase):
    def test_wrong_length_rejected(self):
        gen = ShortId(clock=_FakeClock(), random_source=_SeqRandom([0.0]))
        with self.assertRaises(ShortIdError):
            gen.decode("0" * 14)
        with self.assertRaises(ShortIdError):
            gen.decode("0" * 16)

    def test_invalid_char_rejected(self):
        gen = ShortId(clock=_FakeClock(), random_source=_SeqRandom([0.0]))
        # 'W' is not in base32hex (alphabet ends at 'V').
        with self.assertRaises(ShortIdError):
            gen.decode("0" * 14 + "W")

    def test_non_string_rejected(self):
        gen = ShortId(clock=_FakeClock(), random_source=_SeqRandom([0.0]))
        with self.assertRaises(ShortIdError):
            gen.decode(123)


class TestRandomClamping(unittest.TestCase):
    def test_random_source_returning_one_is_clamped(self):
        # A custom source might return exactly 1.0; we must not produce an
        # out-of-range random value (which would corrupt the payload layout).
        gen = ShortId(clock=_FakeClock(), random_source=_SeqRandom([1.0, 1.0]))
        sid = gen.next_id()
        decoded = gen.decode(sid)
        self.assertLessEqual(decoded["random"], _RAND_MAX)

    def test_random_source_returning_negative_is_clamped(self):
        gen = ShortId(clock=_FakeClock(), random_source=_SeqRandom([-5.0, -5.0]))
        sid = gen.next_id()
        decoded = gen.decode(sid)
        self.assertGreaterEqual(decoded["random"], 0)


if __name__ == "__main__":
    unittest.main()
