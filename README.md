# Short ID

Sortable, collision-resistant identifiers generated from a monotonic clock.

```python
from short_id import ShortId

gen = ShortId()
sid = gen.next_id()           # e.g. "000000000000000"
parts = gen.decode(sid)        # {"timestamp_ms": 0, "counter": 0, "random": 0}
```

## Why

Distributed systems need IDs that are cheap to generate, sort by creation
time, and don't collide across processes. UUIDs sort randomly and are long;
auto-increment requires coordination. This library packs a 43-bit millisecond
timestamp (from a monotonic clock, not wall time), a 20-bit counter, and a
12-bit random value into 15 base32hex characters. IDs from a single instance
sort lexicographically by creation time; the random tail reduces cross-instance
collisions without any coordination.

The trade-off: cross-instance ordering is *not* guaranteed. Two instances
created in the same millisecond may produce IDs that sort in either order,
because the random bits occupy the low positions. If you need strict global
ordering, use a coordinating authority instead.

## Edge cases

- **Clock going backwards.** `time.monotonic` should never do this, but a
  custom clock might. The generator clamps the timestamp to the last seen
  value and continues from there, preserving per-instance ordering.
- **Counter rollover within a millisecond.** After 2^20 IDs in one ms, the
  counter wraps to zero and the visible timestamp advances by one ms. This
  keeps IDs monotonic at the cost of a tiny timestamp drift under extreme
  load.
- **Timestamp overflow.** The 43-bit field supports ~278 years from
  2020-01-01. If the clock exceeds that range, `next_id()` raises
  `OverflowError` (a subclass of `ShortIdError`), rather than silently
  wrapping.

The library takes a `clock` callable in its constructor so tests can inject a
fake. Never assert on wall-clock time in your own tests; pass a fake clock.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

