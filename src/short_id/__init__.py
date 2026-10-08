"""Short ID: sortable, collision-resistant identifiers from a monotonic clock."""

from short_id.core import ShortId, ShortIdError, OverflowError as _Overflow

__all__ = ["ShortId", "ShortIdError", "OverflowError"]

# Re-export under a stable name without shadowing the builtin so callers can
# `from short_id import ShortId, OverflowError` and get our sentinel type.
OverflowError = _Overflow
