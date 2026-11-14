"""BudgetGuard — per-run API call counter with hard abort.

Usage
-----
    budget = BudgetGuard(max_calls=40)
    budget.charge("tts")          # raises BudgetExceeded if limit hit
    budget.charge("stt")

The guard is intentionally simple and NOT thread-safe (single-threaded pipeline).
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its configured API call limit."""

    def __init__(self, max_calls: int, used: int, kind: str) -> None:
        self.max_calls = max_calls
        self.used = used
        self.kind = kind
        super().__init__(
            f"Budget exceeded: attempted {kind!r} call but already used "
            f"{used}/{max_calls} allowed calls. Aborting to protect API credits."
        )


class BudgetGuard:
    """Counts API calls and raises :class:`BudgetExceeded` at the limit.

    Parameters
    ----------
    max_calls:
        Maximum total API calls (TTS + STT combined) for this run.
    """

    def __init__(self, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        self.max_calls = max_calls
        self._used = 0
        self._tts_calls = 0
        self._stt_calls = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def used(self) -> int:
        """Total API calls charged so far."""
        return self._used

    @property
    def tts_calls(self) -> int:
        return self._tts_calls

    @property
    def stt_calls(self) -> int:
        return self._stt_calls

    @property
    def remaining(self) -> int:
        return self.max_calls - self._used

    def charge(self, kind: str = "api") -> None:
        """Record one API call; raise :class:`BudgetExceeded` if the limit is reached.

        Parameters
        ----------
        kind:
            Human-readable label for the error message: ``"tts"`` or ``"stt"``.
        """
        if self._used >= self.max_calls:
            raise BudgetExceeded(self.max_calls, self._used, kind)
        self._used += 1
        if kind == "tts":
            self._tts_calls += 1
        elif kind == "stt":
            self._stt_calls += 1

    def summary(self) -> str:
        return (
            f"API calls used: {self._used}/{self.max_calls} "
            f"(TTS={self._tts_calls}, STT={self._stt_calls})"
        )

    def __repr__(self) -> str:
        return (
            f"BudgetGuard(max_calls={self.max_calls}, used={self._used}, "
            f"tts={self._tts_calls}, stt={self._stt_calls})"
        )

