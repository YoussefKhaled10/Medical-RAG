from dataclasses import asdict, dataclass
from typing import Any

from src.services.ClaimSupportEvaluator import ClaimSupportResult


@dataclass(frozen=True, slots=True)
class PostGenerationSafetyDecision:
    passed: bool
    reason: str
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    faithfulness: float
    minimum_faithfulness: float
    unsupported_claim_ids: tuple[str, ...]


class PostGenerationSafetyGate:
    """Block an answer when claim-level evidence support is insufficient."""

    def __init__(
        self,
        *,
        minimum_faithfulness: float = 0.90,
        block_on_any_unsupported_claim: bool = True,
    ) -> None:
        if not 0.0 <= minimum_faithfulness <= 1.0:
            raise ValueError(
                "minimum_faithfulness must be between 0 and 1"
            )
        self._minimum_faithfulness = minimum_faithfulness
        self._block_on_any_unsupported_claim = (
            block_on_any_unsupported_claim
        )

    def evaluate(
        self,
        results: list[ClaimSupportResult],
    ) -> PostGenerationSafetyDecision:
        total = len(results)
        supported = sum(result.supported for result in results)
        unsupported_ids = tuple(
            result.claim_id for result in results if not result.supported
        )
        unsupported = len(unsupported_ids)
        faithfulness = supported / total if total else 1.0

        below_target = faithfulness < self._minimum_faithfulness
        blocked_by_claim = (
            self._block_on_any_unsupported_claim and unsupported > 0
        )
        passed = not below_target and not blocked_by_claim

        if total == 0:
            reason = "no_factual_claims"
        elif passed:
            reason = "all_claims_supported"
        elif blocked_by_claim:
            reason = "unsupported_claim_detected"
        else:
            reason = "faithfulness_below_threshold"

        return PostGenerationSafetyDecision(
            passed=passed,
            reason=reason,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            faithfulness=round(faithfulness, 6),
            minimum_faithfulness=self._minimum_faithfulness,
            unsupported_claim_ids=unsupported_ids,
        )

    def evaluate_as_dict(
        self,
        results: list[ClaimSupportResult],
    ) -> dict[str, Any]:
        return asdict(self.evaluate(results))
