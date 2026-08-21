import re
from dataclasses import asdict, dataclass
from typing import Any

from src.services.ClaimExtractor import ClaimExtractor


@dataclass(frozen=True, slots=True)
class CitationComplianceDecision:
    passed: bool
    reason: str
    total_claims: int
    cited_claims: int
    uncited_claims: int
    invalid_source_ids: tuple[str, ...]
    uncited_claim_ids: tuple[str, ...]


class CitationComplianceValidator:
    """Validate that every factual claim cites only available source IDs."""

    _CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

    def __init__(self, claim_extractor: ClaimExtractor) -> None:
        self._claim_extractor = claim_extractor

    def evaluate(
        self,
        answer: str,
        *,
        available_source_ids: set[str],
        refusal_sentences: tuple[str, ...] = (),
    ) -> CitationComplianceDecision:
        claims = self._claim_extractor.extract(
            answer,
            refusal_sentences=refusal_sentences,
        )
        uncited_ids = tuple(
            claim.claim_id for claim in claims if not claim.cited_source_ids
        )
        cited_ids = {
            source_id
            for claim in claims
            for source_id in claim.cited_source_ids
        }
        invalid_ids = tuple(sorted(cited_ids - available_source_ids))
        cited_claims = len(claims) - len(uncited_ids)

        if not claims:
            reason = "no_factual_claims"
            passed = True
        elif invalid_ids:
            reason = "invalid_source_id"
            passed = False
        elif uncited_ids:
            reason = "uncited_claim_detected"
            passed = False
        else:
            reason = "citation_structure_valid"
            passed = True

        return CitationComplianceDecision(
            passed=passed,
            reason=reason,
            total_claims=len(claims),
            cited_claims=cited_claims,
            uncited_claims=len(uncited_ids),
            invalid_source_ids=invalid_ids,
            uncited_claim_ids=uncited_ids,
        )

    def evaluate_as_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return asdict(self.evaluate(*args, **kwargs))
