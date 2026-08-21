from dataclasses import asdict, dataclass
from typing import Any

from src.services.ClaimExtractor import ExtractedClaim
from src.services.ClaimSupportEvaluator import ClaimSupportResult


@dataclass(frozen=True, slots=True)
class CitationEvaluationItem:
    claim_id: str
    source_id: str
    source_exists: bool
    evidence_exists: bool
    document_name_matches: bool
    section_title_matches: bool
    page_number_matches: bool
    chunk_id_matches: bool
    claim_support_passed: bool
    metadata_correct: bool
    correct: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CitationAccuracyReport:
    total_claims: int
    cited_claims: int
    uncited_claims: int
    citation_completeness: float
    total_citation_links: int
    correct_citation_links: int
    incorrect_citation_links: int
    citation_accuracy: float | None
    unique_source_count: int
    invalid_source_ids: tuple[str, ...]
    metadata_accuracy: float | None
    claim_support_accuracy: float | None
    passed: bool
    minimum_citation_accuracy: float
    minimum_citation_completeness: float
    items: tuple[CitationEvaluationItem, ...]


class CitationAccuracyEvaluator:
    """Score citation validity, metadata accuracy, and claim support."""

    def __init__(
        self,
        *,
        minimum_citation_accuracy: float = 0.95,
        minimum_citation_completeness: float = 1.0,
    ) -> None:
        for name, value in (
            ("minimum_citation_accuracy", minimum_citation_accuracy),
            ("minimum_citation_completeness", minimum_citation_completeness),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self._minimum_citation_accuracy = minimum_citation_accuracy
        self._minimum_citation_completeness = minimum_citation_completeness

    @staticmethod
    def _index_by_source_id(
        items: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {
            str(item["source_id"]): item
            for item in items
            if item.get("source_id")
        }

    @staticmethod
    def _claim_results_by_id(
        results: list[ClaimSupportResult],
    ) -> dict[str, ClaimSupportResult]:
        return {result.claim_id: result for result in results}

    @staticmethod
    def _same(left: Any, right: Any) -> bool:
        return left == right

    @classmethod
    def _evaluate_link(
        cls,
        *,
        claim: ExtractedClaim,
        source_id: str,
        source: dict[str, Any] | None,
        evidence: dict[str, Any] | None,
        support_result: ClaimSupportResult | None,
    ) -> CitationEvaluationItem:
        source_exists = source is not None
        evidence_exists = evidence is not None

        document_match = bool(
            source_exists
            and evidence_exists
            and cls._same(
                source.get("document_name"), evidence.get("document_name")
            )
        )
        section_match = bool(
            source_exists
            and evidence_exists
            and cls._same(
                source.get("section_title"), evidence.get("section_title")
            )
        )
        page_match = bool(
            source_exists
            and evidence_exists
            and cls._same(
                source.get("page_number"), evidence.get("page_number")
            )
        )
        chunk_match = bool(
            source_exists
            and evidence_exists
            and cls._same(source.get("chunk_id"), evidence.get("chunk_id"))
        )
        metadata_correct = all(
            (document_match, section_match, page_match, chunk_match)
        )
        support_passed = bool(
            support_result
            and support_result.supported
            and source_id in support_result.evaluated_source_ids
        )
        correct = (
            source_exists
            and evidence_exists
            and metadata_correct
            and support_passed
        )

        if not source_exists:
            reason = "source_id_not_found"
        elif not evidence_exists:
            reason = "evidence_not_found"
        elif not metadata_correct:
            reason = "citation_metadata_mismatch"
        elif not support_passed:
            reason = "claim_support_not_verified"
        else:
            reason = "citation_correct"

        return CitationEvaluationItem(
            claim_id=claim.claim_id,
            source_id=source_id,
            source_exists=source_exists,
            evidence_exists=evidence_exists,
            document_name_matches=document_match,
            section_title_matches=section_match,
            page_number_matches=page_match,
            chunk_id_matches=chunk_match,
            claim_support_passed=support_passed,
            metadata_correct=metadata_correct,
            correct=correct,
            reason=reason,
        )

    def evaluate(
        self,
        *,
        claims: list[ExtractedClaim],
        claim_results: list[ClaimSupportResult],
        sources: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> CitationAccuracyReport:
        source_index = self._index_by_source_id(sources)
        evidence_index = self._index_by_source_id(evidence)
        support_index = self._claim_results_by_id(claim_results)

        cited_claims = sum(bool(claim.cited_source_ids) for claim in claims)
        uncited_claims = len(claims) - cited_claims
        completeness = cited_claims / len(claims) if claims else 1.0

        items: list[CitationEvaluationItem] = []
        invalid_ids: set[str] = set()
        unique_ids: set[str] = set()

        for claim in claims:
            for source_id in claim.cited_source_ids:
                unique_ids.add(source_id)
                item = self._evaluate_link(
                    claim=claim,
                    source_id=source_id,
                    source=source_index.get(source_id),
                    evidence=evidence_index.get(source_id),
                    support_result=support_index.get(claim.claim_id),
                )
                items.append(item)
                if not item.source_exists:
                    invalid_ids.add(source_id)

        total_links = len(items)
        correct_links = sum(item.correct for item in items)
        incorrect_links = total_links - correct_links
        citation_accuracy = (
            correct_links / total_links if total_links else None
        )
        metadata_accuracy = (
            sum(item.metadata_correct for item in items) / total_links
            if total_links
            else None
        )
        claim_support_accuracy = (
            sum(item.claim_support_passed for item in items) / total_links
            if total_links
            else None
        )

        if not claims:
            passed = True
        elif total_links == 0:
            passed = False
        else:
            passed = bool(
                citation_accuracy is not None
                and citation_accuracy >= self._minimum_citation_accuracy
                and completeness >= self._minimum_citation_completeness
            )

        return CitationAccuracyReport(
            total_claims=len(claims),
            cited_claims=cited_claims,
            uncited_claims=uncited_claims,
            citation_completeness=round(completeness, 6),
            total_citation_links=total_links,
            correct_citation_links=correct_links,
            incorrect_citation_links=incorrect_links,
            citation_accuracy=(
                round(citation_accuracy, 6)
                if citation_accuracy is not None
                else None
            ),
            unique_source_count=len(unique_ids),
            invalid_source_ids=tuple(sorted(invalid_ids)),
            metadata_accuracy=(
                round(metadata_accuracy, 6)
                if metadata_accuracy is not None
                else None
            ),
            claim_support_accuracy=(
                round(claim_support_accuracy, 6)
                if claim_support_accuracy is not None
                else None
            ),
            passed=passed,
            minimum_citation_accuracy=self._minimum_citation_accuracy,
            minimum_citation_completeness=(
                self._minimum_citation_completeness
            ),
            items=tuple(items),
        )

    def evaluate_as_dict(self, **kwargs: Any) -> dict[str, Any]:
        return asdict(self.evaluate(**kwargs))
