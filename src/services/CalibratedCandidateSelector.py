from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateSelectionSummary:
    input_count: int
    selected_count: int
    absolute_threshold: float
    relative_to_top_ratio: float
    removed_chunk_ids: tuple[str, ...]


class CalibratedCandidateSelector:
    def __init__(self, absolute_threshold: float = 0.0, relative_to_top_ratio: float = 0.0, minimum_results: int = 1) -> None:
        if not 0 <= absolute_threshold <= 1 or not 0 <= relative_to_top_ratio <= 1 or minimum_results < 1:
            raise ValueError("Candidate selector settings are invalid")
        self.absolute_threshold = absolute_threshold
        self.relative_to_top_ratio = relative_to_top_ratio
        self.minimum_results = minimum_results

    def select(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], CandidateSelectionSummary]:
        if not results:
            return [], CandidateSelectionSummary(0, 0, self.absolute_threshold, self.relative_to_top_ratio, ())
        top = float(results[0].get("rerank_score") or 0)
        floor = max(self.absolute_threshold, top * self.relative_to_top_ratio)
        selected = [item for item in results if float(item.get("rerank_score") or 0) >= floor]
        if len(selected) < self.minimum_results:
            selected = results[:self.minimum_results]
        selected_ids = {str(item.get("chunk_id")) for item in selected}
        removed = tuple(str(item.get("chunk_id")) for item in results if str(item.get("chunk_id")) not in selected_ids)
        return selected, CandidateSelectionSummary(len(results), len(selected), self.absolute_threshold, self.relative_to_top_ratio, removed)
