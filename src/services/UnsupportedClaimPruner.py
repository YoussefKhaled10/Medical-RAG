from typing import Any


class UnsupportedClaimPruner:
    """Keep only supported, cited claims in their original order."""

    @staticmethod
    def prune(
        claims: list[Any],
        support_results: list[Any],
    ) -> str:
        support_by_id = {
            str(result.claim_id): result
            for result in support_results
        }
        supported_sentences: list[str] = []

        for claim in claims:
            result = support_by_id.get(str(claim.claim_id))
            if result is None or not bool(result.supported):
                continue

            source_ids = tuple(
                dict.fromkeys(
                    str(source_id).upper().strip()
                    for source_id in claim.cited_source_ids
                    if str(source_id).strip()
                )
            )
            if not source_ids:
                continue

            sentence = str(claim.text).strip().rstrip(" .!?؟؛")
            if not sentence:
                continue

            citations = "".join(
                f"[{source_id}]"
                for source_id in source_ids
            )
            supported_sentences.append(
                f"{sentence} {citations}."
            )

        return " ".join(supported_sentences).strip()
