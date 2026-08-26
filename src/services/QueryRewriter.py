from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    original_query: str
    semantic_query: str
    keyword_query: str
    expansions: tuple[str, ...]


class QueryRewriter:
    """Pass through model-produced retrieval meaning without phrase rules."""

    @staticmethod
    def rewrite(query: str) -> RewrittenQuery:
        normalized = " ".join(str(query).split()).strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return RewrittenQuery(
            original_query=normalized,
            semantic_query=normalized,
            keyword_query=normalized,
            expansions=tuple(),
        )
