import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.services.ClaimExtractor import ExtractedClaim


class SupportJudge(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class ClaimSupportResult:
    claim_id: str
    claim: str
    cited_source_ids: tuple[str, ...]
    evaluated_source_ids: tuple[str, ...]
    supported: bool
    support_score: float
    reason: str


class ClaimSupportEvaluator:
    """Evaluate whether each claim is entailed by its cited evidence only."""

    _CODE_FENCE_PATTERN = re.compile(
        r"```(?:json)?\s*(.*?)\s*```",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        judge: SupportJudge,
        *,
        support_threshold: float = 0.80,
        max_output_tokens: int = 300,
        malformed_response_retries: int = 1,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        if max_output_tokens < 64:
            raise ValueError("max_output_tokens must be at least 64")
        if malformed_response_retries < 0:
            raise ValueError(
                "malformed_response_retries must be zero or greater"
            )

        self._judge = judge
        self._support_threshold = support_threshold
        self._max_output_tokens = max_output_tokens
        self._malformed_response_retries = malformed_response_retries

    @staticmethod
    def _evidence_by_source_id(
        evidence: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {
            str(item["source_id"]): item
            for item in evidence
            if item.get("source_id")
        }

    @staticmethod
    def _build_evidence_text(
        claim: ExtractedClaim,
        evidence_by_id: dict[str, dict[str, Any]],
    ) -> tuple[str, tuple[str, ...]]:
        blocks: list[str] = []
        evaluated_ids: list[str] = []

        for source_id in claim.cited_source_ids:
            item = evidence_by_id.get(source_id)
            if item is None:
                continue
            excerpt = str(item.get("excerpt") or "").strip()
            if not excerpt:
                continue

            evaluated_ids.append(source_id)
            blocks.append(f"[{source_id}]\n{excerpt}")

        return "\n\n".join(blocks), tuple(evaluated_ids)

    @classmethod
    def _candidate_texts(cls, text: str) -> list[str]:
        stripped = text.strip().lstrip("\ufeff")
        candidates: list[str] = []

        fenced = cls._CODE_FENCE_PATTERN.findall(stripped)
        candidates.extend(item.strip() for item in fenced if item.strip())
        candidates.append(stripped)

        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                _, end_index = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            candidates.append(stripped[index:index + end_index])
            break

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _coerce_supported(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "yes", "supported"}:
                return True
            if normalized in {"false", "no", "unsupported"}:
                return False
        raise ValueError("supported must be a boolean")

    @staticmethod
    def _coerce_score(value: Any) -> float:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized.endswith("%"):
                score = float(normalized[:-1]) / 100.0
            else:
                score = float(normalized)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
        else:
            raise ValueError("support_score must be numeric")

        if not 0.0 <= score <= 1.0:
            raise ValueError("support_score must be between 0 and 1")
        return score

    @classmethod
    def _parse_judgment(cls, text: str) -> dict[str, Any]:
        parse_errors: list[str] = []
        for candidate in cls._candidate_texts(text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
                continue

            if not isinstance(payload, dict):
                parse_errors.append("The JSON response is not an object")
                continue

            try:
                supported = cls._coerce_supported(payload.get("supported"))
                support_score = cls._coerce_score(
                    payload.get("support_score")
                )
                reason = payload.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("reason must be a non-empty string")
            except (TypeError, ValueError) as exc:
                parse_errors.append(str(exc))
                continue

            return {
                "supported": supported,
                "support_score": support_score,
                "reason": reason.strip(),
            }

        suffix = f" Last parser error: {parse_errors[-1]}" if parse_errors else ""
        raise ValueError(
            "Support judge did not return a valid judgment JSON object."
            + suffix
        )

    @staticmethod
    def _system_prompt() -> str:
        return """You are an independent evidence-entailment judge.
Evaluate one medical claim against only the supplied cited evidence.
Do not use outside knowledge.
A claim is supported only when every factual detail in the claim is directly stated or clearly entailed by the evidence.
Names, dosages, age groups, durations, frequencies, success rates, comparisons, and recommendations must be explicitly supported.
Related topic overlap is not enough.
Return exactly one JSON object and nothing else.
Use this exact schema:
{"supported": true, "support_score": 0.95, "reason": "brief explanation"}
Requirements:
- supported must be true or false.
- support_score must be a number from 0 to 1.
- reason must be a short non-empty string.
- Do not use markdown, code fences, headings, or commentary."""

    @staticmethod
    def _user_prompt(claim: ExtractedClaim, evidence_text: str) -> str:
        return (
            "CLAIM:\n"
            f"{claim.text}\n\n"
            "CITED EVIDENCE:\n"
            f"{evidence_text}"
        )

    @staticmethod
    def _retry_prompt(
        claim: ExtractedClaim,
        evidence_text: str,
        invalid_response: str,
    ) -> str:
        return (
            "The previous response was not valid JSON. Return only the required "
            "JSON object with supported, support_score, and reason.\n\n"
            "CLAIM:\n"
            f"{claim.text}\n\n"
            "CITED EVIDENCE:\n"
            f"{evidence_text}\n\n"
            "INVALID PREVIOUS RESPONSE:\n"
            f"{invalid_response[:1200]}"
        )

    async def _request_judgment(
        self,
        claim: ExtractedClaim,
        evidence_text: str,
    ) -> dict[str, Any] | None:
        user_prompt = self._user_prompt(claim, evidence_text)
        last_invalid_response = ""

        for attempt in range(self._malformed_response_retries + 1):
            if attempt > 0:
                user_prompt = self._retry_prompt(
                    claim,
                    evidence_text,
                    last_invalid_response,
                )

            generation = await self._judge.generate(
                system_prompt=self._system_prompt(),
                user_prompt=user_prompt,
                temperature=0.0,
                max_output_tokens=self._max_output_tokens,
            )
            last_invalid_response = str(generation.text).strip()

            try:
                return self._parse_judgment(last_invalid_response)
            except ValueError:
                if attempt >= self._malformed_response_retries:
                    return None

        return None

    async def evaluate_claim(
        self,
        claim: ExtractedClaim,
        *,
        evidence_by_id: dict[str, dict[str, Any]],
    ) -> ClaimSupportResult:
        if not claim.cited_source_ids:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                claim=claim.text,
                cited_source_ids=claim.cited_source_ids,
                evaluated_source_ids=(),
                supported=False,
                support_score=0.0,
                reason="The claim has no citation.",
            )

        evidence_text, evaluated_ids = self._build_evidence_text(
            claim,
            evidence_by_id,
        )
        if not evidence_text:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                claim=claim.text,
                cited_source_ids=claim.cited_source_ids,
                evaluated_source_ids=evaluated_ids,
                supported=False,
                support_score=0.0,
                reason=(
                    "None of the cited source IDs has available evidence text."
                ),
            )

        judgment = await self._request_judgment(claim, evidence_text)
        if judgment is None:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                claim=claim.text,
                cited_source_ids=claim.cited_source_ids,
                evaluated_source_ids=evaluated_ids,
                supported=False,
                support_score=0.0,
                reason=(
                    "Claim validation failed closed because the support judge "
                    "did not return valid JSON after retry."
                ),
            )

        supported = (
            judgment["supported"]
            and judgment["support_score"] >= self._support_threshold
        )
        return ClaimSupportResult(
            claim_id=claim.claim_id,
            claim=claim.text,
            cited_source_ids=claim.cited_source_ids,
            evaluated_source_ids=evaluated_ids,
            supported=supported,
            support_score=round(judgment["support_score"], 6),
            reason=judgment["reason"],
        )

    async def evaluate(
        self,
        claims: list[ExtractedClaim],
        *,
        evidence: list[dict[str, Any]],
    ) -> list[ClaimSupportResult]:
        evidence_by_id = self._evidence_by_source_id(evidence)
        results: list[ClaimSupportResult] = []
        for claim in claims:
            results.append(
                await self.evaluate_claim(
                    claim,
                    evidence_by_id=evidence_by_id,
                )
            )
        return results

    async def evaluate_as_dicts(
        self,
        claims: list[ExtractedClaim],
        *,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            asdict(result)
            for result in await self.evaluate(claims, evidence=evidence)
        ]
