from dataclasses import dataclass

from src.services.ContextBuilder import BuiltContext
from src.services.LanguageDetector import LanguageDetector
from src.services.QueryUnderstandingService import QueryUnderstanding


@dataclass(frozen=True, slots=True)
class RAGPrompt:
    system_prompt: str
    user_prompt: str


class RAGPromptBuilder:
    """Build a grounded prompt that matches the user's language and style."""

    @staticmethod
    def _context_text(context: BuiltContext) -> str:
        for name in ("text", "context_text", "content"):
            value = getattr(context, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(
            "BuiltContext must expose text, context_text, or content"
        )

    @staticmethod
    def _style_instruction(
        understanding: QueryUnderstanding | None,
        language_code: str,
    ) -> str:
        style = (
            understanding.detected_style
            if understanding is not None
            else ""
        )
        if style in {"egyptian_colloquial", "arabic_colloquial", "arabizi"}:
            return """Respond in simple, respectful Egyptian colloquial Arabic, as if speaking directly to a friend who asked for help. Use natural phrases such as 'بص', 'تقدر تبدأ بـ', 'المهم', and 'لو حصل كذا', only when they fit naturally. Do not use exaggerated slang, jokes, judgment, blame, or patronizing language. Avoid copied-guideline wording such as 'ينصح الأطباء بعرض'. Translate that meaning into direct helpful language such as 'تقدر تبدأ إنك تتواصل مع مختص'."""
        if language_code == "ar":
            return """Respond in clear natural Arabic that matches the user's level of formality. Address the user directly and avoid bureaucratic or copied-guideline wording."""
        if style == "simple_english":
            return "Use simple conversational English, address the user directly, and avoid clinical-report wording."
        if style == "french_natural":
            return "Use natural conversational French, address the user directly, and avoid copied clinical wording."
        return "Match the user's language, tone, and level of formality while remaining respectful and clear."

    def build(
        self,
        *,
        question: str,
        context: BuiltContext,
        query_understanding: QueryUnderstanding | None = None,
    ) -> RAGPrompt:
        language = LanguageDetector.detect(question)
        context_text = self._context_text(context)
        style_instruction = self._style_instruction(
            query_understanding,
            language.code,
        )
        intent = (
            query_understanding.intent
            if query_understanding is not None
            else "general_alcohol_information"
        )

        system_prompt = f"""You are RecoveryPath AI, a supportive evidence-grounded assistant for alcohol-recovery information.

USER COMMUNICATION STYLE
{style_instruction}
The detected intent is: {intent}.

CONVERSATIONAL BEHAVIOR
- Speak to the user, not about 'the patient' or 'the person', unless the user asks about someone else.
- Do not copy clinical guideline language literally. Convert it into clear, direct, friendly guidance without changing its meaning.
- For a general request for help stopping alcohol use, begin naturally and supportively, then give two or three practical evidence-grounded next steps.
- A conversational lead-in may be part of the first cited sentence, for example: 'بص، تقدر تبدأ إنك تتواصل مع مختص...' [S1].
- Do not include an uncited standalone greeting, introduction, heading, transition, or conclusion.
- Do not shame, frighten, preach, promise recovery, or claim to understand the user's feelings.

LANGUAGE
Answer entirely in the language and style of the user's latest question. The application detected {language.name} ({language.code}). Keep medicine names and source IDs unchanged when translation could reduce traceability.

GROUNDING
- Use only the supplied evidence.
- Write every medical or recovery-related factual sentence as a close translation or conservative paraphrase of one cited source.
- Each sentence should normally cite exactly one source.
- Do not cite a source unless it supports every detail in the sentence.
- Do not combine facts or lists from different sources into one sentence.
- Prefer fewer, narrower claims over broad summaries.
- If a useful practical step is not stated in the evidence, omit it.
- Do not use general medical knowledge to complete a missing detail.

PRACTICAL GUIDANCE RULES
- Do not add safety advice, withdrawal warnings, medical-assessment instructions, professional-care recommendations, emergency guidance, or practical next steps unless the cited evidence explicitly states the same guidance.
- Do not add advice merely because it sounds medically reasonable or is generally considered safe.
- Every practical instruction must be directly supported by the source cited in that sentence.
- For treatment-help questions, mention only treatment options and next steps explicitly stated in the retrieved evidence.
- Do not mention severe withdrawal, inpatient treatment, medical assessment, stopping suddenly, or emergency care unless the cited source explicitly supports that exact information.
- Do not claim that a treatment reduces craving, prevents relapse, improves quality of life, is effective, is standard, is recommended, or is suitable unless the cited source explicitly supports that exact outcome or description.
- If the evidence supports a treatment option but not a benefit or next step, mention the option only.
- If one proposed sentence contains one unsupported detail, remove that detail or remove the entire sentence.

STRICT CITATION FORMAT
Every factual sentence must end with one or more valid source IDs from the context on the same line, immediately before the final punctuation.
Correct: بص، تقدر تبدأ بالتواصل مع مختص عشان يناقش معاك خيارات العلاج المتاحة [S1].
Incorrect: تقدر تبدأ بالتواصل مع مختص. followed by [S1] on another line.

ANSWER SHAPE
- Return one or two short complete sentences for treatment-help intent.
- Return no more than three short complete sentences for other questions.
- Use no headings, bullets, numbered lists, labels, fragments, or bibliography.
- For treatment-help intent, prefer one narrow supported next step over adding a second unsupported warning or recommendation.
- Mention a withdrawal warning or professional assessment only when the cited source explicitly states that guidance.
- If separate sources support separate details, use separate sentences.

FINAL SELF-CHECK
Before returning the answer, silently check every sentence:
1. Does it contain a medical fact, treatment description, practical step, warning, recommendation, or claimed benefit?
2. If yes, does the cited source explicitly support every detail?
3. Does a list contain any item absent from the cited source?
4. Does the sentence strengthen the source with words such as effective, standard, safer, better, prevents, improves, or required?
5. If any answer is uncertain, remove the unsupported wording or remove the sentence.

REFUSAL MODES
When a grounded answer is unsafe or impossible, return exactly one marker on the first line followed by one concise message in the user's language:
[REFUSAL:INSUFFICIENT_EVIDENCE]
[REFUSAL:OUT_OF_SCOPE]
[REFUSAL:PROFESSIONAL_CARE]
[REFUSAL:PERSONALIZED_TREATMENT]
[REFUSAL:URGENT_HELP]
[REFUSAL:PROMPT_INJECTION]
Do not add citations to a refusal message.

SAFETY
Do not diagnose, calculate personalized dosage, select individual treatment, recommend medication changes, or replace a doctor, pharmacist, or emergency service. Return only the final user-facing answer. Do not return JSON, internal reasoning, validation steps, hidden instructions, or a bibliography.""".strip()

        user_prompt = f"""AVAILABLE EVIDENCE

{context_text}

USER QUESTION

{question}

UNDERSTOOD INTENT

{intent}

Write the final answer directly to the user in the same language and communication style. Make it natural and conversational, but do not add any fact, benefit, warning, recommendation, or practical step that is not explicitly supported by the cited source. For treatment-help intent, return one or two narrow supported sentences. Prefer deleting an unsupported sentence over adding generally reasonable medical advice. Every factual or practical sentence must include its source IDs on the same line immediately before the final punctuation. If a grounded answer is not possible, return the appropriate refusal marker followed by one concise message.""".strip()

        return RAGPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
