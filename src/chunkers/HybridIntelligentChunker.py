from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass

from src.schemas.ingestion import DocumentSection, SemanticChunk
from src.stores.llm.GenerationInterface import GenerationInterface

from .ChunkerInterface import ChunkerInterface
from .SemanticChunker import SemanticChunker


@dataclass(slots=True)
class _Block:
    block_id: int
    document_name: str
    section_title: str
    page_number: int
    text: str


@dataclass(slots=True)
class _Group:
    blocks: list[_Block]
    section_title: str
    chunk_type: str = "narrative"


class HybridIntelligentChunker(ChunkerInterface):
    """Gemini plans topics; Python guarantees boundaries, sizes, and deduplication."""

    _GENERIC_TITLES = {
        "", "document information", "introduction", "untitled", "section",
        "general", "content", "document",
    }
    _CHUNK_TYPES = {
        "narrative", "definition", "recommendation", "evidence_table", "list",
        "hotline", "reference", "table_of_contents", "figure_caption", "metadata",
    }
    _SYSTEM_PROMPT = """You are a document chunk-boundary planner.
Return one strict JSON object only. Never rewrite, summarize, translate, omit, or
invent source text. Group contiguous blocks into coherent topics. Preserve headings,
complete paragraphs, lists, tables, definitions, and recommendations. Do not mix
references with narrative content. Use a short, specific section_title taken from or
faithfully describing the supplied text. Avoid generic titles such as Document
information, General, or Section.
Schema: {"segments":[{"start_block":0,"end_block":3,"section_title":"Specific title","chunk_type":"narrative"}]}
Allowed chunk_type: narrative, definition, recommendation, evidence_table, list,
hotline, reference, table_of_contents, figure_caption, metadata.
Every block must appear exactly once, in order, with no gaps or overlap."""

    def __init__(
        self,
        *,
        planner_provider: GenerationInterface,
        fallback_chunker: SemanticChunker,
        minimum_tokens: int = 180,
        target_tokens: int = 380,
        maximum_tokens: int = 550,
        planner_window_tokens: int = 4000,
        planner_max_blocks: int = 40,
        planner_overlap_blocks: int = 0,
        planner_max_output_tokens: int = 900,
        planner_concurrency: int = 2,
        planner_timeout_seconds: float = 30.0,
        use_llm_only_for_complex_sections: bool = True,
        remove_duplicates: bool = True,
        near_duplicate_threshold: float = 0.92,
        boundary_overlap_words: int = 30,
    ) -> None:
        if not 0 < minimum_tokens <= target_tokens <= maximum_tokens:
            raise ValueError("Require 0 < minimum_tokens <= target_tokens <= maximum_tokens")
        self._planner = planner_provider
        self._fallback = fallback_chunker
        self._minimum = minimum_tokens
        self._target = target_tokens
        self._maximum = maximum_tokens
        self._window_tokens = max(maximum_tokens, planner_window_tokens)
        self._max_blocks = max(4, planner_max_blocks)
        self._planner_overlap = max(0, planner_overlap_blocks)
        self._planner_output = max(256, planner_max_output_tokens)
        self._planner_concurrency = max(1, int(planner_concurrency))
        self._planner_timeout = max(5.0, float(planner_timeout_seconds))
        self._complex_only = use_llm_only_for_complex_sections
        self._deduplicate = remove_duplicates
        self._near_duplicate_threshold = min(0.99, max(0.70, near_duplicate_threshold))
        self._boundary_overlap_words = max(12, boundary_overlap_words)

    @staticmethod
    def _tokens(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    @staticmethod
    def _word_tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return bool(re.search(r"[.!?؟؛:][\]\)\}\"'’”]*$", text.strip()))

    @staticmethod
    def _starts_continuation(text: str) -> bool:
        clean = text.lstrip()
        return bool(clean) and (
            clean[0].islower()
            or clean.startswith((")", "]", ",", ";", ":", "-", "–", "—"))
        )

    @staticmethod
    def _longest_boundary_overlap(left: str, right: str, minimum: int) -> int:
        left_words = re.findall(r"\S+", left)
        right_words = re.findall(r"\S+", right)
        max_size = min(len(left_words), len(right_words), 250)
        for size in range(max_size, minimum - 1, -1):
            a = " ".join(left_words[-size:]).casefold()
            b = " ".join(right_words[:size]).casefold()
            if re.sub(r"\W+", "", a) == re.sub(r"\W+", "", b):
                return size
        return 0

    def _build_and_repair_blocks(self, sections: list[DocumentSection]) -> list[_Block]:
        raw: list[_Block] = []
        for section in sections:
            for element in section.elements:
                text = " ".join(str(element.text or "").split()).strip()
                if text:
                    raw.append(_Block(
                        block_id=len(raw),
                        document_name=section.document_name,
                        section_title=section.section_title,
                        page_number=element.page_number,
                        text=text,
                    ))

        repaired: list[_Block] = []
        for block in raw:
            if repaired:
                previous = repaired[-1]
                overlap = self._longest_boundary_overlap(
                    previous.text, block.text, self._boundary_overlap_words
                )
                if overlap:
                    words = block.text.split()
                    block.text = " ".join(words[overlap:]).strip()
                    if not block.text:
                        continue

                cross_page_continuation = (
                    block.page_number > previous.page_number
                    and not self._ends_sentence(previous.text)
                    and self._starts_continuation(block.text)
                    and previous.section_title == block.section_title
                )
                if cross_page_continuation:
                    previous.text = f"{previous.text.rstrip()} {block.text.lstrip()}"
                    continue
            block.block_id = len(repaired)
            repaired.append(block)
        return repaired

    def _windows(self, blocks: list[_Block]) -> list[list[_Block]]:
        windows: list[list[_Block]] = []
        current: list[_Block] = []
        size = 0
        for block in blocks:
            count = self._tokens(block.text)
            if current and (len(current) >= self._max_blocks or size + count > self._window_tokens):
                windows.append(current)
                current = current[-self._planner_overlap:] if self._planner_overlap else []
                size = sum(self._tokens(item.text) for item in current)
            current.append(block)
            size += count
        if current:
            windows.append(current)
        return windows

    async def _plan(self, window: list[_Block]) -> list[tuple[int, int, str, str]]:
        payload = {
            "minimum_tokens": self._minimum,
            "target_tokens": self._target,
            "maximum_tokens": self._maximum,
            "blocks": [{
                "block_id": block.block_id,
                "section_title": block.section_title,
                "page": block.page_number,
                "estimated_tokens": self._tokens(block.text),
                "text": block.text,
            } for block in window],
        }
        generate_json = getattr(self._planner, "generate_json", None)
        if callable(generate_json):
            result = await generate_json(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_output_tokens=self._planner_output,
            )
        else:
            result = await self._planner.generate(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_output_tokens=self._planner_output,
                top_p=None,
            )
        raw = str(result.text or "").strip()
        first, last = raw.find("{"), raw.rfind("}")
        if first < 0 or last < first:
            raise ValueError("Planner returned no JSON object")
        segments = json.loads(raw[first:last + 1]).get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("Planner returned no segments")

        parsed: list[tuple[int, int, str, str]] = []
        for item in segments:
            chunk_type = str(item.get("chunk_type") or "narrative").strip()
            if chunk_type not in self._CHUNK_TYPES:
                chunk_type = "narrative"
            parsed.append((
                int(item["start_block"]), int(item["end_block"]),
                str(item.get("section_title") or "").strip(), chunk_type,
            ))
        expected = [block.block_id for block in window]
        actual: list[int] = []
        for start, end, _, _ in parsed:
            if start > end:
                raise ValueError("Planner returned reversed boundary")
            actual.extend(range(start, end + 1))
        if actual != expected:
            raise ValueError("Planner boundaries contain gaps, overlap, or reordering")
        return parsed

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?؟؛:])\s+|\n{2,}", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _infer_title(self, text: str, proposed: str, original: str, chunk_type: str) -> str:
        clean_proposed = re.sub(r"\s+", " ", proposed).strip(" :-")
        if clean_proposed.casefold() not in self._GENERIC_TITLES and len(clean_proposed) >= 3:
            return clean_proposed[:500]
        if chunk_type == "reference" or re.match(r"^(references|bibliography)\b", text, re.I):
            return "References"
        heading = re.match(
            r"^([A-Z][A-Za-z0-9 /()&,'-]{2,100}?):\s+",
            text.strip(),
        )
        if heading:
            return heading.group(1).strip()[:500]
        clean_original = re.sub(r"\s+", " ", original).strip(" :-")
        if clean_original.casefold() not in self._GENERIC_TITLES:
            return clean_original[:500]
        words = self._word_tokens(text)[:9]
        return " ".join(word.capitalize() for word in words)[:500] or "Document content"

    def _hard_split_text(self, block: _Block) -> list[_Block]:
        sentences = self._sentences(block.text)
        output: list[_Block] = []
        current: list[str] = []
        for sentence in sentences:
            if self._tokens(sentence) > self._maximum:
                if current:
                    output.append(_Block(block.block_id, block.document_name, block.section_title, block.page_number, " ".join(current)))
                    current = []
                words = sentence.split()
                part: list[str] = []
                for word in words:
                    candidate = " ".join([*part, word])
                    if part and self._tokens(candidate) > self._maximum:
                        output.append(_Block(block.block_id, block.document_name, block.section_title, block.page_number, " ".join(part)))
                        part = [word]
                    else:
                        part.append(word)
                if part:
                    output.append(_Block(block.block_id, block.document_name, block.section_title, block.page_number, " ".join(part)))
                continue
            candidate = " ".join([*current, sentence])
            if current and self._tokens(candidate) > self._maximum:
                output.append(_Block(block.block_id, block.document_name, block.section_title, block.page_number, " ".join(current)))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            output.append(_Block(block.block_id, block.document_name, block.section_title, block.page_number, " ".join(current)))
        return output

    def _split_group(self, group: _Group) -> list[_Group]:
        output: list[_Group] = []
        current: list[_Block] = []
        size = 0
        for block in group.blocks:
            parts = [block] if self._tokens(block.text) <= self._maximum else self._hard_split_text(block)
            for part in parts:
                count = self._tokens(part.text)
                if current and size + count > self._maximum:
                    output.append(_Group(current, group.section_title, group.chunk_type))
                    current, size = [], 0
                current.append(part)
                size += count
        if current:
            output.append(_Group(current, group.section_title, group.chunk_type))
        return output

    def _merge_small(self, groups: list[_Group]) -> list[_Group]:
        groups = list(groups)
        index = 0
        while index < len(groups):
            group = groups[index]
            size = self._tokens("\n\n".join(block.text for block in group.blocks))
            if size >= self._minimum or len(groups) == 1:
                index += 1
                continue
            candidates: list[tuple[int, int]] = []
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(groups):
                    neighbor = groups[neighbor_index]
                    neighbor_size = self._tokens("\n\n".join(block.text for block in neighbor.blocks))
                    if neighbor_size + size <= self._maximum:
                        score = 2 if neighbor.section_title == group.section_title else 0
                        score += 1 if neighbor.chunk_type == group.chunk_type else 0
                        candidates.append((score, neighbor_index))
            if not candidates:
                index += 1
                continue
            _, neighbor_index = max(candidates)
            if neighbor_index < index:
                groups[neighbor_index].blocks.extend(group.blocks)
                groups.pop(index)
                index = max(0, neighbor_index)
            else:
                group.blocks.extend(groups[neighbor_index].blocks)
                groups.pop(neighbor_index)
        return groups

    @staticmethod
    def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
        words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if len(words) < size:
            return {tuple(words)} if words else set()
        return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}

    def _near_duplicate(self, left: str, right: str) -> bool:
        a, b = self._shingles(left), self._shingles(right)
        if not a or not b:
            return False
        containment = len(a & b) / min(len(a), len(b))
        jaccard = len(a & b) / len(a | b)
        return containment >= self._near_duplicate_threshold or jaccard >= self._near_duplicate_threshold

    def _python_window_groups(self, window: list[_Block]) -> list[_Group]:
        groups: list[_Group] = []
        current: list[_Block] = []
        size = 0
        for block in window:
            count = self._tokens(block.text)
            if current and (size + count > self._target or current[-1].section_title != block.section_title):
                groups.append(_Group(current, current[0].section_title))
                current, size = [], 0
            current.append(block)
            size += count
        if current:
            groups.append(_Group(current, current[0].section_title))
        return [part for group in groups for part in self._split_group(group)]

    async def _process_window(self, window: list[_Block], semaphore: asyncio.Semaphore) -> tuple[list[_Group], bool, str | None]:
        window_tokens = sum(self._tokens(block.text) for block in window)
        print(f"[HybridIntelligentChunker] Processing window blocks={window[0].block_id}-{window[-1].block_id}, tokens={window_tokens}", flush=True)
        if self._complex_only and window_tokens <= self._maximum:
            title = self._infer_title(window[0].text, "", window[0].section_title, "narrative")
            return [_Group(window, title)], False, None
        try:
            async with semaphore:
                plan = await asyncio.wait_for(self._plan(window), timeout=self._planner_timeout)
            by_id = {block.block_id: block for block in window}
            groups: list[_Group] = []
            for start, end, title, chunk_type in plan:
                blocks = [by_id[index] for index in range(start, end + 1)]
                full_text = "\n\n".join(block.text for block in blocks)
                title = self._infer_title(full_text, title, blocks[0].section_title, chunk_type)
                groups.append(_Group(blocks, title, chunk_type))
            print(f"[HybridIntelligentChunker] Gemini window completed blocks={window[0].block_id}-{window[-1].block_id}, segments={len(groups)}", flush=True)
            return groups, False, None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[HybridIntelligentChunker] Window fallback blocks={window[0].block_id}-{window[-1].block_id}: {message}", flush=True)
            return self._python_window_groups(window), True, message

    async def chunk(self, sections: list[DocumentSection]) -> list[SemanticChunk]:
        blocks = self._build_and_repair_blocks(sections)
        if not blocks:
            return []
        windows = self._windows(blocks)
        print(f"[HybridIntelligentChunker] Starting chunking: blocks={len(blocks)}, windows={len(windows)}, concurrency={self._planner_concurrency}, timeout={self._planner_timeout}s", flush=True)
        semaphore = asyncio.Semaphore(self._planner_concurrency)
        results = await asyncio.gather(*[self._process_window(window, semaphore) for window in windows])

        groups: list[_Group] = []
        fallback_windows = 0
        for window_groups, used_fallback, _ in results:
            groups.extend(window_groups)
            fallback_windows += int(used_fallback)
        groups = [part for group in groups for part in self._split_group(group)]
        groups = self._merge_small(groups)
        groups = [part for group in groups for part in self._split_group(group)]

        chunks: list[SemanticChunk] = []
        accepted_texts: list[str] = []
        exact_hashes: set[str] = set()
        for group in groups:
            text = "\n\n".join(block.text for block in group.blocks).strip()
            if not text:
                continue
            count = self._tokens(text)
            if count > self._maximum:
                raise RuntimeError(f"Hard chunk limit failed: {count} > {self._maximum}")
            normalized = re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if self._deduplicate and (digest in exact_hashes or any(self._near_duplicate(text, old) for old in accepted_texts)):
                continue
            exact_hashes.add(digest)
            accepted_texts.append(text)
            chunks.append(SemanticChunk(
                chunk_id=f"chunk_{len(chunks) + 1:04d}",
                document_name=group.blocks[0].document_name,
                section_title=group.section_title[:500],
                page_number=min(block.page_number for block in group.blocks),
                text=text,
            ))
        print(f"[HybridIntelligentChunker] Chunking completed: windows={len(windows)}, fallback_windows={fallback_windows}, final_chunks={len(chunks)}", flush=True)
        return chunks
