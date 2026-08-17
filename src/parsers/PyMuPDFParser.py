import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf

from src.schemas.ingestion import ParsedElement

from .PDFParserInterface import PDFParserInterface


class PyMuPDFParser(PDFParserInterface):
    """Extract clean, page-aware blocks from text-based PDF documents."""

    def __init__(
        self,
        title_size_ratio: float = 1.25,
        repeated_text_ratio: float = 0.30,
    ) -> None:
        if title_size_ratio <= 1.0:
            raise ValueError("title_size_ratio must be greater than 1.0")
        if not 0.10 <= repeated_text_ratio <= 1.0:
            raise ValueError("repeated_text_ratio must be between 0.10 and 1.0")

        self._title_size_ratio = title_size_ratio
        self._repeated_text_ratio = repeated_text_ratio

    @staticmethod
    def _normalize_line(text: str) -> str:
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @classmethod
    def _clean_text(cls, text: str) -> str:
        lines = [cls._normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return ""

        joined = "\n".join(lines)
        joined = re.sub(r"(?<=[A-Za-z])-[ \t]*\n(?=[a-z])", "", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    @staticmethod
    def _canonical_repeated_text(text: str) -> str:
        normalized = " ".join(text.lower().split())
        normalized = re.sub(r"page\s+\d+\s+of\s+\d+", "page # of #", normalized)
        normalized = re.sub(r"\b\d+\b", "#", normalized)
        return normalized.strip(" .")

    @staticmethod
    def _block_data(block: dict[str, Any]) -> dict[str, Any]:
        lines: list[str] = []
        sizes: list[float] = []
        bold_characters = 0
        total_characters = 0

        for line in block.get("lines", []):
            spans_text: list[str] = []
            for span in line.get("spans", []):
                value = str(span.get("text", ""))
                if not value.strip():
                    continue
                spans_text.append(value)
                size = float(span.get("size", 0.0))
                sizes.extend([size] * max(len(value.strip()), 1))
                characters = len(value.strip())
                total_characters += characters
                font = str(span.get("font", "")).lower()
                flags = int(span.get("flags", 0))
                if "bold" in font or bool(flags & 16):
                    bold_characters += characters

            line_text = "".join(spans_text).strip()
            if line_text:
                lines.append(line_text)

        bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
        return {
            "text": "\n".join(lines),
            "bbox": bbox,
            "font_size": median(sizes) if sizes else 0.0,
            "max_font_size": max(sizes, default=0.0),
            "bold_ratio": bold_characters / total_characters if total_characters else 0.0,
        }

    @staticmethod
    def _explicit_heading(text: str) -> bool:
        value = " ".join(text.split()).strip()
        if not value or len(value) > 180 or value.endswith(('.', ',', ';')):
            return False

        return bool(re.fullmatch(
            r"(?:Contents|Quality statements|Quality statement(?:\s+\d+:.+)?|"
            r"Rationale|Quality measures|Process|Structure|Outcome|Source guidance|"
            r"Definitions of terms used in this quality statement|"
            r"Equality and diversity considerations|Brief intervention|"
            r"Validated alcohol questionnaire|Alcohol-use disorder|Alcohol dependence|"
            r"Community support networks and self-help groups|Brief triage assessment|"
            r"Acute alcohol withdrawal|Locally specified protocols|"
            r"Psychological interventions|Pharmacological interventions|"
            r"Update information|About this quality standard|"
            r"Diversity, equality and language|Endorsing organisation|"
            r"Supporting organisations)",
            value,
            flags=re.IGNORECASE,
        ))

    @classmethod
    def _is_noise(cls, text: str, repeated_texts: set[str]) -> bool:
        value = " ".join(text.split()).strip()
        canonical = cls._canonical_repeated_text(value)
        if canonical in repeated_texts:
            return True
        if re.fullmatch(r"Page\s+\d+\s+of\s+\d+", value, flags=re.IGNORECASE):
            return True
        if value.startswith("© NICE"):
            return True
        if value == "Alcohol-use disorders: diagnosis and management (QS11)":
            return True
        return False

    @staticmethod
    def _is_continuation(previous: dict[str, Any], current: dict[str, Any]) -> bool:
        previous_text = previous["text"].rstrip()
        current_text = current["text"].lstrip()
        if not previous_text or not current_text:
            return False
        if current_text[0].islower() or current_text.startswith((")", "]", ",", ".", ":", ";")):
            return True
        if previous_text.endswith(("-", ",", ":", ";", "(", "[")):
            return True
        if not previous_text.endswith((".", "?", "!")) and len(previous_text.split()) < 22:
            return True
        return False

    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file was not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        try:
            with pymupdf.open(path) as document:
                if document.page_count == 0:
                    raise ValueError("The PDF contains no pages")

                pages: list[list[dict[str, Any]]] = []
                candidates: Counter[str] = Counter()

                for page in document:
                    blocks = [
                        self._block_data(block)
                        for block in page.get_text("dict", sort=True).get("blocks", [])
                        if int(block.get("type", -1)) == 0
                    ]
                    cleaned = []
                    for block in blocks:
                        block["text"] = self._clean_text(block["text"])
                        if not block["text"]:
                            continue
                        cleaned.append(block)
                        y0, y1 = block["bbox"][1], block["bbox"][3]
                        page_height = float(page.rect.height)
                        if y0 < page_height * 0.12 or y1 > page_height * 0.88:
                            candidates[self._canonical_repeated_text(block["text"])] += 1
                    pages.append(cleaned)

                repetition_limit = max(2, int(document.page_count * self._repeated_text_ratio))
                repeated_texts = {
                    text for text, count in candidates.items()
                    if text and count >= repetition_limit
                }

                output: list[ParsedElement] = []
                for page_number, page_blocks in enumerate(pages, start=1):
                    usable = [
                        block for block in page_blocks
                        if not self._is_noise(block["text"], repeated_texts)
                    ]
                    body_sizes = [
                        block["font_size"] for block in usable
                        if block["font_size"] > 0 and len(block["text"]) > 40
                    ]
                    body_size = median(body_sizes) if body_sizes else 10.0

                    merged: list[dict[str, Any]] = []
                    for block in usable:
                        explicit_title = self._explicit_heading(block["text"])
                        if (
                            merged
                            and not explicit_title
                            and not merged[-1].get("is_title", False)
                            and self._is_continuation(merged[-1], block)
                        ):
                            separator = "" if merged[-1]["text"].endswith("-") else "\n"
                            merged[-1]["text"] = self._clean_text(
                                merged[-1]["text"] + separator + block["text"]
                            )
                            merged[-1]["bbox"] = (
                                min(merged[-1]["bbox"][0], block["bbox"][0]),
                                min(merged[-1]["bbox"][1], block["bbox"][1]),
                                max(merged[-1]["bbox"][2], block["bbox"][2]),
                                max(merged[-1]["bbox"][3], block["bbox"][3]),
                            )
                            continue

                        font_title = (
                            block["max_font_size"] >= body_size * self._title_size_ratio
                            and len(block["text"]) <= 180
                            and len(block["text"].split()) <= 20
                            and not block["text"].endswith(('.', ',', ';'))
                        )
                        block["is_title"] = explicit_title or font_title
                        merged.append(block)

                    # Preserve page-one document information as content.
                    if page_number == 1 and merged:
                        for block in merged:
                            if not self._explicit_heading(block["text"]):
                                block["is_title"] = False

                    for block_index, block in enumerate(merged):
                        output.append(ParsedElement(
                            element_index=len(output),
                            text=block["text"],
                            category="Title" if block["is_title"] else "NarrativeText",
                            page_number=page_number,
                            is_title=block["is_title"],
                            is_table=False,
                            metadata={
                                "source": "pymupdf",
                                "page_block_index": block_index,
                                "bbox": list(block["bbox"]),
                                "font_size": block["font_size"],
                                "body_font_size": body_size,
                            },
                        ))

        except pymupdf.FileDataError as exc:
            raise ValueError(f"Invalid or corrupted PDF: {path}") from exc

        if not output:
            raise ValueError("No text found. The PDF may require OCR.")
        return output
