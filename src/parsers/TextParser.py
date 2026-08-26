import re
from pathlib import Path

from src.schemas.ingestion import ParsedElement


class TextParser:
    """Extract clean, structured blocks from plain text or markdown-style text files."""

    def __init__(
        self,
        lines_per_page: int = 50,
        min_paragraph_length: int = 1,
    ) -> None:
        if lines_per_page < 1:
            raise ValueError("lines_per_page must be at least 1")
        self._lines_per_page = lines_per_page
        self._min_paragraph_length = min_paragraph_length

    @staticmethod
    def _read_file(file_path: Path) -> str:
        encodings = ["utf-8", "utf-8-sig", "cp1256", "cp1252", "latin-1"]
        raw_bytes = file_path.read_bytes()
        for encoding in encodings:
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _is_heading(line: str) -> tuple[bool, str]:
        stripped = line.strip()
        if not stripped:
            return False, ""

        # Markdown header check (e.g. # Title, ## Subtitle)
        md_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if md_match:
            return True, md_match.group(2).strip()

        # All uppercase header check (e.g. CLINICAL GUIDANCE FOR ALCOHOL RECOVERY)
        if (
            len(stripped) >= 4
            and len(stripped) <= 120
            and stripped.isupper()
            and not stripped.endswith((".", ",", ";"))
            and any(c.isalpha() for c in stripped)
        ):
            return True, stripped.title()

        # Section-like header (e.g. Section 1: Overview or Chapter 2 - Protocol)
        sec_match = re.match(
            r"^(?:Section|Chapter|Part|Guideline|Quality Statement)\s+\d+[:\-.]?\s*(.*)$",
            stripped,
            re.IGNORECASE,
        )
        if sec_match and len(stripped) <= 150:
            return True, stripped

        return False, stripped

    def parse(self, file_path: str | Path) -> list[ParsedElement]:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Text file was not found: {path}")

        raw_text = self._read_file(path)
        raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        paragraphs = re.split(r"\n\s*\n", raw_text)
        elements: list[ParsedElement] = []
        element_index = 0
        current_line_count = 0

        for paragraph in paragraphs:
            cleaned = paragraph.strip()
            if not cleaned or len(cleaned) < self._min_paragraph_length:
                continue

            lines = cleaned.split("\n")
            current_line_count += len(lines)
            current_page = max(1, (current_line_count // self._lines_per_page) + 1)

            # Check if this paragraph is a single line heading
            if len(lines) == 1:
                is_head, title_text = self._is_heading(cleaned)
                if is_head and title_text:
                    elements.append(
                        ParsedElement(
                            element_index=element_index,
                            text=title_text,
                            category="title",
                            page_number=current_page,
                            is_title=True,
                            is_table=False,
                            metadata={"source_format": "txt", "heading": True},
                        )
                    )
                    element_index += 1
                    continue

            # If multi-line paragraph starts with a heading line
            first_line = lines[0].strip()
            is_head, title_text = self._is_heading(first_line)
            if is_head and len(lines) > 1 and title_text:
                elements.append(
                    ParsedElement(
                        element_index=element_index,
                        text=title_text,
                        category="title",
                        page_number=current_page,
                        is_title=True,
                        is_table=False,
                        metadata={"source_format": "txt", "heading": True},
                    )
                )
                element_index += 1
                remaining_text = " ".join(l.strip() for l in lines[1:] if l.strip())
                if remaining_text:
                    elements.append(
                        ParsedElement(
                            element_index=element_index,
                            text=remaining_text,
                            category="paragraph",
                            page_number=current_page,
                            is_title=False,
                            is_table=False,
                            metadata={"source_format": "txt"},
                        )
                    )
                    element_index += 1
                continue

            # Regular paragraph
            combined_text = " ".join(l.strip() for l in lines if l.strip())
            if combined_text:
                elements.append(
                    ParsedElement(
                        element_index=element_index,
                        text=combined_text,
                        category="paragraph",
                        page_number=current_page,
                        is_title=False,
                        is_table=False,
                        metadata={"source_format": "txt"},
                    )
                )
                element_index += 1

        if not elements and raw_text.strip():
            elements.append(
                ParsedElement(
                    element_index=0,
                    text=raw_text.strip(),
                    category="paragraph",
                    page_number=1,
                    is_title=False,
                    is_table=False,
                    metadata={"source_format": "txt"},
                )
            )

        return elements
