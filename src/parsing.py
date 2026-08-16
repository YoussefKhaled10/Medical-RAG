import argparse
import json
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Table, Title
from unstructured.partition.pdf import partition_pdf


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def clean_text(text: str | None) -> str:
    """Normalize extracted PDF text without destroying paragraph boundaries."""
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_section_title(element: Any) -> bool:
    """Detect section headings using Unstructured element types and metadata."""
    if isinstance(element, Title):
        return True

    category = str(getattr(element, "category", "")).lower()
    if category == "title":
        return True

    text = clean_text(str(element))
    if not text or len(text) > 180 or "\n" in text:
        return False

    word_count = len(text.split())
    numbered_heading = bool(
        re.match(r"^(?:\d+(?:\.\d+)*|[A-Z]|[IVXLC]+)[.)]?\s+\S+", text)
    )
    uppercase_heading = text.isupper() and 1 <= word_count <= 15

    return numbered_heading or uppercase_heading


def extract_element_text(element: Any) -> str:
    """Extract clean text while keeping tables identifiable in the chunk text."""
    text = clean_text(str(element))
    if not text:
        return ""

    if isinstance(element, Table) or str(
        getattr(element, "category", "")
    ).lower() == "table":
        return f"[TABLE]\n{text}\n[/TABLE]"

    return text


def build_sections(elements: list[Any]) -> list[dict[str, Any]]:
    """Group parsed elements into section-aware documents with page ranges."""
    sections: list[dict[str, Any]] = []
    current_title = "Introduction"
    current_parts: list[str] = []
    current_pages: list[int] = []

    def save_current_section() -> None:
        section_text = clean_text("\n\n".join(current_parts))
        if not section_text:
            return

        valid_pages = [page for page in current_pages if page > 0]
        sections.append(
            {
                "section_title": current_title,
                "text": section_text,
                "page_start": min(valid_pages) if valid_pages else None,
                "page_end": max(valid_pages) if valid_pages else None,
            }
        )

    for element in elements:
        text = extract_element_text(element)
        if not text:
            continue

        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None)
        page_number = int(page_number) if page_number is not None else 0

        if is_section_title(element):
            save_current_section()
            current_title = clean_text(str(element))
            current_parts = []
            current_pages = []
            continue

        current_parts.append(text)
        if page_number:
            current_pages.append(page_number)

    save_current_section()
    return sections


def approximate_token_count(text: str) -> int:
    """Return a lightweight token estimate for monitoring and DB metadata."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def create_chunks(
    sections: list[dict[str, Any]],
    asset_id: int,
    document_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split sections and return records compatible with the chunks table."""
    if asset_id <= 0:
        raise ValueError("asset_id must be greater than zero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for section in sections:
        section_chunks = splitter.split_text(section["text"])

        for section_chunk_number, chunk_text in enumerate(section_chunks, start=1):
            chunk_text = clean_text(chunk_text)
            if not chunk_text:
                continue

            stable_key = (
                f"{asset_id}:{document_name}:{section['section_title']}:"
                f"{section_chunk_number}:{chunk_text}"
            )
            chunk_uuid = str(uuid5(NAMESPACE_URL, stable_key))

            chunks.append(
                {
                    "asset_id": asset_id,
                    "chunk_uuid": chunk_uuid,
                    "section_title": section["section_title"],
                    "page_number": section["page_start"],
                    "page_end": section["page_end"],
                    "token_count": approximate_token_count(chunk_text),
                    "chunk_text": chunk_text,
                    "chunk_number": chunk_number,
                    "section_chunk_number": section_chunk_number,
                    "document_name": document_name,
                }
            )
            chunk_number += 1

    return chunks


def parse_pdf_to_chunks(
    pdf_path: str | Path,
    asset_id: int,
    strategy: str = "fast",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Parse one PDF and return section-aware chunks without DB side effects."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file was not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    elements = partition_pdf(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=True,
        include_page_breaks=False,
    )

    print(f"Number of elements: {len(elements)}")

    sections = build_sections(elements)
    print(f"Number of sections: {len(sections)}")

    chunks = create_chunks(
        sections=sections,
        asset_id=asset_id,
        document_name=path.stem,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("\n====================================")
    print("Parsing + Cleaning + Chunking Done")
    print("====================================")
    print(f"Document : {path.stem}")
    print(f"Sections : {len(sections)}")
    print(f"Chunks   : {len(chunks)}")

    return chunks


def save_chunks_json(chunks: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save chunks to UTF-8 JSON and return the final output path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a clinical PDF into section-aware JSON chunks."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument(
        "--asset-id",
        type=int,
        required=True,
        help="Existing assets.id value from PostgreSQL",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <pdf_name>_chunks.json",
    )
    parser.add_argument(
        "--strategy",
        choices=["fast", "hi_res", "ocr_only"],
        default="fast",
        help="Unstructured PDF parsing strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    pdf_path = Path(args.pdf_path)
    output_path = args.output or pdf_path.with_name(
        f"{pdf_path.stem}_chunks.json"
    )

    chunks = parse_pdf_to_chunks(
        pdf_path=pdf_path,
        asset_id=args.asset_id,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    for chunk in chunks[:3]:
        print("\n--- Chunk Preview ---")
        print(json.dumps(chunk, ensure_ascii=False, indent=2))

    saved_path = save_chunks_json(chunks, output_path)
    print(f"\nJSON saved successfully: {saved_path}")


if __name__ == "__main__":
    main()
