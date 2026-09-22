"""Step 1: get text out of an uploaded resume."""
from io import BytesIO

from pypdf import PdfReader

from .errors import InputError


def extract_pdf_text(data: bytes) -> str:
    """Text of every page of a PDF, joined with blank lines."""
    if not data.startswith(b"%PDF"):
        raise InputError("The file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise InputError("The PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"The PDF could not be read: {exc}") from exc

    text = "\n\n".join(pages).strip()
    if len(text) < 50:
        raise InputError(
            "Almost no text could be extracted. The PDF may be a scanned image: export a text-based PDF instead."
        )
    return text


def read_resume_file(filename: str, data: bytes) -> str:
    """Accepts a PDF (normal case) or a plain .txt resume."""
    name = (filename or "").lower()
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    if name.endswith(".pdf") or data.startswith(b"%PDF"):
        return extract_pdf_text(data)
    raise InputError("Please upload the resume as a PDF (or .txt) file.")
