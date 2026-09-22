"""Step 2: text preprocessing.

Turns messy PDF text into clean, comparable text, finds the resume sections, and
splits a job description into individual requirements.
"""
import re
import unicodedata
from typing import Dict, Iterator, List, Tuple

_BULLETS = "•▪●◦‣∙·➢➤►■□◆❖✓✔"
_BULLET_RE = re.compile(f"[{_BULLETS}]")
_DASH_RE = re.compile("[\u2010-\u2015\u2212]")
_PAGE_NO_RE = re.compile(r"(?im)^\s*page\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*$")
_FRACTION_RE = re.compile(r"(?m)^\s*\d+\s*/\s*\d+\s*$")


def clean_text(raw: str) -> str:
    """Normalise unicode, undo PDF line-break damage, unify bullets and dashes, drop page numbers."""
    text = unicodedata.normalize("NFKC", raw or "")  # also turns ligatures like "ﬁ" into "fi"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # "develop-\nment" -> "development"
    text = _BULLET_RE.sub("- ", text)
    text = _DASH_RE.sub("-", text)
    text = _PAGE_NO_RE.sub("", text)
    text = _FRACTION_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\w/])\+?\d[\d\s().-]{8,}\d(?![\w/])")
_YEARS_ONLY_RE = re.compile(r"(?:(?:19|20)\d{2}[\s-]*)+")


def _mask_phone(match: "re.Match") -> str:
    digits = re.sub(r"\D", "", match.group())
    if 10 <= len(digits) <= 13 and not _YEARS_ONLY_RE.fullmatch(match.group()):
        return "[phone]"
    return match.group()


def redact_contacts(text: str) -> str:
    """Hide email addresses and phone numbers (used before text is sent to an LLM)."""
    return _PHONE_RE.sub(_mask_phone, _EMAIL_RE.sub("[email]", text))


# ---------- resume sections ----------

_SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "objective", "career objective", "about me"},
    "experience": {
        "experience", "work experience", "professional experience", "employment", "employment history",
        "work history", "internship", "internships", "relevant experience",
    },
    "education": {"education", "academic background", "academics", "qualifications", "education and training"},
    "skills": {
        "skills", "technical skills", "core competencies", "technologies", "tech stack", "key skills",
        "skills and tools", "technical proficiencies",
    },
    "projects": {"projects", "personal projects", "academic projects", "selected projects", "key projects"},
    "certifications": {"certifications", "certificates", "licenses", "courses", "training", "achievements"},
}
_ALIAS_TO_SECTION = {alias: name for name, aliases in _SECTION_ALIASES.items() for alias in aliases}


def _heading_of(line: str):
    words = re.sub(r"[^a-z& ]", "", line.lower().replace("&", "and")).split()
    if not words or len(words) > 4:
        return None
    return _ALIAS_TO_SECTION.get(" ".join(words))


def split_sections(text: str) -> Dict[str, str]:
    """Split a resume at headings like "Experience" or "SKILLS:". Text before the first heading is "header"."""
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in text.split("\n"):
        heading = _heading_of(line.strip()) if line.strip() else None
        if heading:
            current = heading
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "".join(lines).strip()}


def resume_units(resume_text: str, limit: int = 400) -> List[str]:
    """The resume as short lines/bullets (each one becomes a vector to match requirements against)."""
    units = []
    for raw in clean_text(resume_text).split("\n"):
        line = re.sub(r"^\s*[-*]\s*", "", raw).strip()
        if len(line.split()) >= 3 and not _EMAIL_RE.search(line):  # a contact line says nothing about skills
            units.append(line)
    return units[:limit]


# ---------- job description ----------

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_BOILERPLATE_RE = re.compile(
    r"equal opportunity|\beeo\b|we offer|\bbenefits\b|401\(?k\)?|salary|compensation|about us|who we are|"
    r"apply (?:now|today)|reasonable accommodation|paid time off|health insurance|perks|our mission",
    re.I,
)


def jd_lines(jd_text: str) -> Iterator[Tuple[str, bool]]:
    """Job description as (text, is_bullet) lines. Long paragraphs are split into sentences."""
    for raw in clean_text(jd_text).split("\n"):
        is_bullet = bool(_BULLET_PREFIX_RE.match(raw))
        line = _BULLET_PREFIX_RE.sub("", raw).strip()
        if not line:
            continue
        if len(line.split()) > 30:
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                if sentence.strip():
                    yield sentence.strip(), is_bullet
        else:
            yield line, is_bullet


def split_requirements(jd_text: str, limit: int = 60) -> List[str]:
    """The lines of a job description that read like requirements or duties."""
    lines = [
        line for line, _ in jd_lines(jd_text)
        if 4 <= len(line.split()) <= 60 and not _BOILERPLATE_RE.search(line)
    ]
    return lines[:limit]
