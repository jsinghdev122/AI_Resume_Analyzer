"""Experience gap detection: years of experience and education level."""
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# "Jan 2019 - Mar 2021", "2018-2020", "03/2019 - Present", "June 2020 to current"
_RANGE_RE = re.compile(
    rf"(?:(?<![a-z])(?P<sm>{_MONTH})\.?,?\s+|(?<!\d)(?P<sn>\d{{1,2}})/)?(?P<sy>(?:19|20)\d{{2}})"
    rf"\s*(?:-|to|until|till)\s*"
    rf"(?:(?<![a-z])(?P<em>{_MONTH})\.?,?\s+|(?<!\d)(?P<en>\d{{1,2}})/)?"
    rf"(?:(?P<ey>(?:19|20)\d{{2}})|(?P<now>present|current|now|ongoing|date))",
    re.I,
)


def _month_number(name: Optional[str], number: Optional[str], default: int) -> int:
    if name:
        return _MONTH_NUMBERS[name[:3].lower()]
    if number and 1 <= int(number) <= 12:
        return int(number)
    return default


def date_ranges(text: str, today: date) -> List[Tuple[int, int]]:
    """Date ranges found in the text as (first_month, month_after_last) indexes.

    Both end months count, as on LinkedIn: "Jan 2019 - Dec 2019" is 12 months. A year without a
    month means January at the start and December at the end; "Present" means the current month.
    """
    now_end = today.year * 12 + today.month  # index of the month after the current one
    spans = []
    for m in _RANGE_RE.finditer(text):
        start = int(m["sy"]) * 12 + _month_number(m["sm"], m["sn"], 1) - 1
        if m["now"]:
            end = now_end
        else:
            end = int(m["ey"]) * 12 + _month_number(m["em"], m["en"], 12)
        end = min(end, now_end)
        if end > start and int(m["sy"]) >= 1980:
            spans.append((start, end))
    return spans


def years_from_ranges(spans: List[Tuple[int, int]]) -> float:
    """Total time covered by the ranges, counting overlapping jobs only once."""
    total, current_end, current_start = 0, None, None
    for start, end in sorted(spans):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return round(total / 12, 1)


_STATED_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w/&.-]+\s+){0,4}?experience", re.I
)


def estimate_resume_years(resume_text: str, sections: Dict[str, str], today: date) -> Tuple[Optional[float], str]:
    """Best estimate of total experience, with a note on where it came from.

    Uses date ranges from the experience section (or everything except education), and any
    "N years of experience" statement, whichever is larger.
    """
    scope = sections.get("experience")
    if not scope:
        scope = "\n".join(v for k, v in sections.items() if k not in ("education", "certifications")) or resume_text
    from_dates = years_from_ranges(date_ranges(scope, today))
    stated = [float(v) for v in _STATED_YEARS_RE.findall(resume_text) if float(v) <= 40]
    from_statement = max(stated) if stated else 0.0

    if from_dates == 0 and from_statement == 0:
        return None, "No dated positions or 'years of experience' statement found in the resume."
    if from_statement > from_dates:
        return from_statement, "Taken from a 'years of experience' statement in the resume."
    return from_dates, "Calculated from the date ranges in the experience section."


_JD_YEARS_PATTERNS = [
    re.compile(
        r"(\d{1,2})\s*\+?\s*(?:(?:-|to)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)(?:\s+of)?\s+(?:[\w/&.-]+\s+){0,5}?experience",
        re.I,
    ),
    re.compile(r"experience[^.\n]{0,40}?(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"minimum\s+(?:of\s+)?(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I),
]


def required_years(jd_text: str) -> Optional[int]:
    """The largest "N years of experience" figure in the job description (a heuristic)."""
    values = [int(v) for p in _JD_YEARS_PATTERNS for v in p.findall(jd_text)]
    values = [v for v in values if 1 <= v <= 20]
    return max(values) if values else None


# ---------- education ----------

EDUCATION_NAMES = {0: "none found", 1: "associate degree / diploma", 2: "bachelor's degree", 3: "master's degree", 4: "doctorate"}
_EDUCATION_PATTERNS = [
    (4, re.compile(r"\bph\.?d\b|doctorate|doctoral")),
    (3, re.compile(r"\bmaster'?s?\b|\bm\.s\.?(?=\W|$)|\bmsc\b|\bm\.sc\b|\bm\.?tech\b|\bmba\b|\bm\.eng\b|\bmeng\b")),
    (2, re.compile(r"\bbachelor'?s?\b|\bb\.s\.?(?=\W|$)|\bbsc\b|\bb\.sc\b|\bb\.?tech\b|\bb\.e\.|\bb\.a\.|\bbca\b|undergraduate degree")),
    (1, re.compile(r"associate'?s?\s+degree|\bdiploma\b")),
]


def _levels_in(text: str) -> List[int]:
    lowered = text.lower()
    return [level for level, pattern in _EDUCATION_PATTERNS if pattern.search(lowered)]


def education_check(resume_text: str, jd_text: str) -> Dict:
    """Compares the degree the job asks for (lowest level mentioned) with the highest one in the resume."""
    wanted = _levels_in(jd_text)
    required = min(wanted) if wanted else (2 if re.search(r"\bdegree\b", jd_text.lower()) else None)
    found = max(_levels_in(resume_text), default=0)
    return {
        "required": EDUCATION_NAMES[required] if required else None,
        "found": EDUCATION_NAMES[found],
        "met": None if required is None else found >= required,
    }
