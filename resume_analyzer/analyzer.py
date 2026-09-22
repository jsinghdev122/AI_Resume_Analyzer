"""Combines the pipeline steps into one analysis of a resume against a job description."""
from datetime import date
from typing import Dict, List, Optional

from .errors import InputError
from .experience import education_check, estimate_resume_years, required_years
from .matching import SupportsEmbed, semantic_match
from .preprocess import clean_text, resume_units, split_requirements, split_sections
from .skills import find_skills, jd_skill_importance

# How much each part counts towards the overall score (parts that can't be computed are left out
# and the remaining weights are rescaled).
WEIGHTS = {"semantic": 0.40, "skills": 0.45, "experience": 0.15}
IMPORTANCE_WEIGHT = {"required": 1.0, "preferred": 0.5}
LISTED_ONLY_CREDIT = 0.75  # a skill that only appears in a skills list, never in experience or projects

MIN_RESUME_CHARS = 150
MIN_JD_CHARS = 60


def verdict_for(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 55:
        return "Good match with some gaps"
    if score >= 35:
        return "Partial match"
    return "Weak match"


def _skill_analysis(resume: str, jd_text: str, sections: Dict[str, str]) -> Dict:
    resume_skills = find_skills(resume)
    jd_skills = jd_skill_importance(jd_text)

    # "Demonstrated" = mentioned in experience/projects text, not only in a skills list.
    evidence_text = "\n".join(sections.get(name, "") for name in ("experience", "projects"))
    has_evidence_sections = bool(evidence_text.strip())
    demonstrated = set(find_skills(evidence_text)) if has_evidence_sections else set()

    matched: List[Dict] = []
    missing: List[Dict] = []
    earned = total = 0.0
    for name, (category, importance) in jd_skills.items():
        weight = IMPORTANCE_WEIGHT[importance]
        total += weight
        if name in resume_skills:
            listed_only = has_evidence_sections and name not in demonstrated
            earned += weight * (LISTED_ONLY_CREDIT if listed_only else 1.0)
            matched.append(
                {
                    "name": name,
                    "category": category,
                    "importance": importance,
                    "evidence": "listed only" if listed_only else "demonstrated",
                }
            )
        else:
            missing.append({"name": name, "category": category, "importance": importance})

    order = {"required": 0, "preferred": 1}
    matched.sort(key=lambda s: (order[s["importance"]], s["name"]))
    missing.sort(key=lambda s: (order[s["importance"]], s["name"]))
    extra = sorted(set(resume_skills) - set(jd_skills))[:25]

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "score": round(100 * earned / total, 1) if total else None,
    }


def _experience_analysis(resume: str, jd_text: str, sections: Dict[str, str], today: date) -> Dict:
    needed = required_years(jd_text)
    have, note = estimate_resume_years(resume, sections, today)
    result = {"required_years": needed, "estimated_years": have, "gap_years": None, "met": None, "note": note, "score": None}
    if needed is not None and have is not None:
        result["gap_years"] = round(max(0.0, needed - have), 1)
        result["met"] = have >= needed
        result["score"] = round(100 * min(1.0, have / needed), 1)
    elif needed is None:
        result["note"] = "The job description does not state a number of years."
    return result


def analyze(resume_text: str, jd_text: str, embedder: SupportsEmbed, today: Optional[date] = None) -> Dict:
    """Full analysis. Pure Python + embeddings: no LLM involved."""
    today = today or date.today()
    resume, jd = clean_text(resume_text), clean_text(jd_text)
    if len(resume) < MIN_RESUME_CHARS:
        raise InputError("The resume text is too short to analyse.")
    if len(jd) < MIN_JD_CHARS:
        raise InputError("The job description is too short to analyse.")

    warnings: List[str] = []
    sections = split_sections(resume)
    if len(sections) <= 1:
        warnings.append(
            "No section headings (Experience, Education, Skills...) were recognised in the resume, "
            "so experience dates and skill evidence are less reliable."
        )

    skills = _skill_analysis(resume, jd_text, sections)
    if skills["score"] is None:
        warnings.append("No known skills were found in the job description, so the skills score was skipped.")

    requirements, units = split_requirements(jd_text), resume_units(resume)
    semantic: Optional[Dict] = None
    if requirements and units:
        semantic = semantic_match(requirements, units, embedder)
    else:
        warnings.append("Not enough text to compare meaning, so the semantic score was skipped.")

    experience = _experience_analysis(resume, jd_text, sections, today)
    education = education_check(resume, jd_text)

    parts = {
        "semantic": semantic["score"] if semantic else None,
        "skills": skills["score"],
        "experience": experience["score"],
    }
    available = {name: WEIGHTS[name] for name, value in parts.items() if value is not None}
    if not available:
        raise InputError("Could not compute any score from these texts.")
    scale = sum(available.values())
    overall = round(sum(parts[name] * weight for name, weight in available.items()) / scale, 1)

    return {
        "overall_score": overall,
        "verdict": verdict_for(overall),
        "scores": {
            "semantic": parts["semantic"],
            "skills": parts["skills"],
            "experience": parts["experience"],
            "document_similarity": semantic["document_similarity"] if semantic else None,
        },
        "weights_used": {name: round(weight / scale, 3) for name, weight in available.items()},
        "skills": {"matched": skills["matched"], "missing": skills["missing"], "extra": skills["extra"]},
        "requirements": semantic["matches"] if semantic else [],
        "experience": {k: v for k, v in experience.items() if k != "score"},
        "education": education,
        "warnings": warnings,
    }
