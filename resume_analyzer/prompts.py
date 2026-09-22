"""Builds the text sent to the LLM. Pure Python, so it can be tested without an API key."""
from typing import Dict

from .preprocess import clean_text, redact_contacts

SYSTEM_INSTRUCTIONS = """You are an experienced technical recruiter and resume coach.
You receive a candidate's resume, a job description, and an automated analysis of how well they match.
Write practical feedback that helps the candidate improve their resume for this job.

Rules:
- Ground every point in the resume, the job description, or the analysis. Do not speculate.
- Never invent employers, dates, degrees, tools, projects or results. If you suggest adding a skill or a
  number, say it should only be added if it is true, and use placeholders such as [X%] for figures the
  candidate must supply.
- Be specific and concise. At most 5 items in each list and at most 4 bullet rewrites.
- For each bullet rewrite, "original" must be copied exactly from the resume.
- Do not comment on age, gender, nationality, appearance or other personal characteristics.
- The resume text has its email and phone number hidden on purpose; ignore the placeholders."""

MAX_RESUME_CHARS = 9000
MAX_JD_CHARS = 6000


def analysis_summary(analysis: Dict) -> str:
    """The key findings of the automated analysis, as plain text."""
    scores, skills = analysis["scores"], analysis["skills"]
    lines = [
        f"Overall match score: {analysis['overall_score']}/100 ({analysis['verdict']})",
        f"Semantic similarity score: {scores['semantic']}, skills score: {scores['skills']}, experience score: {scores['experience']}",
    ]
    required_missing = [s["name"] for s in skills["missing"] if s["importance"] == "required"]
    preferred_missing = [s["name"] for s in skills["missing"] if s["importance"] == "preferred"]
    listed_only = [s["name"] for s in skills["matched"] if s["evidence"] == "listed only"]
    lines.append("Required skills missing from the resume: " + (", ".join(required_missing) or "none"))
    lines.append("Preferred skills missing from the resume: " + (", ".join(preferred_missing) or "none"))
    lines.append("Skills that appear only in a skills list, without evidence in experience/projects: "
                 + (", ".join(listed_only) or "none"))

    exp = analysis["experience"]
    lines.append(f"Years of experience: job asks {exp['required_years']}, resume shows about {exp['estimated_years']}")
    edu = analysis["education"]
    lines.append(f"Education: job asks {edu['required']}, resume shows {edu['found']}")

    weakest = sorted((r for r in analysis["requirements"] if r["status"] != "strong"), key=lambda r: r["similarity"])[:6]
    if weakest:
        lines.append("Job requirements the resume covers weakly (requirement -> closest resume line):")
        for r in weakest:
            lines.append(f'- "{r["requirement"]}" -> "{r["evidence"]}" (similarity {r["similarity"]}, {r["status"]})')
    return "\n".join(lines)


def build_prompt(resume_text: str, jd_text: str, analysis: Dict) -> str:
    resume = redact_contacts(clean_text(resume_text))[:MAX_RESUME_CHARS]
    jd = clean_text(jd_text)[:MAX_JD_CHARS]
    return (
        f"=== AUTOMATED ANALYSIS ===\n{redact_contacts(analysis_summary(analysis))}\n\n"
        f"=== JOB DESCRIPTION ===\n{jd}\n\n"
        f"=== RESUME ===\n{resume}\n\n"
        "Now write the structured feedback."
    )
