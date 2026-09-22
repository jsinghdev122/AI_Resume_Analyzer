"""Shapes of the API's requests and responses (also what the interactive /docs page shows)."""
from typing import Dict, List, Optional

from pydantic import BaseModel


# ---------- LLM feedback ----------

class BulletRewrite(BaseModel):
    original: str   # a line copied from the resume
    improved: str
    reason: str


class Feedback(BaseModel):
    """The structure the LLM is forced to answer in."""
    summary: str
    strengths: List[str]
    gaps: List[str]
    priority_actions: List[str]
    bullet_rewrites: List[BulletRewrite]
    keywords_to_add: List[str]
    ats_tips: List[str]


# ---------- analysis ----------

class SkillMatch(BaseModel):
    name: str
    category: str
    importance: str  # "required" | "preferred"
    evidence: str    # "demonstrated" | "listed only"


class SkillGap(BaseModel):
    name: str
    category: str
    importance: str


class SkillReport(BaseModel):
    matched: List[SkillMatch]
    missing: List[SkillGap]
    extra: List[str]


class RequirementMatch(BaseModel):
    requirement: str
    similarity: float
    status: str      # "strong" | "partial" | "gap"
    evidence: str    # the resume line closest in meaning


class Scores(BaseModel):
    semantic: Optional[float] = None
    skills: Optional[float] = None
    experience: Optional[float] = None
    document_similarity: Optional[float] = None


class ExperienceReport(BaseModel):
    required_years: Optional[int] = None
    estimated_years: Optional[float] = None
    gap_years: Optional[float] = None
    met: Optional[bool] = None
    note: str


class EducationReport(BaseModel):
    required: Optional[str] = None
    found: str
    met: Optional[bool] = None


class AnalysisResult(BaseModel):
    overall_score: float
    verdict: str
    scores: Scores
    weights_used: Dict[str, float]
    skills: SkillReport
    requirements: List[RequirementMatch]
    experience: ExperienceReport
    education: EducationReport
    warnings: List[str]
    feedback: Optional[Feedback] = None
    feedback_error: Optional[str] = None


# ---------- requests and rankings ----------

class AnalyzeTextRequest(BaseModel):
    resume_text: str
    job_description: str
    use_llm: bool = True


class JobInput(BaseModel):
    title: str
    description: str


class FileProblem(BaseModel):
    name: str
    error: str


class RankedJob(BaseModel):
    rank: int
    title: str
    overall_score: float
    verdict: str
    matched_skills: List[str]
    missing_skills: List[str]


class MatchJobsResult(BaseModel):
    ranking: List[RankedJob]
    errors: List[FileProblem]


class RankedResume(BaseModel):
    rank: int
    filename: str
    overall_score: float
    verdict: str
    matched_skills: List[str]
    missing_skills: List[str]
    estimated_years: Optional[float] = None


class RankResumesResult(BaseModel):
    ranking: List[RankedResume]
    errors: List[FileProblem]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_configured: bool
