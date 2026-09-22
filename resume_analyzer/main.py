"""FastAPI application.  Run:  uvicorn resume_analyzer.main:app --reload"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google.genai import errors as genai_errors
from pydantic import TypeAdapter, ValidationError

from .analyzer import analyze
from .config import Settings, load_settings
from .embeddings import Embedder
from .errors import InputError
from .feedback import FeedbackGenerator
from .pdf_reader import read_resume_file
from .schemas import (
    AnalysisResult, AnalyzeTextRequest, FileProblem, HealthResponse, JobInput,
    MatchJobsResult, RankedJob, RankedResume, RankResumesResult,
)

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    print(f"Loading embedding model {settings.embed_model} (the first run downloads it)...")
    app.state.embedder = Embedder(settings.embed_model, settings.model_cache_dir)
    app.state.feedback = FeedbackGenerator(settings.gemini_api_key, settings.gemini_model)
    if not app.state.feedback.configured:
        print("GEMINI_API_KEY is not set: analysis works, but AI feedback is skipped.")
    print("Ready: http://localhost:8000")
    yield


app = FastAPI(
    title="AI Resume Analyzer & Job Matching",
    description="Compare a resume with a job description: skills, gaps, semantic similarity and AI feedback.",
    lifespan=Formlifespan,
)


@app.exception_handler(InputError)
async def input_error_handler(request: Request, exc: InputError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------- helpers ----------

def _state(request: Request):
    s = request.app.state
    return s.settings, s.embedder, s.feedback


def _read_upload(upload: UploadFile, settings: Settings) -> str:
    data = upload.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise InputError(f"{upload.filename}: larger than {settings.max_upload_mb} MB.")
    return read_resume_file(upload.filename or "", data)


def _llm_error_text(exc: Exception) -> str:
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return "AI feedback skipped: Gemini rate limit reached. Try again in a moment."
        if exc.code in (401, 403):
            return "AI feedback skipped: Gemini rejected the API key."
        return f"AI feedback skipped: {exc.message or exc}"
    return f"AI feedback failed: {exc}"


def _finish(analysis: Dict, resume_text: str, jd_text: str, generator: FeedbackGenerator, use_llm: bool) -> AnalysisResult:
    """Optionally add the LLM feedback. If the LLM fails, the analysis is still returned."""
    result = dict(analysis, feedback=None, feedback_error=None)
    if use_llm:
        if not generator.configured:
            result["feedback_error"] = "AI feedback skipped: GEMINI_API_KEY is not set."
        else:
            try:
                result["feedback"] = generator.generate(resume_text, jd_text, analysis)
            except Exception as exc:
                result["feedback_error"] = _llm_error_text(exc)
    return AnalysisResult(**result)


def _skill_names(analysis: Dict):
    matched = [s["name"] for s in analysis["skills"]["matched"]]
    missing = [s["name"] for s in analysis["skills"]["missing"] if s["importance"] == "required"]
    return matched, missing


# ---------- endpoints ----------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_HTML)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    settings, _, generator = _state(request)
    return HealthResponse(status="ok", embedding_model=settings.embed_model, llm_configured=generator.configured)


@app.post("/analyze", response_model=AnalysisResult, summary="Analyze a resume PDF against a job description")
def analyze_upload(
    request: Request,
    resume: UploadFile = File(..., description="Resume as a PDF (or .txt)"),
    job_description: str = Form(..., description="The job description text"),
    use_llm: bool = Form(True, description="Also generate AI feedback"),
):
    settings, embedder, generator = _state(request)
    resume_text = _read_upload(resume, settings)
    analysis = analyze(resume_text, job_description, embedder)
    return _finish(analysis, resume_text, job_description, generator, use_llm)


@app.post("/analyze/text", response_model=AnalysisResult, summary="Analyze resume text (JSON) against a job description")
def analyze_text(request: Request, body: AnalyzeTextRequest):
    _, embedder, generator = _state(request)
    analysis = analyze(body.resume_text, body.job_description, embedder)
    return _finish(analysis, body.resume_text, body.job_description, generator, body.use_llm)


@app.post("/match-jobs", response_model=MatchJobsResult, summary="Rank several jobs for one resume")
def match_jobs(
    request: Request,
    resume: UploadFile = File(...),
    jobs: str = Form(..., description='JSON list: [{"title": "...", "description": "..."}]'),
):
    settings, embedder, _ = _state(request)
    try:
        job_list = TypeAdapter(List[JobInput]).validate_json(jobs)
    except ValidationError:
        raise HTTPException(400, 'The "jobs" field must be a JSON list of {"title", "description"} objects.')
    if not job_list or len(job_list) > settings.max_jobs:
        raise HTTPException(400, f"Send between 1 and {settings.max_jobs} jobs.")

    resume_text = _read_upload(resume, settings)
    scored, errors = [], []
    for job in job_list:
        try:
            analysis = analyze(resume_text, job.description, embedder)
        except InputError as exc:
            errors.append(FileProblem(name=job.title, error=str(exc)))
            continue
        matched, missing = _skill_names(analysis)
        scored.append((analysis["overall_score"], job.title, analysis["verdict"], matched, missing))

    scored.sort(key=lambda row: row[0], reverse=True)
    ranking = [
        RankedJob(rank=i, title=t, overall_score=s, verdict=v, matched_skills=m, missing_skills=x)
        for i, (s, t, v, m, x) in enumerate(scored, start=1)
    ]
    return MatchJobsResult(ranking=ranking, errors=errors)


@app.post("/rank-resumes", response_model=RankResumesResult, summary="Rank several resumes for one job")
def rank_resumes(
    request: Request,
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...),
):
    settings, embedder, _ = _state(request)
    if not resumes or len(resumes) > settings.max_resumes:
        raise HTTPException(400, f"Send between 1 and {settings.max_resumes} resumes.")

    scored, errors = [], []
    for upload in resumes:
        name = upload.filename or "resume"
        try:
            analysis = analyze(_read_upload(upload, settings), job_description, embedder)
        except InputError as exc:
            errors.append(FileProblem(name=name, error=str(exc)))
            continue
        matched, missing = _skill_names(analysis)
        scored.append((analysis["overall_score"], name, analysis["verdict"], matched, missing,
                       analysis["experience"]["estimated_years"]))

    scored.sort(key=lambda row: row[0], reverse=True)
    ranking = [
        RankedResume(rank=i, filename=n, overall_score=s, verdict=v, matched_skills=m, missing_skills=x, estimated_years=y)
        for i, (s, n, v, m, x, y) in enumerate(scored, start=1)
    ]
    return RankResumesResult(ranking=ranking, errors=errors)
