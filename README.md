# AI Resume Analyzer & Job Matching System

Upload a resume (PDF) and a job description. The system reports **how well they match**, **which skills are present or missing**, **where the experience falls short**, and uses an LLM to write **specific advice for improving the resume**. It can also rank one resume against many jobs, or many resumes against one job.

Built with FastAPI, local sentence embeddings, and Google Gemini. Ships as a Docker container.

## How it works

```
resume PDF ──► 1. extract text ──► 2. preprocess ──┐
                                                    ├──► 3. analyse ──► score + gaps ──► 4. LLM feedback
job description ─────────────────► 2. preprocess ──┘         │
                                                  skills · semantic similarity · experience · education
```

1. **PDF text extraction** (`pdf_reader.py`): `pypdf` reads every page. Password-protected files, non-PDFs and scanned images (no text layer) are rejected with a clear message.
2. **Preprocessing** (`preprocess.py`): Unicode normalisation, repairing words split by line breaks, unifying bullets and dashes, dropping page numbers, detecting resume sections (Experience, Skills, Education...), and splitting the job description into individual requirements. Email addresses and phone numbers are hidden before any text is sent to the LLM.
3. **Analysis** (`analyzer.py`), made of three checks:
   - **Skills** (`skills.py`): a curated taxonomy of about 250 skills is matched with word-boundary-safe patterns (so `Java` is not found inside `JavaScript`, and `go` is not a skill). Job skills are tagged **required** or **nice to have** from wording like "a plus" or headings like "Preferred qualifications". Skills that appear only in a skills list, never in experience or projects, get partial credit.
   - **Semantic similarity** (`embeddings.py`, `matching.py`): every job requirement and every resume line is turned into a vector (`all-MiniLM-L6-v2`, run locally through `fastembed`). Each requirement is matched to the closest resume line by cosine similarity, which shows *evidence* for the requirement or shows a gap. This catches matches that share meaning but not keywords ("built dashboards" ~ "data visualization").
   - **Experience and education** (`experience.py`): total years are estimated from the date ranges in the experience section (overlapping jobs counted once) and any "N years of experience" statement, then compared with the years the job asks for. The degree level is compared as well.
4. **LLM feedback** (`prompts.py`, `feedback.py`): Gemini receives the resume, the job description, and the computed findings, and must answer in a fixed JSON structure (summary, strengths, gaps, priority actions, bullet rewrites, keywords, ATS tips). The prompt forbids inventing facts. If the LLM is unavailable, the analysis is still returned.

### Scoring

```
overall = 40% semantic match + 45% skills + 15% experience
```

Parts that cannot be computed (for example, a job description that names no years) are left out and the remaining weights are rescaled. `verdict`: 75+ strong, 55-75 good with gaps, 35-55 partial, below 35 weak. The weights and thresholds are heuristics chosen for transparency, not a validated hiring model.

## Run it

```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
set GEMINI_API_KEY=your-key    # optional; macOS/Linux: export GEMINI_API_KEY='your-key'
uvicorn resume_analyzer.main:app --reload
```

A free Gemini key is available at https://aistudio.google.com/app/apikey. Without a key, everything works except the AI feedback.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/analyze` | Resume PDF + job description text: full analysis and optional AI feedback |
| POST | `/analyze/text` | Same, with the resume as JSON text |
| POST | `/match-jobs` | One resume, several jobs: jobs ranked by fit |
| POST | `/rank-resumes` | Several resumes, one job: resumes ranked by fit |
| GET | `/health` | Status, embedding model, whether an LLM key is set |

```bash
# full analysis (set use_llm=true for AI feedback)
curl -X POST http://localhost:8000/analyze \
  -F "resume=@samples/resume_sample.pdf" \
  -F "job_description=<samples/job_description_sample.txt" \
  -F "use_llm=false"

# rank jobs for a resume
curl -X POST http://localhost:8000/match-jobs \
  -F "resume=@samples/resume_sample.pdf" \
  -F "jobs=<samples/jobs.json"

# rank resumes for a job
curl -X POST http://localhost:8000/rank-resumes \
  -F "resumes=@samples/resume_sample.pdf" -F "resumes=@another.pdf" \
  -F "job_description=<samples/job_description_sample.txt"
```

The response of `/analyze` contains `overall_score`, `verdict`, `scores`, `skills` (`matched`, `missing`, `extra`), `requirements` (each with its closest resume line), `experience`, `education`, `warnings`, and `feedback`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Enables AI feedback (`GOOGLE_API_KEY` also works) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model used for feedback |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `MODEL_CACHE_DIR` | fastembed default | Where model files are stored (`/opt/models` in Docker) |
| `MAX_UPLOAD_MB` | `5` | Largest resume accepted |
| `MAX_JOBS` / `MAX_RESUMES` | `20` / `25` | Batch limits |

## Limitations

- Scanned (image-only) PDFs are not supported; there is no OCR.
- Skill detection depends on the taxonomy in `skills.py`. Skills that are not in it are ignored by the skills score (the semantic score still sees them). Extending it is a matter of adding entries.
- Experience years come from pattern matching on dates; unusual date formats can be missed, and the result is an estimate.
- Layout-heavy resumes (columns, tables) can come out of the PDF with jumbled reading order.
- A match score is a screening aid, not a hiring decision. `/rank-resumes` should support a human reviewer, never replace one.

## Project layout

```
resume_analyzer/
  main.py         FastAPI app and endpoints
  analyzer.py     combines skills, semantic match, experience into one score
  pdf_reader.py   PDF -> text
  preprocess.py   cleaning, redaction, sections, requirement splitting
  skills.py       skill taxonomy and required/preferred detection
  experience.py   years of experience and education
  embeddings.py   local sentence embeddings (with a cache)
  matching.py     requirement-to-resume semantic matching
  prompts.py      LLM prompt construction
  feedback.py     Gemini call with structured JSON output
  schemas.py      request/response models
  config.py       settings
static/index.html   web UI

```