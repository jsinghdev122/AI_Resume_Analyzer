"""Step 3: semantic similarity between a resume and a job description."""
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

# Cosine-similarity thresholds, tuned for sentence-transformers/all-MiniLM-L6-v2.
# A different embedding model has a different similarity scale: re-tune these if you change EMBED_MODEL.
STRONG = 0.55    # the resume clearly talks about this requirement
PARTIAL = 0.40   # related, but not a direct match
FLOOR = 0.20     # average best-match similarity that maps to a semantic score of 0
CEILING = 0.65   # ...and to a score of 100


class SupportsEmbed(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def status_for(similarity: float) -> str:
    if similarity >= STRONG:
        return "strong"
    if similarity >= PARTIAL:
        return "partial"
    return "gap"


def semantic_match(requirements: List[str], units: List[str], embedder: SupportsEmbed) -> Dict:
    """For every job requirement, find the resume line that is closest in meaning.

    Returns per-requirement results, the average best similarity, a 0-100 score, and the
    cosine similarity of the two documents as a whole (mean of their line vectors).
    """
    req_vectors = embedder.embed(requirements)
    unit_vectors = embedder.embed(units)

    similarities = req_vectors @ unit_vectors.T  # (requirements, resume lines)
    best_index = similarities.argmax(axis=1)
    best = similarities.max(axis=1)

    matches = []
    for i, requirement in enumerate(requirements):
        similarity = float(best[i])
        matches.append(
            {
                "requirement": requirement,
                "similarity": round(similarity, 3),
                "status": status_for(similarity),
                "evidence": units[int(best_index[i])],
            }
        )

    mean_best = float(best.mean())
    score = 100 * min(1.0, max(0.0, (mean_best - FLOOR) / (CEILING - FLOOR)))

    resume_doc, job_doc = unit_vectors.mean(axis=0), req_vectors.mean(axis=0)
    document_similarity = float(resume_doc @ job_doc / (np.linalg.norm(resume_doc) * np.linalg.norm(job_doc) + 1e-9))

    return {
        "matches": matches,
        "mean_best_similarity": round(mean_best, 3),
        "score": round(score, 1),
        "document_similarity": round(document_similarity, 3),
    }
