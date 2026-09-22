"""Step 4: structured feedback from an LLM (Google Gemini)."""
from typing import Dict, Optional

from google import genai
from google.genai import types

from .prompts import SYSTEM_INSTRUCTIONS, build_prompt
from .schemas import Feedback


class FeedbackGenerator:
    """Creates the Gemini client on first use, so the app runs (without feedback) when no key is set."""

    def __init__(self, api_key: Optional[str], model: str):
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def generate(self, resume_text: str, jd_text: str, analysis: Dict) -> Feedback:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_prompt(resume_text, jd_text, analysis),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=Feedback,  # the model must answer with JSON in exactly this shape
                temperature=0.3,
            ),
        )
        if not response.text:
            raise RuntimeError("The model returned an empty answer (it may have been blocked).")
        return Feedback.model_validate_json(response.text)
