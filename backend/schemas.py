"""
Pydantic models for the HTTP boundary.

Interview line: "Every request and response shape is declared here, so a
malformed request fails with a clear 422 before it ever reaches business
logic, and FastAPI generates the Swagger docs from these classes for free."
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------ analysis
class GapReport(BaseModel):
    matched: List[str]
    missing: List[str]
    extra: List[str]
    match_ratio: float


class UploadResponse(BaseModel):
    session_id: str
    candidate_name: str
    resume_skills: List[str]
    jd_skills: List[str]
    gap: GapReport
    max_questions: int


# ------------------------------------------------------------ interview
class QuestionResponse(BaseModel):
    session_id: str
    question_id: str
    question: str
    skill: str
    difficulty: str
    question_number: int
    total_questions: int
    finished: bool = False


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str
    fluency_score: Optional[float] = Field(
        default=None,
        description="0-1 delivery score from the speaking analyser, audio mode only",
    )


class ScoreBreakdown(BaseModel):
    keyword_score: float
    cosine_score: float
    llm_score: float
    final_score: float
    feedback: str
    # Delivery, not content. Reported separately on purpose: a nervous answer
    # should not lower a technically correct score, but the candidate still
    # needs to see it. Only present in audio mode.
    fluency_score: Optional[float] = None


class AnswerResponse(BaseModel):
    session_id: str
    scores: ScoreBreakdown
    next_difficulty: str
    questions_asked: int
    finished: bool


# ------------------------------------------------------------ report
class QuestionRecord(BaseModel):
    question: str
    answer: str
    skill: str
    difficulty: str
    scores: Dict[str, float]
    feedback: str


class ReportSummary(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    improvement_tips: List[str]
    overall_recommendation: str


class ReportResponse(BaseModel):
    session_id: str
    candidate_name: str
    overall_score: float
    per_skill_scores: Dict[str, float]
    questions: List[QuestionRecord]
    summary: ReportSummary
