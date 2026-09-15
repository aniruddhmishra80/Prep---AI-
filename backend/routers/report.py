"""
GET /report/{session_id}/final
"""

from fastapi import APIRouter, HTTPException

from backend import db
from backend.core import report as report_module
from backend.schemas import QuestionRecord, ReportResponse, ReportSummary

router = APIRouter()


@router.get("/{session_id}/final", response_model=ReportResponse)
def final_report(session_id: str):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    answers = db.get_answers(session_id)
    if not answers:
        raise HTTPException(status_code=400, detail="Answer at least one question first.")

    overall = report_module.overall_score(answers)
    summary = report_module.build_summary(answers, session["jd_skills"], overall)

    return ReportResponse(
        session_id=session_id,
        candidate_name=session["candidate_name"],
        overall_score=overall,
        per_skill_scores=report_module.per_skill_scores(answers),
        questions=[
            QuestionRecord(
                question=record["question"],
                answer=record["answer"],
                skill=record["skill"] or "general",
                difficulty=record["difficulty"] or "medium",
                scores={k: v for k, v in record["scores"].items() if isinstance(v, (int, float))},
                feedback=record["feedback"] or "",
            )
            for record in answers
        ],
        summary=ReportSummary(**summary),
    )
