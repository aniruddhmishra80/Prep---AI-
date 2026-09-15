"""
GET  /interview/{session_id}/next-question
POST /interview/answer

Interview line: "The question loop is bounded by MAX_QUESTIONS, and the answer
endpoint validates that the question_id the client sends matches the question
the server is actually waiting on. Otherwise a stale client could score an
answer against the wrong question."
"""

from fastapi import APIRouter, HTTPException

from backend import config, db
from backend.core import interview as logic
from backend.core import questions, rag, scoring
from backend.schemas import AnswerRequest, AnswerResponse, QuestionResponse, ScoreBreakdown

router = APIRouter()


def _load_session(session_id: str) -> dict:
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@router.get("/{session_id}/next-question", response_model=QuestionResponse)
def next_question(session_id: str):
    session = _load_session(session_id)
    asked = session["asked_count"]

    # hard stop, so the interview actually ends
    if asked >= config.MAX_QUESTIONS:
        return QuestionResponse(
            session_id=session_id,
            question_id="",
            question="Interview complete. Generate your report.",
            skill="",
            difficulty="",
            question_number=asked,
            total_questions=config.MAX_QUESTIONS,
            finished=True,
        )

    skill = logic.pick_skill(session["gap"], asked)
    difficulty = logic.difficulty_label(session["difficulty_index"])

    # RAG: retrieve the chunks most relevant to this skill
    context = rag.retrieve_context(session_id, query=skill)
    history = [record["question"] for record in db.get_answers(session_id)]

    question = questions.generate_question(skill, difficulty, context, history)

    # remember what we asked, including the keywords and ideal answer used to grade it
    db.update_session(session_id, current_question=question)

    return QuestionResponse(
        session_id=session_id,
        question_id=question["question_id"],
        question=question["question"],
        skill=question["skill"],
        difficulty=question["difficulty"],
        question_number=asked + 1,
        total_questions=config.MAX_QUESTIONS,
        finished=False,
    )


@router.post("/answer", response_model=AnswerResponse)
def submit_answer(payload: AnswerRequest):
    session = _load_session(payload.session_id)
    current = session["current_question"]

    if not current:
        raise HTTPException(status_code=400, detail="No question is currently open for this session.")
    if current["question_id"] != payload.question_id:
        raise HTTPException(status_code=409, detail="This answer does not match the open question.")

    scores = scoring.score_answer(current, payload.answer)

    # delivery score from the speaking analyser, audio mode only
    if payload.fluency_score is not None:
        scores["fluency_score"] = round(payload.fluency_score, 3)

    db.save_answer(
        session_id=payload.session_id,
        question_id=payload.question_id,
        question=current["question"],
        answer=payload.answer,
        skill=current["skill"],
        difficulty=current["difficulty"],
        scores=scores,
        feedback=scores["feedback"],
    )

    # adapt: this score decides the next question's difficulty
    new_index = logic.next_difficulty_index(session["difficulty_index"], scores["final_score"])
    asked = session["asked_count"] + 1
    db.update_session(
        payload.session_id,
        difficulty_index=new_index,
        asked_count=asked,
        current_question=None,
    )

    return AnswerResponse(
        session_id=payload.session_id,
        scores=ScoreBreakdown(
            keyword_score=scores["keyword_score"],
            cosine_score=scores["cosine_score"],
            llm_score=scores["llm_score"],
            final_score=scores["final_score"],
            feedback=scores["feedback"],
            fluency_score=scores.get("fluency_score"),
        ),
        next_difficulty=logic.difficulty_label(new_index),
        questions_asked=asked,
        finished=asked >= config.MAX_QUESTIONS,
    )


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """
    Delete a finished interview: the SQLite rows and the Chroma directory.

    Without this, every interview leaves a vector store on disk forever.
    """
    if db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    db.delete_session(session_id)
    rag.delete_store(session_id)
    return {"deleted": session_id}
