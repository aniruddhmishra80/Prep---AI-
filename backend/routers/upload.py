"""
POST /upload/resume-jd

Interview line: "Routers do request/response translation only. All the logic
lives in core/, so I can unit-test every step without starting an HTTP server."
"""

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend import config, db
from backend.core import gap as gap_module
from backend.core import parser, rag, skills
from backend.schemas import UploadResponse

router = APIRouter()


@router.post("/resume-jd", response_model=UploadResponse)
async def upload_resume_and_jd(
    candidate_name: str = Form(...),
    resume: UploadFile = File(...),
    jd_text: str = Form(""),
    jd_file: UploadFile | None = File(None),
):
    """Upload a resume PDF plus a JD (pasted text or a second PDF)."""
    session_id = str(uuid.uuid4())[:8]

    # 1. save and parse the resume
    resume_path = parser.save_upload(
        await resume.read(), f"{session_id}_resume.pdf", config.UPLOAD_DIR
    )
    resume_content = parser.load_pdf_text(resume_path)
    if not resume_content:
        raise HTTPException(
            status_code=400,
            detail="No text found in the resume PDF. It may be a scanned image, which needs OCR.",
        )

    # 2. the JD is either a pasted string or a second PDF
    if jd_file is not None:
        jd_path = parser.save_upload(
            await jd_file.read(), f"{session_id}_jd.pdf", config.UPLOAD_DIR
        )
        jd_content = parser.load_pdf_text(jd_path)
    else:
        jd_content = parser.normalise(jd_text)

    if not jd_content:
        raise HTTPException(status_code=400, detail="Provide a job description as text or PDF.")

    # 3. skills out of both documents
    resume_skills = skills.extract_skills(resume_content, doc_type="resume")
    jd_skills = skills.extract_skills(jd_content, doc_type="job description")

    # 4. gap analysis
    gap_result = gap_module.compute_gap(resume_skills, jd_skills)

    # 5. persist the session
    db.create_session(
        session_id=session_id,
        candidate_name=candidate_name,
        resume_text=resume_content,
        jd_text=jd_content,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        gap=gap_result,
    )

    # 6. build the vector store for this session
    rag.index_documents(session_id, resume_content, jd_content)

    return UploadResponse(
        session_id=session_id,
        candidate_name=candidate_name,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        gap=gap_result,
        max_questions=config.MAX_QUESTIONS,
    )
