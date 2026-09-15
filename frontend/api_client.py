"""
The only file in the frontend that knows the backend exists.

Interview line: "The Streamlit app is a thin client. It has no business logic -
every decision happens behind the API. If I ever replace Streamlit with React,
nothing in the backend changes."
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 120           # LLM calls are slow, don't time out at 10 seconds


class ApiError(Exception):
    pass


def _handle(response: requests.Response):
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text
        raise ApiError(str(detail))
    return response.json()


def health():
    return _handle(requests.get(f"{API_URL}/health", timeout=10))


def upload(candidate_name, resume_bytes, resume_name, jd_text="", jd_bytes=None, jd_name=None):
    files = {"resume": (resume_name, resume_bytes, "application/pdf")}
    if jd_bytes:
        files["jd_file"] = (jd_name, jd_bytes, "application/pdf")
    data = {"candidate_name": candidate_name, "jd_text": jd_text}
    return _handle(requests.post(f"{API_URL}/upload/resume-jd", data=data, files=files, timeout=TIMEOUT))


def next_question(session_id):
    return _handle(requests.get(f"{API_URL}/interview/{session_id}/next-question", timeout=TIMEOUT))


def submit_answer(session_id, question_id, answer, fluency_score=None):
    payload = {
        "session_id": session_id,
        "question_id": question_id,
        "answer": answer,
        "fluency_score": fluency_score,
    }
    return _handle(requests.post(f"{API_URL}/interview/answer", json=payload, timeout=TIMEOUT))


def final_report(session_id):
    return _handle(requests.get(f"{API_URL}/report/{session_id}/final", timeout=TIMEOUT))
