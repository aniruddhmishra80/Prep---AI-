"""
The only file in the frontend that knows the backend exists.

Interview line: "The Streamlit app is a thin client. It has no business logic -
every decision happens behind the API. If I ever replace Streamlit with React,
nothing in the backend changes."

Deployment note worth knowing: the API URL is read from Streamlit secrets first,
then the environment, then localhost. So the same file works unchanged on my
laptop and on Streamlit Community Cloud.
"""

import os
import time

import requests

try:
    import streamlit as st
except ImportError:                                 # running outside Streamlit
    st = None


def _setting(key: str, default: str = "") -> str:
    """Streamlit secrets win, then environment variables, then the default."""
    if st is not None:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:                           # no secrets.toml at all
            pass
    return os.getenv(key, default)


# rstrip("/") because "https://x.onrender.com/" + "/health" is a 404
API_URL = _setting("API_URL", "http://127.0.0.1:8000").rstrip("/")

TIMEOUT = 180       # LLM calls are slow, and a cold Render instance is slower
WAKE_TIMEOUT = 90   # a sleeping free instance needs ~50s to come back up


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


def health(wake: bool = True):
    """
    Ping the backend.

    Render's free tier spins the service down after 15 minutes idle and takes
    about 50 seconds to wake. A 10 second timeout here reports "backend down"
    when the backend is merely asleep, so this retries through the cold start
    instead of failing on the first attempt.
    """
    attempts = 3 if wake else 1
    last = None
    for attempt in range(attempts):
        try:
            return _handle(requests.get(f"{API_URL}/health", timeout=WAKE_TIMEOUT))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(3)
    raise ApiError(f"No response from {API_URL} — {last}")


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
