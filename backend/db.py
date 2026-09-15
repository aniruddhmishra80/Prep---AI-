"""
SQLite storage for sessions and answered questions.

Interview line: "Sessions are in SQLite, not a Python dict, so they survive a
server restart and work when uvicorn runs with more than one worker. It's a
single file, zero configuration."

Two tables:
  sessions  - one row per interview
  qa        - one row per answered question, linked by session_id
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    candidate_name    TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    resume_text       TEXT,
    jd_text           TEXT,
    resume_skills     TEXT,      -- json list
    jd_skills         TEXT,      -- json list
    gap               TEXT,      -- json dict
    difficulty_index  INTEGER DEFAULT 1,
    asked_count       INTEGER DEFAULT 0,
    current_question  TEXT       -- json dict of the question awaiting an answer
);

CREATE TABLE IF NOT EXISTS qa (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    question_id   TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    skill         TEXT,
    difficulty    TEXT,
    scores        TEXT,          -- json dict
    feedback      TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


def _now() -> str:
    """UTC timestamp. datetime.utcnow() is deprecated from Python 3.12."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------ sessions
def create_session(
    session_id: str,
    candidate_name: str,
    resume_text: str,
    jd_text: str,
    resume_skills: List[str],
    jd_skills: List[str],
    gap: Dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO sessions
               (session_id, candidate_name, created_at, resume_text, jd_text,
                resume_skills, jd_skills, gap, difficulty_index, asked_count,
                current_question)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                candidate_name,
                _now(),
                resume_text,
                jd_text,
                json.dumps(resume_skills),
                json.dumps(jd_skills),
                json.dumps(gap),
                config.START_DIFFICULTY,
                0,
                None,
            ),
        )


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None

    session = dict(row)
    session["resume_skills"] = json.loads(session["resume_skills"] or "[]")
    session["jd_skills"] = json.loads(session["jd_skills"] or "[]")
    session["gap"] = json.loads(session["gap"] or "{}")
    session["current_question"] = json.loads(session["current_question"] or "null")
    return session


def update_session(session_id: str, **fields: Any) -> None:
    if not fields:
        return
    encoded = {
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in fields.items()
    }
    assignments = ", ".join(f"{key} = ?" for key in encoded)
    with _connect() as conn:
        conn.execute(
            f"UPDATE sessions SET {assignments} WHERE session_id = ?",
            (*encoded.values(), session_id),
        )


# ------------------------------------------------------------ answers
def save_answer(
    session_id: str,
    question_id: str,
    question: str,
    answer: str,
    skill: str,
    difficulty: str,
    scores: Dict[str, float],
    feedback: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO qa
               (session_id, question_id, question, answer, skill, difficulty,
                scores, feedback, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                question_id,
                question,
                answer,
                skill,
                difficulty,
                json.dumps(scores),
                feedback,
                _now(),
            ),
        )


def get_answers(session_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM qa WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    answers = []
    for row in rows:
        record = dict(row)
        record["scores"] = json.loads(record["scores"] or "{}")
        answers.append(record)
    return answers


def delete_session(session_id: str) -> bool:
    """Delete a session and every answer belonging to it."""
    with _connect() as conn:
        conn.execute("DELETE FROM qa WHERE session_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    return cursor.rowcount > 0
