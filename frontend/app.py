"""
Streamlit frontend. Run from the project root:
    streamlit run frontend/app.py

Three screens, driven by st.session_state:
    Setup     - upload resume + JD, see the skill gap
    Interview - one question at a time, typed or spoken
    Report    - scores, per-skill breakdown, written feedback
"""

import json
import tempfile

import streamlit as st

import api_client
import voice

st.set_page_config(page_title="AI Interview Coach", page_icon="🎙", layout="wide")

# ---------------------------------------------------------------- state
DEFAULTS = {
    "session_id": None,
    "candidate_name": "",
    "gap": None,
    "question": None,
    "history": [],
    "finished": False,
    "audio_mode": False,
    "last_transcript": "",
    "last_speaking": None,
    "report": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### AI Interview Coach")
    st.caption("Upload your resume and a job description, then answer questions "
               "generated from the gap between them.")

    try:
        info = api_client.health()
        st.success(f"Backend connected · {info['llm_provider']}")
    except Exception:  # noqa: BLE001
        st.error("Backend not reachable. Start it with:\n\n`uvicorn backend.main:app --reload`")

    if st.session_state.session_id:
        st.markdown("---")
        st.write(f"**Session** `{st.session_state.session_id}`")
        st.write(f"**Answered** {len(st.session_state.history)}")
        if st.button("Start over", use_container_width=True):
            reset()
            st.rerun()

    st.markdown("---")
    st.session_state.audio_mode = st.toggle(
        "Voice mode", value=st.session_state.audio_mode,
        help="Reads questions aloud and records your answer from the microphone. "
             "Works when you run Streamlit locally.",
    )

step = "Setup" if not st.session_state.session_id else ("Report" if st.session_state.report else "Interview")


# ================================================================ SETUP
if step == "Setup":
    st.title("Set up your interview")

    name = st.text_input("Your name", placeholder="Aniruddh Mishra")
    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])

    tab_text, tab_pdf = st.tabs(["Paste the job description", "Upload a JD PDF"])
    with tab_text:
        jd_text = st.text_area("Job description", height=220,
                               placeholder="Paste the full job posting here")
    with tab_pdf:
        jd_file = st.file_uploader("Job description (PDF)", type=["pdf"], key="jdpdf")

    if st.button("Analyse and start", type="primary"):
        missing = []
        if not name.strip():
            missing.append("your name")
        if resume_file is None:
            missing.append("a resume PDF")
        if not jd_text.strip() and jd_file is None:
            missing.append("a job description")

        if missing:
            st.warning(f"Still need {' and '.join(missing)}.")
        else:
            with st.spinner("Reading both documents and comparing skills..."):
                try:
                    result = api_client.upload(
                        candidate_name=name,
                        resume_bytes=resume_file.getvalue(),
                        resume_name=resume_file.name,
                        jd_text=jd_text or "",
                        jd_bytes=jd_file.getvalue() if jd_file else None,
                        jd_name=jd_file.name if jd_file else None,
                    )
                    st.session_state.session_id = result["session_id"]
                    st.session_state.candidate_name = result["candidate_name"]
                    st.session_state.gap = result["gap"]
                    st.rerun()
                except api_client.ApiError as exc:
                    st.error(str(exc))

# ================================================================ INTERVIEW
elif step == "Interview":
    gap = st.session_state.gap or {}

    st.title(f"Interview · {st.session_state.candidate_name}")

    with st.expander("Skill gap from your resume vs the job description", expanded=not st.session_state.history):
        st.metric("Match ratio", f"{int(gap.get('match_ratio', 0) * 100)}%")
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown("**You have, they want**")
        col_a.write(", ".join(gap.get("matched", [])) or "—")
        col_b.markdown("**They want, you're missing**")
        col_b.write(", ".join(gap.get("missing", [])) or "—")
        col_c.markdown("**Extra you bring**")
        col_c.write(", ".join(gap.get("extra", [])) or "—")
        st.caption("Missing skills are asked first — that's where the useful signal is.")

    # fetch a question if none is open
    if st.session_state.question is None and not st.session_state.finished:
        with st.spinner("Writing your next question..."):
            try:
                question = api_client.next_question(st.session_state.session_id)
            except api_client.ApiError as exc:
                st.error(str(exc))
                st.stop()
        if question["finished"]:
            st.session_state.finished = True
        else:
            st.session_state.question = question
            st.session_state.last_transcript = ""
            st.session_state.last_speaking = None
            if st.session_state.audio_mode:
                voice.speak(question["question"])

    if st.session_state.finished:
        st.success("All questions answered.")
        if st.button("Generate my report", type="primary"):
            with st.spinner("Scoring the full transcript..."):
                try:
                    st.session_state.report = api_client.final_report(st.session_state.session_id)
                    st.rerun()
                except api_client.ApiError as exc:
                    st.error(str(exc))

    elif st.session_state.question:
        question = st.session_state.question
        st.progress(
            (question["question_number"] - 1) / question["total_questions"],
            text=f"Question {question['question_number']} of {question['total_questions']} · "
                 f"{question['skill']} · {question['difficulty']}",
        )
        st.markdown(f"#### {question['question']}")

        # ---------------- voice input
        if st.session_state.audio_mode:
            col_rec, col_up = st.columns(2)
            if col_rec.button("Record my answer", use_container_width=True):
                with st.spinner("Listening... speak now"):
                    transcript = voice.listen()
                if transcript:
                    st.session_state.last_transcript = transcript
                    st.session_state.last_speaking = voice.analyse_speaking(transcript)
                else:
                    st.warning("Nothing was captured. Type your answer instead.")

            audio_file = col_up.file_uploader("or upload a WAV recording", type=["wav"], key="wav")
            if audio_file is not None and not st.session_state.last_transcript:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_file.getvalue())
                    transcript = voice.transcribe_file(tmp.name)
                if transcript:
                    st.session_state.last_transcript = transcript
                    st.session_state.last_speaking = voice.analyse_speaking(transcript)

        answer = st.text_area(
            "Your answer",
            value=st.session_state.last_transcript,
            height=180,
            placeholder="Type your answer, or turn on voice mode in the sidebar to speak it.",
        )

        if st.session_state.last_speaking:
            speaking = st.session_state.last_speaking
            st.caption(
                f"Delivery · {speaking['word_count']} words · "
                f"{speaking['filler_count']} filler words · "
                f"fluency {speaking['fluency_score']} — {speaking['note']}"
            )

        if st.button("Submit answer", type="primary", disabled=not answer.strip()):
            with st.spinner("Scoring your answer..."):
                try:
                    result = api_client.submit_answer(
                        st.session_state.session_id,
                        question["question_id"],
                        answer,
                        fluency_score=(st.session_state.last_speaking or {}).get("fluency_score"),
                    )
                except api_client.ApiError as exc:
                    st.error(str(exc))
                    st.stop()

            scores = result["scores"]
            st.session_state.history.append({"question": question["question"], "scores": scores})
            st.session_state.question = None
            st.session_state.finished = result["finished"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Keyword", scores["keyword_score"])
            col2.metric("Similarity", scores["cosine_score"])
            col3.metric("Rubric", scores["llm_score"])
            col4.metric("Final", scores["final_score"])
            st.info(scores["feedback"])
            st.caption(f"Next question difficulty: {result['next_difficulty']}")
            st.button("Continue")


# ================================================================ REPORT
elif step == "Report":
    report = st.session_state.report
    st.title(f"Report · {report['candidate_name']}")

    st.metric("Overall score", f"{report['overall_score']:.2f} / 1.00")
    st.markdown(f"**{report['summary']['overall_recommendation']}**")

    st.markdown("#### Score by skill")
    st.bar_chart(report["per_skill_scores"])

    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("#### What went well")
        for item in report["summary"]["strengths"]:
            st.markdown(f"- {item}")
    with col_w:
        st.markdown("#### What to work on")
        for item in report["summary"]["weaknesses"]:
            st.markdown(f"- {item}")

    st.markdown("#### Do this next")
    for tip in report["summary"]["improvement_tips"]:
        st.markdown(f"- {tip}")

    st.markdown("#### Full transcript")
    for index, record in enumerate(report["questions"], start=1):
        with st.expander(f"Q{index} · {record['skill']} · {record['scores'].get('final_score', 0)}"):
            st.markdown(f"**{record['question']}**")
            st.write(record["answer"])
            st.caption(record["feedback"])

    st.download_button(
        "Download report as JSON",
        data=json.dumps(report, indent=2),
        file_name=f"interview_report_{report['session_id']}.json",
        mime="application/json",
    )
