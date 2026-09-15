"""
Build the final report.

Interview line: "The numbers are computed in Python - overall average and a
per-skill breakdown grouped from the answer rows. The LLM is used once, at the
end, only for the qualitative part: strengths, weaknesses, improvement tips and
a one-line fit recommendation. One call, not one per question."
"""

from collections import defaultdict
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.llm import get_llm


class ReportSummary(BaseModel):
    strengths: List[str] = Field(description="2 to 4 specific strengths")
    weaknesses: List[str] = Field(description="2 to 4 specific weak areas")
    improvement_tips: List[str] = Field(description="3 concrete, actionable tips")
    overall_recommendation: str = Field(
        description="One sentence, e.g. 'Good fit for an SDE fresher role, needs work on system design'"
    )


summary_parser = PydanticOutputParser(pydantic_object=ReportSummary)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are writing feedback for a candidate after a mock technical interview.

Role targeted (from the job description skills): {jd_skills}
Overall score: {overall_score} out of 1.0

Transcript with scores:
{transcript}

Write honest, specific feedback. Refer to what the candidate actually said, not
generic advice.

{format_instructions}"""
).partial(format_instructions=summary_parser.get_format_instructions())


def per_skill_scores(answers: List[Dict]) -> Dict[str, float]:
    buckets = defaultdict(list)
    for record in answers:
        buckets[record.get("skill") or "general"].append(
            record.get("scores", {}).get("final_score", 0.0)
        )
    return {skill: round(sum(v) / len(v), 3) for skill, v in buckets.items()}


def overall_score(answers: List[Dict]) -> float:
    scores = [r.get("scores", {}).get("final_score", 0.0) for r in answers]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def build_summary(answers: List[Dict], jd_skills: List[str], overall: float) -> Dict:
    transcript = "\n\n".join(
        f"Q{i}: {r['question']}\nA: {r['answer']}\n"
        f"Score: {r.get('scores', {}).get('final_score', 0)}"
        for i, r in enumerate(answers, start=1)
    )

    chain = SUMMARY_PROMPT | get_llm() | summary_parser
    try:
        result = chain.invoke(
            {
                "jd_skills": ", ".join(jd_skills) or "not specified",
                "overall_score": overall,
                "transcript": transcript or "No answers recorded.",
            }
        )
        return result.model_dump()
    except Exception as exc:                       # noqa: BLE001
        print(f"[report] summary failed ({exc})")
        return {
            "strengths": ["Completed the interview."],
            "weaknesses": ["Automatic summary was unavailable."],
            "improvement_tips": ["Re-run the report to generate feedback."],
            "overall_recommendation": f"Overall score {overall}.",
        }
