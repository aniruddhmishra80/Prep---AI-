"""
Generate the next interview question.

Interview line: "The prompt gets four things - the target skill, the difficulty,
the chunks RAG retrieved for that skill, and the last couple of questions so it
doesn't repeat itself. PydanticOutputParser gives the model the exact schema and
validates what comes back. I never assume the model obeys, so the whole call is
wrapped and a templated question takes over on failure. A bad model response
degrades the interview instead of killing it."

The model also returns an ideal_answer. That is what the cosine scorer compares
the candidate against, so cosine measures 'is this close to a correct answer'
rather than 'does this sound like the resume'.
"""

import uuid
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.llm import get_llm


class InterviewQuestion(BaseModel):
    question: str = Field(description="One interview question, no preamble")
    expected_keywords: List[str] = Field(
        description="3 to 6 lowercase keywords a good answer would contain"
    )
    ideal_answer: str = Field(
        description="A strong 3 to 4 sentence model answer to this question"
    )


parser = PydanticOutputParser(pydantic_object=InterviewQuestion)

PROMPT = ChatPromptTemplate.from_template(
    """You are ARIA, a technical interviewer. Ask the candidate one question.

Target skill: {skill}
Difficulty: {difficulty}

Context retrieved from the candidate's resume and the job description:
{context}

Questions already asked in this interview (do not repeat them):
{history}

Rules:
- ask exactly one question about {skill}
- ground it in the candidate's own experience when the context supports it
- {difficulty} difficulty: easy = definition or basic usage, medium = applied
  or comparison, hard = design trade-off, debugging or scale
- keep it under 40 words and conversational, it will be read aloud

{format_instructions}"""
).partial(format_instructions=parser.get_format_instructions())


def generate_question(
    skill: str,
    difficulty: str,
    context: str,
    history: List[str],
) -> Dict:
    """Return a question dict. Always succeeds, one way or another."""
    chain = PROMPT | get_llm() | parser
    try:
        result = chain.invoke(
            {
                "skill": skill,
                "difficulty": difficulty,
                "context": context or "No context retrieved.",
                "history": "\n".join(history[-3:]) or "None yet.",
            }
        )
        return {
            "question_id": str(uuid.uuid4()),
            "question": result.question.strip(),
            "expected_keywords": [k.lower().strip() for k in result.expected_keywords],
            "ideal_answer": result.ideal_answer.strip(),
            "skill": skill,
            "difficulty": difficulty,
        }
    except Exception as exc:                       # noqa: BLE001
        print(f"[questions] generation failed ({exc}); using fallback template")
        return _fallback(skill, difficulty)


def _fallback(skill: str, difficulty: str) -> Dict:
    return {
        "question_id": str(uuid.uuid4()),
        "question": (
            f"Walk me through how you have used {skill}. "
            f"What was the problem and what did you build?"
        ),
        "expected_keywords": [skill.lower()],
        "ideal_answer": (
            f"A strong answer explains a concrete project using {skill}, the "
            f"decisions taken, and the outcome."
        ),
        "skill": skill,
        "difficulty": difficulty,
    }
