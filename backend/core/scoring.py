"""
Score a free-text answer with three independent signals.

Interview line, and this is the heart of the project: "There is no answer key.
An open-ended spoken answer can be right in a hundred phrasings, so there is
nothing to diff against. I stopped hunting for one perfect metric and combined
three imperfect ones that fail in different ways:

  keyword (20%) - cheap, deterministic, transparent.
                  Fails on paraphrase: right idea, different words, score zero.
  cosine  (30%) - catches meaning across different wording.
                  Fails on confident nonsense that uses the right vocabulary.
  llm     (50%) - the only signal that can judge whether the answer is correct.
                  Fails by being non-deterministic and costing a call.

Weighting them means no single failure mode decides the score. The LLM carries
the most weight because it is the only one that can say 'that is wrong'.

Honest caveat I volunteer before being asked: the weights are reasoned, not
tuned. Validating them properly needs a set of human-scored answers to fit
against, which I did not have."

Why cosine and not Euclidean distance? Cosine measures the angle between two
vectors and ignores magnitude. A long answer and a short answer that say the
same thing point the same direction but have different lengths. Euclidean would
punish the length difference. The question is what was said, not how much.
"""

from typing import Dict, List

import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend import config
from backend.llm import get_embeddings, get_llm


# ---------------------------------------------------------------- 1. keyword
def keyword_score(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of the expected keywords that appear in the answer."""
    if not expected_keywords:
        return 0.0
    lowered = (answer or "").lower()
    hits = sum(1 for kw in expected_keywords if kw and kw.lower() in lowered)
    return round(hits / len(expected_keywords), 3)


# ---------------------------------------------------------------- 2. cosine
def cosine_score(answer: str, ideal_answer: str) -> float:
    """Cosine similarity between the answer and the model answer, clamped to 0-1."""
    if not answer.strip() or not ideal_answer.strip():
        return 0.0
    try:
        embeddings = get_embeddings()
        vectors = embeddings.embed_documents([answer, ideal_answer])
        a, b = np.array(vectors[0]), np.array(vectors[1])
        similarity = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return round(max(0.0, min(1.0, similarity)), 3)
    except Exception as exc:                       # noqa: BLE001
        print(f"[scoring] cosine failed ({exc})")
        return 0.0


# ---------------------------------------------------------------- 3. llm rubric
class RubricScore(BaseModel):
    score: float = Field(description="0 to 10, how good the answer is")
    feedback: str = Field(description="Two sentences of specific feedback")


rubric_parser = PydanticOutputParser(pydantic_object=RubricScore)

RUBRIC_PROMPT = ChatPromptTemplate.from_template(
    """You are grading one interview answer. Be strict but fair.

Question: {question}
Reference answer: {ideal_answer}
Candidate answer: {answer}

Grade on correctness, depth and clarity. Score 0-10:
0-3 wrong or empty, 4-6 partially correct or shallow, 7-8 solid,
9-10 correct with depth and a concrete example.

{format_instructions}"""
).partial(format_instructions=rubric_parser.get_format_instructions())


def llm_score(question: str, answer: str, ideal_answer: str) -> Dict:
    chain = RUBRIC_PROMPT | get_llm() | rubric_parser
    try:
        result = chain.invoke(
            {"question": question, "answer": answer, "ideal_answer": ideal_answer}
        )
        return {
            "score": round(max(0.0, min(10.0, result.score)) / 10, 3),
            "feedback": result.feedback.strip(),
        }
    except Exception as exc:                       # noqa: BLE001
        print(f"[scoring] llm rubric failed ({exc})")
        return {"score": 0.0, "feedback": "Automatic grading was unavailable for this answer."}


# ---------------------------------------------------------------- aggregator
def score_answer(question: Dict, answer: str) -> Dict:
    """Run all three scorers and blend them into one final score."""
    keyword = keyword_score(answer, question.get("expected_keywords", []))
    cosine = cosine_score(answer, question.get("ideal_answer", ""))
    rubric = llm_score(question.get("question", ""), answer, question.get("ideal_answer", ""))

    final = (
        config.WEIGHT_KEYWORD * keyword
        + config.WEIGHT_COSINE * cosine
        + config.WEIGHT_LLM * rubric["score"]
    )

    return {
        "keyword_score": keyword,
        "cosine_score": cosine,
        "llm_score": rubric["score"],
        "final_score": round(final, 3),
        "feedback": rubric["feedback"],
    }
