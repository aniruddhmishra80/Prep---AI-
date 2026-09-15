"""
Pull a clean skill list out of resume text and JD text.

Interview line: "The skills come from the LLM, not a hardcoded vocabulary, so a
skill I never thought of still gets picked up. I use PydanticOutputParser, so
the model is told the exact JSON schema to return and LangChain validates the
response against it. If the model fails or the API is down, a small
word-boundary regex fallback keeps the app usable."

The regex uses \\b word boundaries on purpose: plain substring matching makes
'java' match inside 'javascript' and report Java as a skill the candidate has.
"""

import re
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.llm import get_llm

MAX_CHARS = 6000        # keep the prompt small and cheap


class SkillList(BaseModel):
    """The shape we force the model to answer in."""

    skills: List[str] = Field(
        description="Lowercase technical skills, tools, frameworks and concepts"
    )


parser = PydanticOutputParser(pydantic_object=SkillList)

PROMPT = ChatPromptTemplate.from_template(
    """You are a technical recruiter reading a {doc_type}.

Extract the technical skills it mentions.

Rules:
- lowercase only
- no duplicates
- languages, frameworks, libraries, tools, databases, cloud services and
  technical concepts only
- no soft skills, no job titles, no company names
- at most 25 skills

{format_instructions}

{doc_type} text:
{text}"""
).partial(format_instructions=parser.get_format_instructions())


def extract_skills(text: str, doc_type: str = "resume") -> List[str]:
    """LLM extraction, with a regex fallback if anything goes wrong."""
    chain = PROMPT | get_llm() | parser          # this is the whole LCEL chain
    try:
        result = chain.invoke({"text": text[:MAX_CHARS], "doc_type": doc_type})
        skills = {s.strip().lower() for s in result.skills if s and s.strip()}
        if skills:
            return sorted(skills)
    except Exception as exc:                      # noqa: BLE001
        print(f"[skills] LLM extraction failed ({exc}); using regex fallback")
    return regex_fallback(text)


# ---------------------------------------------------------------- fallback
FALLBACK_VOCABULARY = [
    "python", "java", "javascript", "typescript", "c++", "sql", "html", "css",
    "react", "node.js", "fastapi", "flask", "django", "streamlit",
    "langchain", "rag", "llm", "prompt engineering", "vector database",
    "chromadb", "faiss", "pinecone", "embeddings",
    "machine learning", "deep learning", "nlp", "cnn", "transformers",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "opencv",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux",
    "mongodb", "postgresql", "mysql", "redis", "rest api", "microservices",
]


def regex_fallback(text: str) -> List[str]:
    """Word-boundary matching against a small known vocabulary."""
    lowered = (text or "").lower()
    found = [
        skill
        for skill in FALLBACK_VOCABULARY
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", lowered)
    ]
    return sorted(set(found))
