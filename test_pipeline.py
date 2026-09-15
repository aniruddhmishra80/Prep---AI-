"""
Test the logic without starting a server, an LLM or a browser.

    python test_pipeline.py

Interview line: "Because all the logic lives in core/ and not in the routers, I
can test the pure functions directly. Gap analysis, keyword scoring, adaptive
difficulty and the speaking analyser have no LLM dependency at all, so these
tests run offline in under a second."
"""

import sys

from backend.core.gap import compute_gap
from backend.core.interview import next_difficulty_index, pick_skill
from backend.core.scoring import keyword_score
from backend.core.skills import regex_fallback

failures = []


def check(name, got, expected):
    if got == expected:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, expected {expected!r}")
        failures.append(name)


print("\ngap analysis")
gap = compute_gap(["python", "fastapi", "opencv"], ["python", "fastapi", "docker", "aws"])
check("matched", gap["matched"], ["fastapi", "python"])
check("missing", gap["missing"], ["aws", "docker"])
check("extra", gap["extra"], ["opencv"])
check("match ratio", gap["match_ratio"], 0.5)

print("\nskill regex uses word boundaries")
# the classic bug: substring matching makes 'javascript' report 'java'
found = regex_fallback("I build frontends with JavaScript and React.")
check("javascript found", "javascript" in found, True)
check("java NOT falsely found", "java" in found, False)

print("\nkeyword scoring")
check("all hit", keyword_score("I used chunking and embeddings", ["chunking", "embeddings"]), 1.0)
check("half hit", keyword_score("I used chunking", ["chunking", "embeddings"]), 0.5)
check("none hit", keyword_score("no idea", ["chunking", "embeddings"]), 0.0)

print("\nadaptive difficulty  (0=easy 1=medium 2=hard)")
check("strong answer goes up", next_difficulty_index(1, 0.85), 2)
check("weak answer goes down", next_difficulty_index(1, 0.30), 0)
check("middling answer holds", next_difficulty_index(1, 0.55), 1)
check("cannot exceed hard", next_difficulty_index(2, 0.95), 2)
check("cannot go below easy", next_difficulty_index(0, 0.10), 0)

print("\nskill plan asks missing skills first")
check("first skill", pick_skill(gap, 0), "aws")
check("wraps around", pick_skill(gap, 4), "aws")

print("\nspeaking analysis")
from frontend.voice import analyse_speaking  # noqa: E402

clean = analyse_speaking(
    "I built a retrieval pipeline that chunks the resume, embeds each chunk with "
    "MiniLM and stores the vectors in Chroma so the model only sees relevant text."
)
filler = analyse_speaking("um so like uh basically you know um i mean like i sort of um did it")
check("clean speech scores higher", clean["fluency_score"] > filler["fluency_score"], True)
check("fillers counted", filler["filler_count"] > 4, True)

print(f"\n{'all tests passed' if not failures else str(len(failures)) + ' test(s) failed'}\n")
sys.exit(1 if failures else 0)
