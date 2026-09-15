"""
Compare resume skills against JD skills.

Interview line: "This is deliberately not machine learning. Once both sides are
clean lists, the comparison is three set operations. The gap is what drives the
whole interview: missing skills get asked first, because that is where the
useful signal is."
"""

from typing import Dict, List


def compute_gap(resume_skills: List[str], jd_skills: List[str]) -> Dict:
    resume_set = {s.lower().strip() for s in resume_skills}
    jd_set = {s.lower().strip() for s in jd_skills}

    matched = resume_set & jd_set          # candidate has it, job wants it
    missing = jd_set - resume_set          # job wants it, candidate lacks it
    extra = resume_set - jd_set            # candidate has it, job did not ask

    match_ratio = round(len(matched) / len(jd_set), 2) if jd_set else 0.0

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "match_ratio": match_ratio,
    }
