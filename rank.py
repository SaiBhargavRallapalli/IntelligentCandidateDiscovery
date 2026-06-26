#!/usr/bin/env python3
"""
Intelligent Candidate Discovery & Ranking System
Redrob AI Hackathon — Sai Bhargav Rallapalli

Architecture: Multi-signal weighted scoring with semantic role analysis.
No external API calls. Runs on CPU in under 2 minutes for 100K candidates.

Scoring pipeline (weights):
  Role substance      30% — Is this person actually in ML/AI/Search engineering?
  Technical skills    25% — Weighted Tier-1/2/3 skill match with proficiency scaling
  Experience quality  20% — Right YoE band + product company background
  Behavioral signals  15% — Active, responsive, short notice period
  Location fit        10% — Pune/Noida preferred; NCR/Mumbai/Hyderabad acceptable

Anti-pattern detection applied before scoring:
  - Consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) → heavy penalty
  - Keywords in skills but career history contradicts them → trust multiplier penalty
  - Honeypot patterns (expert + 0 months used) → forced disqualification
  - Non-ML title with no AI substance in career → low role score regardless of skill list
"""

import json
import re
import sys
import argparse
import csv
from datetime import datetime, date
from typing import Dict, List, Tuple

# ── Skill Taxonomy ──────────────────────────────────────────────────────────────

# Tier 1 — JD explicitly calls these "things you absolutely need"
TIER1_SKILLS = {
    # Embeddings & retrieval infrastructure
    "embeddings", "embedding", "dense retrieval", "sentence transformers",
    "sentence-transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector database", "vector db", "vector search",
    "semantic search", "hybrid search", "information retrieval", "bm25",
    "dense passage retrieval", "dpr", "bi-encoder", "cross-encoder", "neural retrieval",
    "retrieval", "e5", "bge", "openai embeddings",
    # Ranking evaluation (JD specifically calls out NDCG, MRR, MAP)
    "learning to rank", "ltr", "ndcg", "mrr", "map", "mean average precision",
    "ranking evaluation", "offline evaluation", "a/b testing", "online evaluation",
    "precision@k", "recall@k",
    # Python (explicitly mentioned as important in JD)
    "python",
}

# Tier 2 — JD "would like but won't reject for"
TIER2_SKILLS = {
    "nlp", "natural language processing", "transformers", "bert", "roberta", "gpt",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft", "finetuning",
    "rag", "retrieval augmented generation", "llama", "mistral", "prompt engineering",
    "pytorch", "tensorflow", "hugging face", "huggingface", "keras",
    "recommendation system", "recommendation", "recommender", "collaborative filtering",
    "feature engineering", "model evaluation", "xgboost", "gradient boosting", "lightgbm",
    "mlops", "model serving", "production ml", "ml infrastructure",
    "distributed systems", "inference optimization", "model compression",
    "spacy", "gensim", "nltk", "text classification", "knowledge graph",
}

# Tier 3 — adjacent / infrastructure
TIER3_SKILLS = {
    "machine learning", "deep learning", "neural network", "scikit-learn", "sklearn",
    "search", "data science", "statistics", "probability",
    "aws", "gcp", "azure", "cloud", "kubernetes", "docker",
    "spark", "hadoop", "data engineering", "data pipeline",
    "sql", "postgresql", "redis", "kafka", "airflow",
}

# Skills with no value for this role — list keywords that indicate off-target profile
NEGATIVE_SKILL_INDICATORS = {
    "photoshop", "illustrator", "figma", "ux design", "ui design",
    "excel", "word", "powerpoint", "accounting", "tally",
    "sales", "marketing", "seo", "content writing", "social media",
    "autocad", "solidworks", "matlab (only)",
}

# ── Role / Career Taxonomy ──────────────────────────────────────────────────────

STRONG_ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "artificial intelligence engineer", "data scientist", "applied scientist",
    "nlp engineer", "search engineer", "recommendation engineer",
    "ranking engineer", "mlops engineer", "research scientist",
    "deep learning engineer", "applied ml engineer", "senior ai engineer",
    "principal ml", "staff ml", "ml research", "ai researcher",
}

MODERATE_ML_TITLES = {
    "software engineer", "backend engineer", "senior engineer", "staff engineer",
    "principal engineer", "tech lead", "data engineer", "platform engineer",
    "computer vision engineer",  # adjacent, not ideal
}

NEGATIVE_TITLES = {
    "marketing", "hr manager", "human resources", "accountant", "accountant",
    "sales executive", "sales manager", "customer support", "graphic designer",
    "content writer", "operations manager", "civil engineer",
    "mechanical engineer", "electrical engineer", "supply chain",
    "logistics", "finance", "project manager",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "l&t infotech", "lt infotech", "syntel", "kpit",
    "igate", "niit technologies", "mastech", "zensar", "birlasoft",
}

# ── Location Scoring ────────────────────────────────────────────────────────────

LOCATION_SCORES = {
    "pune": 1.0, "noida": 1.0,
    "delhi": 0.95, "new delhi": 0.95, "gurgaon": 0.95, "gurugram": 0.95,
    "greater noida": 0.95, "faridabad": 0.85,
    "mumbai": 0.90, "navi mumbai": 0.85, "thane": 0.80,
    "bengaluru": 0.85, "bangalore": 0.85,
    "hyderabad": 0.85,
    "chennai": 0.50,
    "kolkata": 0.40,
    "ahmedabad": 0.35,
    "indore": 0.30, "jaipur": 0.30, "bhopal": 0.25,
    "vizag": 0.25, "visakhapatnam": 0.25,
}

PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}

# Production ML keywords in role descriptions (signals real deployment experience)
PRODUCTION_KEYWORDS = {
    "production", "deployed", "real users", "at scale", "serving", "inference",
    "latency", "throughput", "sla", "a/b test", "experiment", "rollout",
    "billion", "million requests", "k rps", "online system",
}

DESCRIPTION_ML_KEYWORDS = {
    "embedding", "retrieval", "ranking", "recommendation", "search",
    "vector", "similarity", "nlp", "language model", "transformer",
    "machine learning", "deep learning", "model", "ml pipeline",
    "feature store", "training pipeline", "inference",
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def norm(text: str) -> str:
    return text.lower().strip()


def contains_any(text: str, keywords: set) -> bool:
    t = norm(text)
    return any(kw in t for kw in keywords)


def count_matches(text: str, keywords: set) -> int:
    t = norm(text)
    return sum(1 for kw in keywords if kw in t)


def days_since(date_str: str, ref: date = None) -> int:
    if not date_str:
        return 9999
    ref = ref or date.today()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (ref - d).days
    except Exception:
        return 9999


# ── Honeypot Detection ──────────────────────────────────────────────────────────

def honeypot_penalty(c: dict) -> float:
    """
    Returns a penalty multiplier (0.0 = disqualify, 1.0 = clean).
    Detects impossible profiles as described in submission_spec.
    """
    skills = c.get("skills", [])
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    # Pattern 1: Expert proficiency with 0 months used across many skills
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3:
        return 0.0  # Honeypot — disqualify

    # Pattern 2: profile YoE vs sum of career durations wildly inconsistent
    declared_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if declared_yoe > 0:
        implied_years = total_career_months / 12.0
        # If claimed YoE is >3x more than career durations, suspicious
        if implied_years > 0 and declared_yoe > implied_years * 3.5:
            return 0.1

    # Pattern 3: many "advanced"/"expert" skills but all with duration_months=0
    high_prof_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", 1) == 0
    )
    if high_prof_zero >= 5:
        return 0.15

    return 1.0


# ── Location Score ──────────────────────────────────────────────────────────────

def score_location(c: dict) -> float:
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    country = norm(profile.get("country", ""))
    location = norm(profile.get("location", ""))
    willing = signals.get("willing_to_relocate", False)

    # Check location against known scores
    for city, score in LOCATION_SCORES.items():
        if city in location:
            return score

    # India but unlisted city
    if country == "india":
        return 0.30 if not willing else 0.35

    # International: only credit if willing to relocate
    if willing:
        return 0.30
    return 0.05


# ── Skills Match Score ──────────────────────────────────────────────────────────

def score_skills(c: dict) -> Tuple[float, int, int, List[str]]:
    """
    Returns (normalized_score, tier1_count, total_ai_count, matched_t1_skills).
    Weights: Tier1 × 3, Tier2 × 2, Tier3 × 1.
    Proficiency multiplier applied per skill.
    Duration multiplier: skills with 0 months are heavily discounted.
    Also pulls validated scores from skill_assessment_scores.
    """
    skills = c.get("skills", [])
    assessment_scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})

    total_weighted = 0.0
    max_possible = 0.0
    tier1_names = []
    tier1_count = 0
    total_ai_count = 0

    for s in skills:
        name = norm(s.get("name", ""))
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)

        # Determine tier
        if any(t1 in name for t1 in TIER1_SKILLS):
            tier = 3
            tier1_names.append(s["name"])
            tier1_count += 1
            total_ai_count += 1
        elif any(t2 in name for t2 in TIER2_SKILLS):
            tier = 2
            total_ai_count += 1
        elif any(t3 in name for t3 in TIER3_SKILLS):
            tier = 1
            total_ai_count += 1
        else:
            continue  # Non-relevant skill

        # Check if negatively indicating (someone who listed irrelevant skills gets no credit)
        if contains_any(name, NEGATIVE_SKILL_INDICATORS):
            continue

        prof_w = PROFICIENCY_WEIGHT.get(prof, 0.15)

        # Duration trust multiplier: 0 months = 0.1x, <6mo = 0.5x, >=12mo = 1.0x
        if duration == 0:
            dur_w = 0.10
        elif duration < 6:
            dur_w = 0.50
        elif duration < 12:
            dur_w = 0.75
        else:
            dur_w = 1.0

        # Boost if validated assessment score exists
        assessment_boost = 1.0
        for assessed_skill, assessed_score in assessment_scores.items():
            if any(part in norm(assessed_skill) for part in name.split()):
                if assessed_score >= 80:
                    assessment_boost = 1.25
                elif assessed_score >= 60:
                    assessment_boost = 1.10
                break

        contribution = tier * prof_w * dur_w * assessment_boost
        max_single = tier * 1.0 * 1.0 * 1.0  # max possible per skill
        total_weighted += contribution
        max_possible += max_single

    # Normalize against a "ideal candidate" baseline of 8 Tier-1 skills at expert/12mo
    ideal_baseline = 8 * 3 * 1.0 * 1.0  # 8 × tier3 × expert × 12mo+
    score = min(total_weighted / ideal_baseline, 1.0) if ideal_baseline > 0 else 0.0

    # Penalty if has many negative (off-role) skills
    neg_skills = sum(1 for s in skills if contains_any(norm(s.get("name", "")), NEGATIVE_SKILL_INDICATORS))
    if neg_skills >= 3:
        score *= 0.85

    return score, tier1_count, total_ai_count, tier1_names[:5]


# ── Role Substance Score ────────────────────────────────────────────────────────

def score_role_substance(c: dict) -> Tuple[float, str]:
    """
    Evaluates whether the candidate has genuine AI/ML/Search engineering experience.
    Goes beyond title keywords to inspect career descriptions.
    Returns (score, key_signal_string).
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    current_title = norm(profile.get("current_title", ""))
    current_industry = norm(profile.get("current_industry", ""))

    # 1. Check current title
    title_is_strong_ml = any(kw in current_title for kw in STRONG_ML_TITLES)
    title_is_moderate = any(kw in current_title for kw in MODERATE_ML_TITLES)
    title_is_negative = any(kw in current_title for kw in NEGATIVE_TITLES)

    # 2. Consulting-only career check
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(firm in norm(r.get("company", "")) for firm in CONSULTING_FIRMS)
    )
    consulting_fraction = consulting_months / max(total_months, 1)

    # 3. Analyze career descriptions for production ML substance
    all_descriptions = " ".join(r.get("description", "") for r in career).lower()
    production_hits = count_matches(all_descriptions, PRODUCTION_KEYWORDS)
    ml_substance_hits = count_matches(all_descriptions, DESCRIPTION_ML_KEYWORDS)

    # 4. Fraction of career spent in ML/AI roles
    ml_career_months = 0
    ml_role_companies = []
    for role in career:
        role_title = norm(role.get("title", ""))
        if any(kw in role_title for kw in STRONG_ML_TITLES):
            ml_career_months += role.get("duration_months", 0)
            ml_role_companies.append(role.get("company", "?"))
        elif any(kw in role_title for kw in MODERATE_ML_TITLES):
            # Moderate title: credit 50% if descriptions show ML substance
            desc = norm(role.get("description", ""))
            ml_desc_hits = count_matches(desc, DESCRIPTION_ML_KEYWORDS)
            if ml_desc_hits >= 3:
                ml_career_months += role.get("duration_months", 0) * 0.7
            elif ml_desc_hits >= 1:
                ml_career_months += role.get("duration_months", 0) * 0.3

    ml_fraction = ml_career_months / max(total_months, 1)

    # ── Compute base score ──
    if title_is_strong_ml:
        base = 0.90
        key_signal = f"AI/ML role: {profile.get('current_title')}"
    elif title_is_moderate and ml_fraction >= 0.4:
        base = 0.70
        key_signal = f"Eng role with strong ML career ({ml_fraction:.0%} time)"
    elif title_is_moderate and ml_substance_hits >= 5:
        base = 0.55
        key_signal = f"Eng background; ML substance in descriptions"
    elif title_is_negative:
        base = 0.05
        key_signal = f"Non-ML role ({profile.get('current_title')}); likely irrelevant"
    else:
        base = 0.25
        key_signal = f"Unclear role ({profile.get('current_title')})"

    # ── Adjustments ──

    # Consulting-only penalty (JD explicitly says this is a disqualifier)
    if consulting_fraction >= 0.95 and total_months > 12:
        base *= 0.25
        key_signal += f"; consulting-only career ({consulting_fraction:.0%})"
    elif consulting_fraction >= 0.70:
        base *= 0.55
        key_signal += f"; mostly consulting ({consulting_fraction:.0%})"

    # Production ML substance bonus
    if production_hits >= 3 and ml_substance_hits >= 5:
        base = min(base * 1.20, 1.0)
    elif production_hits >= 1 and ml_substance_hits >= 3:
        base = min(base * 1.10, 1.0)

    # Research-only penalty (no production deployment signals)
    if ml_substance_hits >= 3 and production_hits == 0 and ml_fraction >= 0.5:
        base *= 0.70  # ML person but no production signals
        key_signal += "; no production deployment signals"

    return min(base, 1.0), key_signal


# ── Experience Quality Score ────────────────────────────────────────────────────

def score_experience(c: dict) -> Tuple[float, str]:
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)

    # YoE band scoring (JD: 5-9 years, sweet spot 6-8)
    if 6 <= yoe <= 8:
        yoe_score = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        yoe_score = 0.90
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.75
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        yoe_score = 0.55
    elif 2 <= yoe < 3:
        yoe_score = 0.35
    elif yoe < 2:
        yoe_score = 0.10
    else:  # >14 years
        yoe_score = 0.45

    # Product company diversity (not just one megacorp or pure services)
    company_sizes = [r.get("company_size", "") for r in career]
    has_startup = any(s in ("1-10", "11-50", "51-200") for s in company_sizes)
    has_mid = any(s in ("201-500", "501-1000") for s in company_sizes)
    has_large = any(s in ("1001-5000", "5001-10000", "10001+") for s in company_sizes)

    # Variety bonus: having both startup and larger company experience
    if has_startup and (has_mid or has_large):
        diversity_bonus = 0.10
    elif has_startup:
        diversity_bonus = 0.05
    else:
        diversity_bonus = 0.0

    # Job-hopping penalty (average tenure <18 months signals title-chaser, per JD)
    if len(career) >= 3:
        avg_tenure = sum(r.get("duration_months", 0) for r in career) / len(career)
        if avg_tenure < 12:
            hopping_penalty = 0.70
        elif avg_tenure < 18:
            hopping_penalty = 0.85
        else:
            hopping_penalty = 1.0
    else:
        hopping_penalty = 1.0

    # GitHub activity bonus (JD says "open-source contributions in AI/ML space")
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 70:
        github_bonus = 0.08
    elif github_score >= 40:
        github_bonus = 0.04
    elif github_score >= 10:
        github_bonus = 0.01
    else:
        github_bonus = 0.0

    # Education tier bonus
    edu = c.get("education", [])
    edu_bonus = 0.0
    for e in edu:
        tier = e.get("tier", "unknown")
        if tier == "tier_1":
            edu_bonus = max(edu_bonus, 0.06)
        elif tier == "tier_2":
            edu_bonus = max(edu_bonus, 0.03)

    raw = (yoe_score + diversity_bonus + github_bonus + edu_bonus) * hopping_penalty
    score = min(raw, 1.0)

    desc = f"{yoe:.1f}yr exp"
    if has_startup:
        desc += "; startup background"
    if github_score >= 40:
        desc += f"; GitHub {github_score:.0f}"

    return score, desc


# ── Behavioral / Availability Score ────────────────────────────────────────────

def score_behavioral(c: dict, ref_date: date = None) -> Tuple[float, str]:
    signals = c.get("redrob_signals", {})
    ref = ref_date or date.today()

    components = {}

    # Open to work (hard signal)
    components["open_to_work"] = 1.0 if signals.get("open_to_work_flag") else 0.20

    # Recency of activity (last seen on platform)
    days_inactive = days_since(signals.get("last_active_date", ""), ref)
    if days_inactive <= 14:
        components["recency"] = 1.0
    elif days_inactive <= 30:
        components["recency"] = 0.90
    elif days_inactive <= 60:
        components["recency"] = 0.70
    elif days_inactive <= 90:
        components["recency"] = 0.45
    elif days_inactive <= 180:
        components["recency"] = 0.20
    else:
        components["recency"] = 0.05

    # Recruiter response rate (JD says active engagement matters)
    rr = signals.get("recruiter_response_rate", 0)
    components["response_rate"] = rr  # Already 0-1

    # Notice period (JD: prefers <30 days, can buy out 30 days)
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:
        components["notice"] = 1.0
    elif notice <= 30:
        components["notice"] = 0.90
    elif notice <= 60:
        components["notice"] = 0.60
    elif notice <= 90:
        components["notice"] = 0.30
    else:
        components["notice"] = 0.05

    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0.5)
    components["interview_completion"] = icr

    # Offer acceptance rate (exclude -1 = no history)
    oar = signals.get("offer_acceptance_rate", -1)
    if oar >= 0:
        components["offer_acceptance"] = oar
    else:
        components["offer_acceptance"] = 0.5  # neutral if no history

    # Profile completeness (signals serious candidate)
    completeness = signals.get("profile_completeness_score", 50) / 100
    components["completeness"] = completeness

    # Weighted combination
    weights = {
        "open_to_work": 0.30,
        "recency": 0.25,
        "response_rate": 0.20,
        "notice": 0.15,
        "interview_completion": 0.05,
        "offer_acceptance": 0.03,
        "completeness": 0.02,
    }

    score = sum(weights[k] * components[k] for k in weights)

    # Build description
    parts = []
    if signals.get("open_to_work_flag"):
        parts.append("open to work")
    if days_inactive <= 30:
        parts.append(f"active {days_inactive}d ago")
    parts.append(f"response rate {rr:.0%}")
    if notice <= 30:
        parts.append(f"notice {notice}d")

    return score, "; ".join(parts)


# ── Main Scoring ────────────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "role": 0.30,
    "skills": 0.25,
    "experience": 0.20,
    "behavioral": 0.15,
    "location": 0.10,
}


def score_candidate(c: dict) -> dict:
    hp = honeypot_penalty(c)
    if hp == 0.0:
        return {
            "candidate_id": c["candidate_id"],
            "final_score": 0.0,
            "role_score": 0.0, "skills_score": 0.0,
            "exp_score": 0.0, "behavioral_score": 0.0,
            "location_score": 0.0,
            "honeypot": True,
            "reasoning": "HONEYPOT: expert skills with 0 months usage — impossible profile",
            "tier1_count": 0, "total_ai_skills": 0,
        }

    role_score, role_signal = score_role_substance(c)
    skills_score, t1_count, ai_count, t1_names = score_skills(c)
    exp_score, exp_signal = score_experience(c)
    behavioral_score, beh_signal = score_behavioral(c)
    location_score = score_location(c)

    final = (
        SCORE_WEIGHTS["role"] * role_score
        + SCORE_WEIGHTS["skills"] * skills_score
        + SCORE_WEIGHTS["experience"] * exp_score
        + SCORE_WEIGHTS["behavioral"] * behavioral_score
        + SCORE_WEIGHTS["location"] * location_score
    ) * hp  # Apply honeypot multiplier (1.0 for clean profiles, <1.0 for suspicious)

    # Build reasoning string
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    title = profile.get("current_title", "?")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "?")
    notice = signals.get("notice_period_days", "?")

    t1_str = ", ".join(t1_names[:3]) if t1_names else "none"
    reasoning = (
        f"{title}, {yoe:.1f}yr, {loc}; "
        f"role={role_score:.2f} [{role_signal[:50]}]; "
        f"skills={skills_score:.2f} [{t1_count} Tier-1: {t1_str}]; "
        f"avail={beh_signal[:60]}; notice={notice}d"
    )

    return {
        "candidate_id": c["candidate_id"],
        "final_score": round(final, 6),
        "role_score": round(role_score, 4),
        "skills_score": round(skills_score, 4),
        "exp_score": round(exp_score, 4),
        "behavioral_score": round(behavioral_score, 4),
        "location_score": round(location_score, 4),
        "honeypot": False,
        "reasoning": reasoning,
        "tier1_count": t1_count,
        "total_ai_skills": ai_count,
    }


# ── Output Generation ───────────────────────────────────────────────────────────

def build_submission_row(result: dict, rank: int) -> dict:
    """Produce the final CSV row with enriched human-readable reasoning."""
    return {
        "candidate_id": result["candidate_id"],
        "rank": rank,
        "score": result.get("csv_score", round(result["final_score"], 4)),
        "reasoning": result["reasoning"],
    }


# ── Entry Point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer JD")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of candidates to output (default: 100)")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...", flush=True)
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates. Scoring...", flush=True)

    results = []
    for i, c in enumerate(candidates):
        results.append(score_candidate(c))
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i+1:,} / {len(candidates):,}...", flush=True)

    print("Ranking...", flush=True)

    # Sort: primary = final_score descending, tie-break = candidate_id ascending
    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    top_n = results[:args.top_n]

    # Round scores to 4dp first, then re-sort so tie-break on candidate_id ascending
    # is consistent with what gets written to the CSV
    for r in top_n:
        r["csv_score"] = round(r["final_score"], 4)
    top_n.sort(key=lambda r: (-r["csv_score"], r["candidate_id"]))

    # Verify scores are non-increasing
    for i in range(len(top_n) - 1):
        assert top_n[i]["csv_score"] >= top_n[i + 1]["csv_score"], \
            f"Score ordering violated at rank {i+1}"

    print(f"Writing top {args.top_n} to {args.out}...", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, result in enumerate(top_n, start=1):
            writer.writerow(build_submission_row(result, rank))

    # Print summary stats for verification
    honeypots_in_top = sum(1 for r in top_n if r.get("honeypot"))
    print(f"\nDone! Summary:")
    print(f"  Total candidates scored: {len(candidates):,}")
    print(f"  Honeypots detected (excluded from top-{args.top_n}): "
          f"{sum(1 for r in results if r.get('honeypot'))}")
    print(f"  Honeypots in top-{args.top_n}: {honeypots_in_top}  (should be 0)")
    print(f"  Score range: {top_n[-1]['final_score']:.4f} – {top_n[0]['final_score']:.4f}")
    print(f"\nTop 10 candidates:")
    for r in top_n[:10]:
        print(f"  #{results.index(r)+1:3d}  {r['candidate_id']}  "
              f"score={r['final_score']:.4f}  "
              f"role={r['role_score']:.2f} skills={r['skills_score']:.2f} "
              f"exp={r['exp_score']:.2f} beh={r['behavioral_score']:.2f} "
              f"loc={r['location_score']:.2f}")
    print(f"\nOutput written to: {args.out}")


if __name__ == "__main__":
    main()
