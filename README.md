# Intelligent Candidate Discovery & Ranking — Redrob AI Hackathon

**Team:** Team Arcturus 
**Contact:** bhargav.incredibl@gmail.com 
**Challenge:** Rank top 100 candidates from 100K profiles for the *Senior AI Engineer (Founding Team)* JD

---

## Architecture

A **multi-signal weighted scoring system** that goes beyond keyword matching to understand semantic fit between a candidate's career substance and the JD's nuanced requirements.

### Scoring Pipeline

| Component | Weight | What it measures |
|---|---|---|
| Role substance | 30% | Is this person actually in AI/ML/Search engineering? (not just listing AI skills) |
| Technical skills | 25% | Weighted Tier-1/2/3 skill match with proficiency + duration trust |
| Experience quality | 20% | Right YoE band (5–9yr) + product company background + GitHub |
| Behavioral signals | 15% | Active, responsive, short notice period |
| Location fit | 10% | Pune/Noida (1.0) → Delhi NCR/Mumbai (0.9–0.95) → Hyderabad/Bengaluru (0.85) |

### Key Design Decisions

**Role substance over keyword stuffing (30% weight):**  
The JD explicitly warns that "a Marketing Manager listing AI skills is not a fit." The system analyzes career title history and role descriptions for *production ML substance* — words like "production", "deployed", "real users", "A/B test", "serving" — not just skill-list keywords.

**Tier-1 skill taxonomy:**  
Skills are classified into three tiers matching the JD's "absolutely need / would like / adjacent" breakdown. Tier-1 includes embeddings/vector-DB infrastructure, ranking evaluation metrics (NDCG/MRR/MAP), and Python. Proficiency and duration_months both scale the contribution to prevent zero-duration "expert" padding.

**Consulting-only career penalty:**  
JD disqualifies candidates whose entire career is at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini. The system detects consulting fraction and applies a 0.25× penalty if >95% of career is at these firms.

**Behavioral availability as a multiplier:**  
A perfect-on-paper candidate with 0% recruiter response rate and inactive for 6 months scores low in behavioral (15% weight). The behavioral component weights: open_to_work (30%), recency (25%), response rate (20%), notice period (15%), interview completion + offer acceptance (8%).

**Honeypot detection:**  
Candidates with 3+ skills at "expert" proficiency but duration_months=0 are disqualified (score = 0). This catches the impossible-profile honeypots without needing external data.

---

## Reproduce the Submission

### Requirements

- Python 3.8+ (no pip install needed — standard library only)

### Generate submission CSV

```bash
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

Or from inside the data directory:

```bash
python rank.py --candidates candidates.jsonl --out ../submission.csv
```

**Runtime:** ~14 seconds on a single CPU core for 100K candidates.  
**Memory:** < 2 GB peak.  
**No network access required.**

### Validate

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

---

## Score Component Details

### Skill Tiers

**Tier 1 (critical — 3× weight):** embeddings, sentence-transformers, FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, vector search, semantic search, hybrid search, information retrieval, BM25, learning to rank, NDCG, MRR, MAP, A/B testing, Python

**Tier 2 (important — 2× weight):** NLP, transformers, BERT, LLM, fine-tuning, LoRA, RAG, PyTorch, TensorFlow, recommendation systems, feature engineering, MLOps, XGBoost, distributed systems

**Tier 3 (adjacent — 1× weight):** machine learning, deep learning, AWS/GCP/Azure, Docker, Kubernetes, SQL, Spark, data engineering

### Anti-Patterns Applied

- Consulting-only career → score × 0.25
- No production deployment signals in ML career → score × 0.70  
- High-proficiency skills with 0 months used → trust multiplier ×0.10 per skill
- Job-hopping (avg tenure <12 months) → experience score × 0.70
- Non-ML current title regardless of skill list → role score capped at 0.05–0.25

---

## File Structure

```
.
├── rank.py                  # Main ranking script (no external dependencies)
├── requirements.txt         # Standard library only
├── README.md
├── submission.csv           # Generated output (top 100 ranked candidates)
├── submission_metadata.yaml # Portal metadata
└── India_runs_data_and_ai_challenge/
    ├── candidates.jsonl     # 100K candidate profiles
    ├── candidate_schema.json
    ├── job_description.docx
    ├── sample_submission.csv
    └── validate_submission.py
```
