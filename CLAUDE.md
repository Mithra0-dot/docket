# Docket — AI Risk Investigation Platform

## What this is
An AI system for a fintech buildathon (Razorpay AI Buildathon, AI Risk 
Manager track, deadline Sept 5 2026) that investigates ambiguous fraud 
cases and produces grounded, auditable judgments — not just fraud scores.

## Core principle — read this before building anything
Build every feature for real. Never simulate, mock, or hardcode a core 
feature's output to make a demo look finished. If something can't be 
built for real in the time available, say so explicitly rather than 
faking it. I need to defend every claim to a judging panel.

## Stack
- Backend: FastAPI (Python)
- Database: PostgreSQL with pgvector extension (pgvector/pgvector:pg16 
  Docker image)
- ML: scikit-learn / XGBoost for fraud scoring, SHAP for explainability
- LLM: Local Ollama model for the investigation agent (zero-cost, no 
  external API dependency)
- Frontend: React (Vite)
- Deployment: Docker Compose (api, db, ollama, frontend services)
- Dataset: Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)

## Frontend style
Dark aesthetic throughout — dark background, light text, accent colors 
for status/severity (e.g. red/amber for risk tiers, green for cleared, 
teal/blue for neutral data). Clean and minimal, not flashy — this is a 
risk-analyst tool, not a consumer app.

## Two-page dashboard structure
- Page 1 — Case Investigation: queue of ambiguous cases → SHAP breakdown 
  → retrieved precedent cases (via pgvector) → AI investigation memo → 
  follow-up chat
- Page 2 — Risk Overview: fraud-ring network graph, cost-aware impact 
  panel, reliability-by-category tracking, adversarial probe summary

## Feature priority (core vs. stretch)
CORE — must be fully real and working:
1. Fraud scoring model (real metrics, not just "it runs")
2. Tiered escalation (auto-clear / auto-block / ambiguous) based on real 
   score-boundary + SHAP-conflict rules
3. Real SHAP explainability per prediction
4. pgvector precedent retrieval (real similarity search, real past 
   outcomes)
5. LLM investigation memo + follow-up chat, grounded in retrieved data 
   only — never inventing statistics
6. Fraud-ring detection via graph analysis (shared device/IP/payment 
   method)
7. Cost-aware decisioning (weight false positive/negative cost by 
   transaction amount)

STRETCH — build only after core is solid, and it's fine if these end up 
"designed but not fully wired up":
8. Self-reported reliability tracking by case category
9. Adversarial probe (minimal perturbation to flip a decision)

## Process rules
- We're building in stages. After each stage, tell me plainly: what you 
  built, what's real vs. approximated, and any limitations.
- Set up docker-compose from Stage 1 so `docker compose up` works at any 
  point going forward.
- If you hit a real blocker, report the actual error immediately — never 
  silently substitute a fake/simulated version of a feature.