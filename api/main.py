import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import joblib
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explain_model import classify
from risk_overview import RiskRow, build_graph, cost_comparison
from train_model import FEATURE_COLUMNS

app = FastAPI(title="Docket API")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/fraud_model.joblib"))


class TransactionInput(BaseModel):
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ExplanationResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    tier: str
    shap_conflict: bool
    top_contributors: list[FeatureContribution]


class PrecedentResponse(BaseModel):
    transaction_id: int
    distance: float
    fraud_label: int
    features: dict[str, float]


class InvestigationResponse(BaseModel):
    transaction_id: int
    memo: str
    model: str
    precedents: list[PrecedentResponse]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    transaction_id: int
    answer: str


class ChatMessage(BaseModel):
    role: str
    content: str


CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@app.on_event("startup")
def ensure_investigation_tables() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        shap_columns = {
            row[0]
            for row in connection.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'shap_values'"""
            ).fetchall()
        }
        if "values" in shap_columns and "contributions" not in shap_columns:
            connection.execute(
                "ALTER TABLE shap_values RENAME COLUMN values TO contributions"
            )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS investigation_memos (
                transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                memo TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS investigation_chat (
                message_id BIGSERIAL PRIMARY KEY,
                transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.commit()


def load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run train_model.py first.",
        )
    return joblib.load(MODEL_PATH)


def vector_literal(values: list[float]) -> str:
    return "[{}]".format(",".join(str(float(value)) for value in values))


def investigation_context(
    transaction_id: int,
) -> tuple[dict[str, object], list[PrecedentResponse], str | None, str | None]:
    feature_sql = ", ".join(FEATURE_COLUMNS)
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        row = connection.execute(
            f"""SELECT t.transaction_id, {feature_sql}, sv.contributions, sv.tier,
                       im.memo, im.model
                FROM transactions t
                JOIN shap_values sv USING (transaction_id)
                LEFT JOIN investigation_memos im USING (transaction_id)
                WHERE t.transaction_id = %s""",
            (transaction_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Ambiguous transaction explanation not found")
        values = list(row[1:31])
        contributions = row[31]
        tier = row[32]
        if tier != "ambiguous":
            raise HTTPException(status_code=400, detail="Investigation is only available for ambiguous transactions")
        connection.execute("SET LOCAL ivfflat.probes = 100")
        precedents = connection.execute(
            f"""SELECT t.transaction_id,
                       ce.embedding <=> %s::vector AS distance, t.class, {feature_sql}
                FROM case_embeddings ce
                JOIN transactions t USING (transaction_id)
                WHERE ce.transaction_id <> %s
                ORDER BY ce.embedding <=> %s::vector
                LIMIT 5""",
            (vector_literal(values), transaction_id, vector_literal(values)),
        ).fetchall()
        if len(precedents) < 5:
            raise HTTPException(status_code=503, detail="Not enough embedded precedent cases")
        history = connection.execute(
            """SELECT role, content FROM investigation_chat
               WHERE transaction_id = %s ORDER BY message_id""",
            (transaction_id,),
        ).fetchall()
    transaction = dict(zip(FEATURE_COLUMNS, values))
    context = {
        "transaction_id": transaction_id,
        "transaction": transaction,
        "shap_top_contributors": [
            {"feature": feature, "shap_value": value}
            for feature, value in sorted(
                contributions.items(), key=lambda item: abs(item[1]), reverse=True
            )[:5]
        ],
    }
    precedent_response = [
        PrecedentResponse(
            transaction_id=precedent[0],
            distance=float(precedent[1]),
            fraud_label=int(precedent[2]),
            features=dict(zip(FEATURE_COLUMNS, precedent[3:])),
        )
        for precedent in precedents
    ]
    context["precedents"] = [precedent.model_dump() for precedent in precedent_response]
    return context, precedent_response, row[33], row[34]


def call_ollama(system: str, user_content: str, history: list[tuple[str, str]] = ()) -> str:
    messages = [{"role": role, "content": content} for role, content in history]
    messages.append({"role": "user", "content": user_content})
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace").strip()
        detail = f"Ollama request failed: HTTP {error.code} {error.reason}"
        if response_body:
            detail += f"; response: {response_body}"
        raise HTTPException(status_code=502, detail=detail) from error
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {error}") from error

    text = body.get("message", {}).get("content") or body.get("response", "")
    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned no response content")
    return str(text).strip()


def call_claude(system: str, user_content: str, history: list[tuple[str, str]] = ()) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
    messages = [{"role": role, "content": content} for role, content in history]
    messages.append({"role": "user", "content": user_content})
    try:
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=system,
            messages=messages,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Anthropic client is not installed") from None
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Claude request failed: {error}") from error
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def call_template_memo(
    system: str, user_content: str, history: list[tuple[str, str]] = ()
) -> str:
    """Create a grounded memo without an external model or generated facts."""
    evidence = json.loads(user_content.split("Evidence:\n", 1)[1])
    transaction = evidence["transaction"]
    contributors = evidence["shap_top_contributors"]
    precedents = evidence["precedents"]
    fraud_count = sum(precedent["fraud_label"] for precedent in precedents)
    precedent_count = len(precedents)
    contributor_lines = "\n".join(
        f"- {item['feature']}: {float(item['shap_value']):+.6f} ({'fraud' if item['shap_value'] > 0 else 'legitimate'} signal)"
        for item in contributors
    ) or "- No SHAP contributors were supplied."
    precedent_rate = fraud_count / precedent_count if precedent_count else 0.0
    return f"""Template-based investigation memo

Transaction: {evidence['transaction_id']}
Amount: {float(transaction['amount']):.2f}

Suspicious signals
{contributor_lines}

What precedents suggest
Of the {precedent_count} most similar past cases, {fraud_count} were confirmed fraud ({precedent_rate:.1%}). These are retrieved historical outcomes, not proof about this transaction.

Recommendation
Escalate for analyst review. This memo was generated deterministically from the supplied transaction, SHAP, and precedent evidence; no LLM call was made."""


logger = logging.getLogger(__name__)
MEMO_GENERATOR = os.getenv("MEMO_GENERATOR", "template").lower()


def call_investigation_memo(
    system: str, user_content: str, history: list[tuple[str, str]] = ()
) -> tuple[str, str]:
    if MEMO_GENERATOR != "ollama":
        return call_template_memo(system, user_content, history), "template-fallback"
    try:
        return call_ollama(system, user_content, history), OLLAMA_MODEL
    except HTTPException as error:
        logger.warning(
            "Ollama investigation failed; using template fallback. Error: %s",
            error.detail,
        )
        return call_template_memo(system, user_content, history), "template-fallback"
    except Exception as error:
        logger.exception(
            "Ollama investigation failed unexpectedly; using template fallback. Error: %s",
            error,
        )
        return call_template_memo(system, user_content, history), "template-fallback"


GROUNDING_SYSTEM = """You are Docket's fraud investigation analyst. Use ONLY the transaction,
SHAP values, and precedent cases supplied in the user message. Do not use outside knowledge
or invent statistics, fraud rates, patterns, or claims. Treat precedent labels as outcomes,
not as proof about the current transaction. If the supplied evidence is insufficient, say so.
Return concise, auditable reasoning and distinguish evidence from uncertainty."""


def save_memo(transaction_id: int, memo: str, model: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            """INSERT INTO investigation_memos (transaction_id, memo, model)
               VALUES (%s, %s, %s)
               ON CONFLICT (transaction_id) DO UPDATE SET memo = EXCLUDED.memo,
                   model = EXCLUDED.model, created_at = CURRENT_TIMESTAMP""",
            (transaction_id, memo, model),
        )
        connection.commit()


@app.get("/health")
def health() -> dict[str, str]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute("SELECT 1")
    return {"status": "ok"}


def risk_rows() -> list[RiskRow]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        rows = connection.execute(
            """SELECT t.transaction_id, t.amount, t.class, rs.score, sv.tier
               FROM transactions t
               JOIN risk_scores rs USING (transaction_id)
               JOIN shap_values sv USING (transaction_id)
               ORDER BY t.transaction_id"""
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=503, detail="Risk scores and explanations are not available")
    return [
        RiskRow(
            transaction_id=int(row[0]),
            amount=float(row[1]),
            label=int(row[2]),
            score=float(row[3]),
            tier=row[4],
        )
        for row in rows
    ]


@app.get("/risk-overview/graph")
def risk_overview_graph(max_nodes: int = 5000) -> dict[str, object]:
    if max_nodes < 1 or max_nodes > 50000:
        raise HTTPException(status_code=400, detail="max_nodes must be between 1 and 50000")
    return build_graph(risk_rows(), max_nodes=max_nodes)


@app.get("/risk-overview/cost")
def risk_overview_cost() -> dict[str, object]:
    return cost_comparison(risk_rows())


@app.get("/cases")
def investigation_cases(limit: int = 100, offset: int = 0) -> dict[str, object]:
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be 1-500 and offset must be non-negative")
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        rows = connection.execute(
            """SELECT t.transaction_id, t.amount, rs.score, sv.tier
               FROM transactions t
               JOIN risk_scores rs USING (transaction_id)
               JOIN shap_values sv USING (transaction_id)
               WHERE sv.tier = 'ambiguous'
               ORDER BY rs.score DESC, t.transaction_id
               LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
        total = connection.execute(
            "SELECT COUNT(*) FROM shap_values WHERE tier = 'ambiguous'"
        ).fetchone()[0]
    return {
        "cases": [
            {"transaction_id": row[0], "amount": float(row[1]), "score": float(row[2]), "tier": row[3]}
            for row in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/score", response_model=ScoreResponse | list[ScoreResponse])
def score_transaction(
    payload: Annotated[TransactionInput | list[TransactionInput], "transaction or batch"]
):
    model = load_model()
    transactions = payload if isinstance(payload, list) else [payload]
    features = [[item.model_dump()[column] for column in FEATURE_COLUMNS] for item in transactions]
    probabilities = model.predict_proba(features)[:, 1].tolist()
    results = [ScoreResponse(fraud_probability=probability) for probability in probabilities]
    return results if isinstance(payload, list) else results[0]


@app.get("/transactions/{transaction_id}/explanation", response_model=ExplanationResponse)
def transaction_explanation(transaction_id: int):
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        row = connection.execute(
            """SELECT rs.score, sv.contributions, sv.tier, sv.conflict
               FROM risk_scores rs JOIN shap_values sv USING (transaction_id)
               WHERE rs.transaction_id = %s""",
            (transaction_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No score and SHAP explanation found for transaction")
    score, values, tier, conflict = row
    top_values = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    return ExplanationResponse(
        transaction_id=transaction_id,
        fraud_probability=score,
        tier=tier,
        shap_conflict=conflict,
        top_contributors=[
            FeatureContribution(
                feature=feature,
                shap_value=value,
                direction="fraud" if value > 0 else "legitimate",
            )
            for feature, value in top_values
        ],
    )


@app.post("/transactions/{transaction_id}/investigate", response_model=InvestigationResponse)
def investigate_transaction(transaction_id: int):
    context, precedents, existing_memo, existing_model = investigation_context(transaction_id)
    memo, model_name = (existing_memo, existing_model) if existing_memo is not None else call_investigation_memo(
        GROUNDING_SYSTEM,
        """Investigate this ambiguous transaction. Produce a short memo with exactly these
sections: Suspicious signals, What precedents suggest, Recommendation. The recommendation
must be only further block, clear, or escalate. Ground every statement in the supplied data.

Evidence:
""" + json.dumps(context, sort_keys=True),
    )
    if existing_memo is None:
        save_memo(transaction_id, memo, model_name)
    return InvestigationResponse(
        transaction_id=transaction_id, memo=memo, model=model_name, precedents=precedents
    )


@app.get("/transactions/{transaction_id}/precedents", response_model=list[PrecedentResponse])
def transaction_precedents(transaction_id: int):
    _, precedents, _, _ = investigation_context(transaction_id)
    return precedents


@app.post("/transactions/{transaction_id}/chat", response_model=ChatResponse)
def investigation_chat(transaction_id: int, payload: ChatRequest):
    context, _, memo, _ = investigation_context(transaction_id)
    if memo is None:
        raise HTTPException(status_code=400, detail="Generate the investigation memo first")
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        history = connection.execute(
            """SELECT role, content FROM investigation_chat
               WHERE transaction_id = %s ORDER BY message_id""",
            (transaction_id,),
        ).fetchall()
    context["investigation_memo"] = memo
    answer = call_ollama(
        GROUNDING_SYSTEM,
        "Answer the analyst's follow-up question using only this grounded evidence. If it is "
        "not answerable from the evidence, say so plainly.\n\nEvidence:\n"
        + json.dumps(context, sort_keys=True)
        + "\n\nQuestion:\n" + payload.question,
        history,
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            "INSERT INTO investigation_chat (transaction_id, role, content) VALUES (%s, 'user', %s), (%s, 'assistant', %s)",
            (transaction_id, payload.question, transaction_id, answer),
        )
        connection.commit()
    return ChatResponse(transaction_id=transaction_id, answer=answer)


@app.get("/transactions/{transaction_id}/chat", response_model=list[ChatMessage])
def investigation_chat_history(transaction_id: int):
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        rows = connection.execute(
            "SELECT role, content FROM investigation_chat WHERE transaction_id = %s ORDER BY message_id",
            (transaction_id,),
        ).fetchall()
    return [ChatMessage(role=row[0], content=row[1]) for row in rows]
