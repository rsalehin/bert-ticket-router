# Specifications: BERT-based Customer Service Ticket Routing System

**Version:** 0.1.0
**Status:** Draft, pending review
**Last updated:** 2026-04-25

---

## 1. Problem Statement

Customer service teams receive unstructured natural-language messages that
must be triaged into the right department with appropriate priority and SLA.
Manual triage is slow, inconsistent, and does not scale. We will build an
automated intent-classification system that converts a customer message into
a structured ticket with an intent label, routing department, priority, SLA,
and tags.

## 2. Goals and Non-Goals

### Goals
- Fine-tune BERT-base-uncased on the PolyAI/banking77 dataset (77 intents).
- Achieve macro-F1 >= 0.90 and top-3 accuracy >= 0.97 on the held-out test set.
- Expose inference via a FastAPI HTTP endpoint with a typed JSON contract.
- Persist generated tickets in a SQLite database via SQLAlchemy ORM.
- Provide a React + Vite + Tailwind frontend that visualizes the pipeline
  (message -> tokens -> model -> intent -> ticket).
- Run fully locally on the developer machine; train on Colab Pro A100.

### Non-Goals (explicitly out of scope)
- Multi-lingual support (English-only via banking77).
- Multi-label classification (banking77 is single-label).
- Online learning or continuous retraining.
- Authentication, multi-tenant access control, or user accounts.
- Cloud deployment, autoscaling, or production-grade observability.
- Domain transfer beyond banking.

## 3. Stakeholders and Users

- **Primary user (demo persona):** a customer-service operations lead who
  wants to see how an ML system would route incoming tickets.
- **Builder:** the developer (you), also the maintainer.
- **Reviewer (loop persona):** the AI pair-programmer guiding TDD and review.

## 4. User Stories

- US-1: As an ops lead, I want to paste a customer message and see the
  predicted intent with a confidence score, so that I can judge model quality
  at a glance.
- US-2: As an ops lead, I want to sample a random message from the test set
  and see prediction vs ground truth, so that I can spot-check accuracy.
- US-3: As an ops lead, I want each prediction to produce a full ticket
  (department, priority, SLA, tags), so that I see real routing behavior.
- US-4: As a developer, I want a /classify HTTP endpoint that returns top-k
  intents with confidences, so that any frontend can integrate with it.
- US-5: As a developer, I want all generated tickets persisted in SQL, so
  that I can audit, query, or replay them later.

## 5. Functional Requirements

| ID    | Requirement                                                                                                                                          | Verification              |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| FR-1  | System loads PolyAI/banking77 from HuggingFace Datasets and reports split sizes.                                                                     | Unit test on data loader  |
| FR-2  | Tokenizer converts message to BERT WordPiece IDs, padded/truncated to max_len=64.                                                                    | Unit test on tokenizer    |
| FR-3  | Model = BERT-base-uncased + dropout(0.1) + Linear(768 -> 77), trained with custom PyTorch loop.                                                      | Unit test on forward pass |
| FR-4  | Training uses AdamW (lr=2e-5, wd=0.01), linear warmup (10 percent), 4 epochs, batch=64, bf16 on A100.                                                | Training log inspection   |
| FR-5  | Evaluation reports accuracy, macro-F1, per-class P/R/F1, confusion matrix, top-1/3/5 accuracy.                                                       | Eval script output        |
| FR-6  | Routing layer maps each of 77 intents to {department, priority, SLA, tags} via configs/routing.yaml.                                                 | Unit test on routing      |
| FR-7  | FastAPI POST /classify accepts {message: str} and returns {intent, confidence, top_k, ticket}.                                                       | API integration test      |
| FR-8  | All /classify calls persist a ticket row to SQLite (id, message, intent, confidence, department, priority, sla, tags, created_at).                   | DB integration test       |
| FR-9  | Frontend lets the user paste a message or sample a random test-set message, then visualizes the pipeline and renders the ticket.                     | Manual demo               |
| FR-10 | Frontend shows top-3 alternatives with confidence bars and a correct/incorrect indicator vs ground truth (when available).                           | Manual demo               |

## 6. Non-Functional Requirements

| ID     | Requirement                                                                | Target                            |
| ------ | -------------------------------------------------------------------------- | --------------------------------- |
| NFR-1  | Model accuracy: macro-F1 on banking77 test set                             | >= 0.90                           |
| NFR-2  | Top-3 accuracy on banking77 test set                                       | >= 0.97                           |
| NFR-3  | Inference latency, single message, A100 GPU                                | p95 < 50 ms                       |
| NFR-4  | Inference latency, single message, modern laptop CPU                       | p95 < 500 ms                      |
| NFR-5  | Training wall-clock on Colab Pro A100                                      | < 15 minutes                      |
| NFR-6  | Codebase: type hints, ruff-clean, pytest passing                           | 100 percent of merged code        |
| NFR-7  | Reproducibility: fixed seed, pinned deps, deterministic splits             | Same metrics +/- 0.5 across runs  |
| NFR-8  | API contract: documented via FastAPI OpenAPI schema                        | Auto-generated /docs              |

## 7. Constraints

- **Stack (hard):** Python 3.12+, PyTorch, HuggingFace Transformers, FastAPI,
  SQLite (dev) via SQLAlchemy ORM, React + Vite + TypeScript + Tailwind.
- **Training environment:** Google Colab Pro, A100 GPU.
- **Serving environment:** local machine only, no cloud.
- **Dataset:** PolyAI/banking77, no augmentation in v0.1.
- **Backbone model:** bert-base-uncased (110M params), no distillation or
  quantization in v0.1.
- **License:** dataset CC-BY-4.0, model Apache-2.0; project license MIT.

## 8. Data Specification

- **Source:** datasets.load_dataset("PolyAI/banking77")
- **Splits:** train (10003), test (3080). We carve a val split from train
  (10 percent, stratified by label).
- **Schema:** { "text": str, "label": int in [0, 76] }
- **Label space:** 77 banking intents (e.g. card_arrival,
  transfer_not_received_by_recipient, lost_or_stolen_card).
- **Preprocessing:** lowercase (handled by tokenizer), strip whitespace,
  truncate to 64 tokens.

## 9. API Contract (preview, finalized in Architecture phase)

POST /classify

Request:
{ "message": "My card hasnt arrived after two weeks" }
Response:
{
"intent": "card_arrival",
"confidence": 0.94,
"top_k": [
{"intent": "card_arrival", "confidence": 0.94},
{"intent": "card_delivery_estimate", "confidence": 0.04},
{"intent": "lost_or_stolen_card", "confidence": 0.01}
],
"ticket": {
"id": "tkt_01HXXXX",
"department": "Cards",
"priority": "P3",
"sla_hours": 24,
"tags": ["card", "delivery", "shipping"],
"created_at": "2026-04-25T12:34:56Z"
}
}
## 10. Success Criteria (Definition of Done for v0.1)

- [ ] Model checkpoint achieves NFR-1 and NFR-2; metrics committed to artifacts/reports/.
- [ ] FastAPI service runs locally, /classify works end-to-end, /docs is browsable.
- [ ] SQLite database accumulates ticket rows on each call, verified by query.
- [ ] React frontend runs via npm run dev, can sample random messages and display tickets.
- [ ] All unit and integration tests pass via pytest.
- [ ] README.md documents setup, training, serving, demo.

## 11. Risks and Mitigations

| Risk                                                | Likelihood | Impact | Mitigation                                                    |
| --------------------------------------------------- | ---------- | ------ | ------------------------------------------------------------- |
| Banking77 class imbalance hurts macro-F1            | Medium     | Medium | Stratified val split; class-weighted loss as fallback.        |
| Colab session disconnects mid-training              | Medium     | Low    | Checkpoint per epoch; download weights or push to HF Hub.     |
| Frontend/backend contract drift                     | Medium     | Medium | Single source of truth: Pydantic schemas + auto-gen TS types. |
| Latency target missed on CPU                        | Low        | Low    | Acceptable for v0.1; document as limitation.                  |
| Routing rules feel arbitrary without domain expert  | High       | Low    | Document assumptions in routing.yaml; mark as v0.1 heuristic. |

## 12. Glossary

- **Intent:** the customer underlying purpose, drawn from a fixed label set.
- **Macro-F1:** unweighted mean of per-class F1, robust to class imbalance.
- **Top-k accuracy:** fraction of samples where the true label is in the top-k predictions.
- **SLA:** Service Level Agreement; here, the target response time.
- **[CLS] token:** special BERT token whose final hidden state is used as the sentence embedding.

---

**Approval:** awaiting human-in-the-loop sign-off before proceeding to System Architecture.
