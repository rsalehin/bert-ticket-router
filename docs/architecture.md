# Architecture: BERT-based Customer Service Ticket Routing System

**Version:** 0.1.0
**Status:** Draft, in progress
**Last updated:** 2026-04-25
**Companion to:** [specs.md](specs.md)

---

## 1. Component Diagram
+-------------------------------------------+
|  Frontend (React + Vite + TS + Tailwind)  |  Presentation layer
|  - paste / sample message                 |
|  - render pipeline animation              |
|  - render ticket card                     |
+-------------------------------------------+
| HTTP / JSON
v
+-------------------------------------------+
|  API Layer (FastAPI + Uvicorn)            |  Transport layer
|  - Pydantic request validation            |
|  - Pydantic response serialization        |
|  - error -> HTTP status mapping           |
|  - dependency injection (model, DB)       |
+-------------------------------------------+
| Python calls
v
+-------------------------------------------+
|  Service Layer (src/)                     |  Domain logic
|  - predict.py      (inference pipeline)   |
|  - routing.py      (intent -> ticket)     |
|  - persistence.py  (DB CRUD)              |
|  - data.py         (dataset access)       |
+-------------------------------------------+
|
+-----------+-----------+
v                       v
+----------------+   +-------------------+
| Model          |   | SQLite Database   |
| (PyTorch,      |   | (SQLAlchemy ORM)  |
|  in-memory     |   | tickets table     |
|  singleton)    |   |                   |
+----------------+   +-------------------+
### Layered dependency rule

Each layer depends only on the layer below. The frontend never imports
SQLAlchemy. The API never imports PyTorch directly. The service layer never
knows it is being called over HTTP. This is enforced by import discipline,
not by tooling — violations are caught in code review.

### Component lifecycles

| Component         | Created                      | Destroyed                    |
| ----------------- | ---------------------------- | ---------------------------- |
| FastAPI process   | `uvicorn` start              | process kill                 |
| Model singleton   | FastAPI startup event        | process kill                 |
| Routing config    | FastAPI startup event        | process kill                 |
| DB connection     | per-request (via dep. inj.)  | end of request               |
| SQLite file       | first run (auto-created)     | manual delete                |
| React app         | browser tab open             | tab close / reload           |

The model and routing config are loaded **once** at server startup. Loading
BERT-base on each request would add ~3 seconds of latency — unacceptable.

---

## 2. Data Flow: One `/classify` Request
[1] User submits message in React UI
|
v
[2] POST /classify  body: { "message": "..." }
|
v
[3] FastAPI route handler invoked
|
v
[4] Pydantic validates request body
|   on failure -> 422 Unprocessable Entity
v
[5] predict.classify(message) -> Prediction
| - tokenize (max_len=64)
| - move tensors to device
| - model forward pass
| - softmax
| - argmax + top-k extraction
v
[6] routing.build_ticket(prediction, message) -> Ticket
| - lookup intent in routing config
| - resolve dept, priority, SLA, tags
| - generate ticket id (ULID)
| - timestamp
v
[7] persistence.save_ticket(ticket) -> persisted Ticket
| - SQLAlchemy INSERT
| - commit transaction
| on failure -> 500 Internal Server Error
v
[8] Pydantic serializes ClassifyResponse
| { intent, confidence, top_k, ticket }
v
[9] HTTP 200 + JSON body returned
|
v
[10] React updates state, animates pipeline,
renders ticket card with top-3 alternatives
### Why DB write is synchronous (not fire-and-forget)

If the DB is down, the user must know. A fire-and-forget background task
would silently lose tickets on a crash. Synchronous write trades a little
latency (~5 ms for SQLite local) for correctness. We can revisit if NFR-3/4
is at risk.

### Validation is at the boundary, not inside

Pydantic validates **on entry** and **on exit**. Inside the service layer,
functions receive typed objects and trust them. This is the "parse, dont
validate" principle: validate once at the edge, then operate on parsed types.

---

## 3. Module Breakdown (`src/`)

Each module has a single responsibility. The dependency graph is acyclic:
modules listed earlier never import from modules listed later.

### `src/schemas.py`
**Owns:** Pydantic models for all internal and API-facing types.
**Why separate:** shared by predict, routing, persistence, API, and tests. Living in one place avoids cross-module coupling.
**Exports:** `ClassifyRequest`, `ClassifyResponse`, `Prediction`, `IntentScore`, `Ticket`, `RoutingRule`.
**Imports:** `pydantic`, stdlib only.

### `src/config.py`
**Owns:** application settings (model name, max_len, paths, DB URL, top-k).
**Why separate:** values used by training, inference, and serving. Centralized to make changes safe.
**Exports:** `Settings` (Pydantic BaseSettings), `get_settings()`.
**Imports:** `pydantic_settings`, stdlib.

### `src/data.py`
**Owns:** dataset loading, train/val/test split, tokenization, DataLoader construction.
**Why separate:** dataset concerns change independently from model concerns.
**Exports:** `load_banking77()`, `make_splits()`, `make_dataloaders()`, `tokenize_batch()`.
**Imports:** `datasets`, `transformers`, `torch`, `config`, `schemas`.

### `src/model.py`
**Owns:** the BERT classifier definition (BERT-base + dropout + linear head). No training, no inference logic — just the `nn.Module`.
**Why separate:** the model architecture is reusable across training, evaluation, and serving.
**Exports:** `BertClassifier`, `load_pretrained_classifier(path)`.
**Imports:** `torch`, `transformers`, `config`, `schemas`.

### `src/train.py`
**Owns:** custom PyTorch training loop (forward, loss, backward, optimizer step, scheduler, checkpointing).
**Why separate:** training is a one-shot offline process. The serving stack does not import this.
**Exports:** `train(config_path)` as the entry point.
**Imports:** `torch`, `data`, `model`, `config`, `schemas`.

### `src/evaluate.py`
**Owns:** metrics computation (accuracy, macro-F1, per-class P/R/F1, confusion matrix, top-k accuracy) and report generation.
**Why separate:** evaluation logic is reused by training (per-epoch eval) and by post-hoc analysis.
**Exports:** `evaluate(model, dataloader)`, `compute_metrics(y_true, y_pred, y_probs)`, `save_report(metrics, path)`.
**Imports:** `torch`, `sklearn`, `numpy`, `model`, `schemas`.

### `src/routing.py`
**Owns:** mapping from intent label to a structured `Ticket` (department, priority, SLA, tags), driven by `configs/routing.yaml`.
**Why separate:** routing rules are business logic, not ML logic. Will change without retraining.
**Exports:** `load_routing_rules(path)`, `build_ticket(prediction, message) -> Ticket`.
**Imports:** `pyyaml`, `ulid`, `config`, `schemas`.

### `src/predict.py`
**Owns:** the inference pipeline. Loads the trained model once, performs tokenize -> forward -> softmax -> top-k for a single message.
**Why separate:** the FastAPI layer needs a single, stable entry point for inference. Hides PyTorch from the API.
**Exports:** `Classifier` (class with `__init__(checkpoint_path)` and `classify(message) -> Prediction`).
**Imports:** `torch`, `transformers`, `model`, `config`, `schemas`.

### `src/persistence.py`
**Owns:** SQLAlchemy ORM models, session management, ticket CRUD.
**Why separate:** isolates DB concerns. The rest of the codebase only sees `Ticket` (Pydantic), never `TicketRow` (SQLAlchemy).
**Exports:** `TicketRow` (ORM), `init_db()`, `get_session()`, `save_ticket(session, ticket)`, `list_tickets(session, limit)`.
**Imports:** `sqlalchemy`, `config`, `schemas`.

### `src/__init__.py`
Empty file marking `src/` as a Python package. Selective re-exports may be added later for ergonomics.

### Dependency graph
schemas.py     <-  (no deps)
config.py      <-  (no deps)
data.py        <-  config, schemas
model.py       <-  config, schemas
routing.py     <-  config, schemas
persistence.py <-  config, schemas
predict.py     <-  model, schemas, config
train.py       <-  data, model, config, schemas
evaluate.py    <-  data, model, schemas
**Invariant:** no cycles. Lower-listed modules can import upper-listed ones, never the reverse. Violations are blocked at code review.

### What lives outside `src/`

- `api/server.py` imports from `src.predict`, `src.routing`, `src.persistence`, `src.schemas`. The API layer composes services; it owns no domain logic.
- `tests/` mirrors `src/` structure: `tests/test_data.py`, `tests/test_model.py`, etc.
- `notebooks/` may import from `src/` for exploratory work but contains no logic that the rest of the system depends on.

---

## 4. Type Contracts

### Single source of truth

Types are defined **once** in Python via Pydantic models in `src/schemas.py`.
FastAPI auto-generates an OpenAPI JSON schema. We use `openapi-typescript`
(npm) to generate `frontend/src/types/api.ts` from the OpenAPI schema.
The frontend then imports these types directly.
[src/schemas.py (Pydantic)]
|
v
[FastAPI /openapi.json]
|
v  npx openapi-typescript ...
[frontend/src/types/api.ts]
|
v
[React components]
**Rule:** never hand-write types in TypeScript that mirror Python types.
Always regenerate after backend changes. The generated file is committed
(so the frontend builds without running the backend) but never edited by hand.

### API types

#### `ClassifyRequest`
| Field   | Type | Constraints                 |
| ------- | ---- | --------------------------- |
| message | str  | min_length=1, max_length=1000 |

#### `IntentScore`
| Field      | Type  | Constraints     |
| ---------- | ----- | --------------- |
| intent     | str   | non-empty       |
| confidence | float | 0.0 <= x <= 1.0 |

#### `Ticket`
| Field      | Type                       | Constraints                 |
| ---------- | -------------------------- | --------------------------- |
| id         | str                        | ULID, 26 chars              |
| department | str                        | non-empty                   |
| priority   | Literal["P1", "P2", "P3"]  | closed enum                 |
| sla_hours  | int                        | > 0                         |
| tags       | list[str]                  | 1-10 items, each non-empty  |
| created_at | datetime                   | UTC, ISO 8601 on the wire   |

#### `ClassifyResponse`
| Field      | Type             | Constraints                     |
| ---------- | ---------------- | ------------------------------- |
| intent     | str              | non-empty (top-1 prediction)    |
| confidence | float            | 0.0 <= x <= 1.0                 |
| top_k      | list[IntentScore]| length == 3                     |
| ticket     | Ticket           | -                               |

### Internal domain types

Used by service-layer functions, not exposed at the HTTP boundary.

#### `Prediction`
| Field   | Type             | Notes                      |
| ------- | ---------------- | -------------------------- |
| intent  | str              | top-1 label                |
| confidence | float         | top-1 probability          |
| top_k   | list[IntentScore]| length == 3                |

`predict.classify(message)` returns this. `routing.build_ticket(prediction, message)` consumes it and returns a `Ticket`.

#### `RoutingRule`
| Field      | Type                       |
| ---------- | -------------------------- |
| intent     | str                        |
| department | str                        |
| priority   | Literal["P1", "P2", "P3"]  |
| sla_hours  | int                        |
| tags       | list[str]                  |

Loaded from `configs/routing.yaml` at startup. 77 rules, one per intent.

### Identifier strategy: ULID

Tickets use ULID (Universally Unique Lexicographically Sortable Identifier).

- **Format:** 26-char Crockford base32 string, e.g. `01HXR3F8Z4G2N6P7Q9S0T2U4V6`
- **Property:** lexicographic sort = creation-time sort. Globally unique. Opaque (does not leak counts).
- **Why not int autoincrement:** leaks ticket counts; predictable.
- **Why not uuid4:** globally unique but not sortable, harder to debug.
- **Library:** `python-ulid` on the backend.

### Datetime convention

- **Storage:** UTC `datetime` (no timezone-naive datetimes anywhere).
- **Wire format:** ISO 8601 with `Z` suffix, e.g. `2026-04-25T12:34:56Z`.
- **Parsing on frontend:** `new Date(iso)`.

### Validation philosophy: parse, dont validate

Pydantic validates **at the boundary** (request entry, response exit).
Inside the service layer, functions accept already-parsed typed objects
and trust them. We do not re-validate the same data twice.

If a request fails validation, FastAPI returns **422 Unprocessable Entity**
with a structured error body listing offending fields. We do not customize
this — the default is sufficient.

---

## 5. Database Schema

### Engine and migration tool

- **Engine:** SQLite (file-based, zero-setup). DB file at `artifacts/tickets.db`.
- **ORM:** SQLAlchemy 2.x (declarative, typed).
- **Migrations:** Alembic, configured from project root.
- **Path to PostgreSQL:** all SQLAlchemy code uses portable types. Migration is a connection-string change plus replacing JSON-in-TEXT columns with native JSONB.

### `tickets` table

| Column         | SQL type   | Null?    | Default       | Notes                              |
| -------------- | ---------- | -------- | ------------- | ---------------------------------- |
| id             | TEXT       | NOT NULL | -             | PRIMARY KEY. 26-char ULID.         |
| message        | TEXT       | NOT NULL | -             | Original customer input.           |
| intent         | TEXT       | NOT NULL | -             | Top-1 predicted intent.            |
| confidence     | REAL       | NOT NULL | -             | Top-1 probability, 0.0..1.0.       |
| top_k_json     | TEXT       | NOT NULL | -             | JSON array of {intent, confidence}.|
| department     | TEXT       | NOT NULL | -             | From routing layer.                |
| priority       | TEXT       | NOT NULL | -             | "P1" / "P2" / "P3".                |
| sla_hours      | INTEGER    | NOT NULL | -             | From routing layer, > 0.           |
| tags_json      | TEXT       | NOT NULL | -             | JSON array of strings.             |
| model_version  | TEXT       | NOT NULL | -             | Audit trail, e.g. "bert-base@v0.1".|
| created_at     | TIMESTAMP  | NOT NULL | now() UTC     | Server-side, never client-supplied.|

### Indexes

| Index name              | Columns        | Purpose                          |
| ----------------------- | -------------- | -------------------------------- |
| (implicit) PK           | id             | Lookup by ID.                    |
| ix_tickets_created_at   | created_at     | "recent tickets" queries.        |
| ix_tickets_department   | department     | Filter by department.            |
| ix_tickets_intent       | intent         | Filter by intent.                |

### Why JSON-in-TEXT for `top_k` and `tags`

SQLite has no native array type. Options considered:

1. **Separate `ticket_tags` and `ticket_top_k` tables** — relationally pure but introduces 3-way joins for a single ticket read. Overkill for v0.1.
2. **Comma-separated string** — fragile (what if a tag contains a comma).
3. **JSON in TEXT** — pragmatic, queryable with `json_extract()`, and migrates cleanly to PostgreSQL `JSONB`.

We use option 3.

### Why `model_version` is stored

When the model is retrained, predictions for the same input may change. Without recording which model produced a ticket, post-hoc analysis is impossible. The cost is one extra TEXT column; the benefit is permanent auditability.

### `model_version` format

`{base-model}@v{semver}-{git-sha}` — e.g. `bert-base-uncased@v0.1.0-a1b2c3d`.

Set at server startup from a manifest file written next to the model checkpoint during training. Never inferred at request time.

### ORM model vs API model

Two distinct types represent a ticket:

| Aspect          | `TicketRow` (SQLAlchemy)              | `Ticket` (Pydantic)                  |
| --------------- | ------------------------------------- | ------------------------------------ |
| Module          | `src/persistence.py`                  | `src/schemas.py`                     |
| Purpose         | Database row mapping                  | API and domain shape                 |
| `top_k`         | `top_k_json: str`                     | `top_k: list[IntentScore]`           |
| `tags`          | `tags_json: str`                      | `tags: list[str]`                    |
| Created by      | `Session.add(...)`                    | `Ticket(...)`                        |

`persistence.save_ticket(session, ticket: Ticket) -> Ticket` performs:

1. Pydantic `Ticket` -> SQLAlchemy `TicketRow` (serializing JSON columns).
2. `session.add(row); session.commit()`.
3. `TicketRow` -> `Ticket` (deserializing JSON columns), returned to caller.

The rest of the codebase imports only `Ticket` from `schemas`. `TicketRow` is an implementation detail of `persistence.py`.

### Connection lifecycle

Per FastAPI dependency injection:

- A new `Session` is created per request via `Depends(get_session)`.
- The session is closed at the end of the request (success or failure).
- No long-lived sessions, no global session, no thread-safety concerns.

The SQLAlchemy `Engine` (connection pool) **is** a singleton — created once at startup, shared across requests.

### Initial migration

Alembic generates `alembic/versions/0001_create_tickets.py`. This becomes the baseline. All future schema changes are new migrations on top.

---

## 6. Configuration Management

### Three categories of config — handled separately

| Category               | Example                                  | Storage                       |
| ---------------------- | ---------------------------------------- | ----------------------------- |
| Environment-dependent  | DB URL, checkpoint path, log level       | Environment variables / `.env`|
| Application defaults   | `max_len=64`, `top_k=3`, `device`        | Code, overridable via env     |
| Domain config          | 77 routing rules                         | `configs/routing.yaml`        |

Mixing these is an anti-pattern. Routing rules are not environment-dependent. Env vars are not where you store 77 records of business data.

### Backend: Pydantic Settings

Implementation in `src/config.py`:

```python
class Settings(BaseSettings):
    # Paths
    model_checkpoint_path: Path = Path("artifacts/model")
    routing_config_path: Path = Path("configs/routing.yaml")
    db_url: str = "sqlite:///artifacts/tickets.db"

    # Model and inference
    base_model_name: str = "bert-base-uncased"
    max_len: int = 64
    top_k: int = 3
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Training (used by train.py only)
    batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42

    # Server
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model_version: str = "bert-base-uncased@v0.1.0-dev"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
    )
```

### Singleton access

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`lru_cache` ensures one validated instance per process. FastAPI injects it via `Depends(get_settings)`. Tests bypass the cache by instantiating `Settings(...)` directly.

### Configuration precedence (highest wins)

1. Explicit kwargs passed to `Settings(...)` (used in tests).
2. Environment variables prefixed `APP_` (e.g. `APP_DB_URL`).
3. Variables in a `.env` file at project root.
4. Field defaults in `Settings` class.

### Files

- `.env` — real values, **gitignored**, lives on each machine.
- `.env.example` — template, **committed**, lists variable names with dummy values. New developers copy this to `.env` and edit.

### Routing config

`configs/routing.yaml` — list of 77 entries, one per intent. Format:

```yaml
- intent: card_arrival
  department: Cards
  priority: P3
  sla_hours: 24
  tags: [card, delivery, shipping]
- intent: lost_or_stolen_card
  department: Security
  priority: P1
  sla_hours: 1
  tags: [card, security, fraud]
```

Loaded once at FastAPI startup by `routing.load_routing_rules(path)`. Held in memory as `dict[str, RoutingRule]`. Hot-reload is not supported in v0.1 — restart the server to apply changes.

A loader assertion verifies all 77 banking77 intents are covered. Missing intent at startup fails loud, not at request time.

### Frontend: Vite environment variables

`frontend/.env` (gitignored) and `frontend/.env.example` (committed). Vite inlines variables prefixed `VITE_` at build time.

| Variable           | Purpose                            | Example                  |
| ------------------ | ---------------------------------- | ------------------------ |
| `VITE_API_BASE_URL`| Backend URL the frontend calls     | `http://localhost:8000`  |

Access in code: `import.meta.env.VITE_API_BASE_URL`. Inlined at `npm run build`, so it must be set before building.

### Secrets

v0.1 has no real secrets (no auth, no third-party APIs, SQLite). The `.env` discipline is set up now so adding secrets later is a matter of adding lines, not rearchitecting.

**Rule:** if a value should never appear in a screenshot or a log, it is a secret. Secrets go in `.env` only; `.env` is `.gitignore`'d.

---

## 7. Error Handling

### Principle: errors propagate, boundaries translate

The service layer raises domain exceptions. The API layer catches them and maps to HTTP status + structured JSON. The frontend reads status + error code and renders user-facing UI.
[service layer]   raises AppError subclass
|
v
[API layer]       FastAPI exception handler -> HTTPResponse
|
v
[frontend]        switch on error.code -> UI message
No silent catches. No leaked stack traces. No raw exceptions reaching the client.

### Error categories

| Kind                    | Source                                      | Client receives                      |
| ----------------------- | ------------------------------------------- | ------------------------------------ |
| Expected (user error)   | Bad input, missing routing rule             | Specific 4xx + descriptive message   |
| Unexpected (bug/infra)  | DB down, model not loaded, library exception| Generic 500 + correlation ID         |

Unexpected errors are logged server-side with full traceback. Clients only see generic messages and an ID for support.

### Custom exception hierarchy

Defined in `src/errors.py`:

```python
class AppError(Exception):
    """Base for all domain errors."""
    code: str = "APP_ERROR"
    http_status: int = 500

class ValidationError(AppError):
    """Application-level validation, beyond Pydantic."""
    code = "VALIDATION_ERROR"
    http_status = 400

class IntentNotInRoutingError(AppError):
    """Predicted intent has no routing rule."""
    code = "INTENT_NOT_IN_ROUTING"
    http_status = 500

class ModelNotLoadedError(AppError):
    """Inference attempted before model is ready."""
    code = "MODEL_NOT_LOADED"
    http_status = 503

class PersistenceError(AppError):
    """Database operation failed."""
    code = "PERSISTENCE_ERROR"
    http_status = 503
```

Each subclass carries its `code` and default `http_status`. The API layer uses these directly.

### HTTP status code mapping

| Status | Meaning                       | When raised                                    |
| ------ | ----------------------------- | ---------------------------------------------- |
| 200    | OK                            | Successful classification                      |
| 400    | Bad Request                   | `ValidationError` (app-level)                  |
| 422    | Unprocessable Entity          | Pydantic validation failure (FastAPI default)  |
| 500    | Internal Server Error         | Unexpected exception, `IntentNotInRoutingError`|
| 503    | Service Unavailable           | Model not loaded, DB unreachable               |

### Error response schema

All error responses (4xx and 5xx) use a single shape:

```json
{
  "error": {
    "code": "INTENT_NOT_IN_ROUTING",
    "message": "No routing rule found for intent 'foo'.",
    "correlation_id": "01HXR3F8Z4G2N6P7Q9S0T2U4V6"
  }
}
```

Defined as a Pydantic model `ErrorResponse` in `src/schemas.py` so it appears in the OpenAPI schema and reaches the frontend types.

### Correlation IDs

Every request is assigned a ULID at the API boundary by middleware:

1. Generated on request entry.
2. Bound to the logger context (every log line in the request includes it).
3. Returned in the `X-Correlation-ID` response header.
4. Included in any error body.

A user reporting "the demo broke at 14:32" gives you the ID; you `grep` logs for it and see the full request lifecycle.

### Logging

- **Library:** `structlog` for structured (JSON) logs.
- **Level:** controlled by `Settings.log_level`.
- **What is logged:**
  - request received (path, method, correlation_id)
  - prediction outcome (intent, confidence, latency)
  - errors (code, message, traceback for unexpected)
  - DB writes (ticket_id, latency)
- **What is not logged:**
  - secrets (none in v0.1, but discipline starts now)
  - full request body in "production-mode" — toggleable via setting

### Pydantic does the input validation

Request body validation is Pydantic's job. We do not write input validation in handlers. Bad input -> automatic 422 with field-level details.

We add application-level validation only where Pydantic cannot help — for example, "the predicted intent must exist in `routing.yaml`." That raises `IntentNotInRoutingError`.

### Frontend error handling

The API client (`frontend/src/lib/api.ts`) wraps `fetch`:
response.ok && status === 200 -> return parsed JSON
status in 4xx                  -> throw ApiError(error.code, error.message)
status in 5xx                  -> throw ApiError("INTERNAL", "...", correlation_id)
network failure                -> throw NetworkError()

UI components catch these and render:

| Error kind        | UI behavior                                                          |
| ----------------- | -------------------------------------------------------------------- |
| 4xx (user error)  | Inline message near the input, e.g. "Message cannot be empty."       |
| 5xx (server bug)  | Toast with "Something went wrong. ID: <correlation_id>".             |
| Network failure   | Banner: "Cannot reach server. Check the backend is running."         |

No errors are silently dropped. Every failure mode produces visible feedback.

---

## 8. Model Lifecycle

### Loading: eager, at server startup

The model is loaded **once** when the FastAPI process starts, **before** the
server accepts traffic. If loading fails, the process exits with a clear
error — never silently degrade to a non-working server.

Implementation uses FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.classifier = Classifier(
        checkpoint_path=settings.model_checkpoint_path,
        device=resolve_device(settings.device),
        max_len=settings.max_len,
    )
    yield
    # cleanup (none needed for in-process model)
```

### Why not lazy load on first request

Lazy load would make the first `/classify` call take ~4 seconds (model load) instead of ~50 ms. Predictable startup is preferred over predictable first-request latency. Eager load also surfaces bad checkpoints immediately rather than mid-demo.

### Storage: in-memory singleton

The `Classifier` instance lives on `app.state.classifier`. All request handlers receive the same instance via FastAPI dependency injection (`Depends(get_classifier)`). No module-level globals (those break tests).

PyTorch inference (forward pass with `eval()` + `inference_mode`) is read-only on module weights — safe to share across concurrent requests.

### `Classifier` class responsibilities

`src/predict.py` defines:

```python
class Classifier:
    def __init__(self, checkpoint_path: Path, device: str, max_len: int): ...

    @torch.inference_mode()
    def classify(self, message: str) -> Prediction: ...

    @property
    def model_version(self) -> str: ...

    @property
    def num_labels(self) -> int: ...
```

It owns: the tokenizer, the model (`.eval()` mode), the device, the ordered label list, the model version string. It does not know about HTTP, routing, or persistence.

### Two PyTorch details that matter

1. **`.eval()`** — disables dropout. Without it, every prediction is stochastic. Easy to forget; we set it once at construction.
2. **`@torch.inference_mode()`** — disables autograd tracking for the wrapped function. Faster than `torch.no_grad()`. Standard for inference paths.

### Model artifact layout

`artifacts/model/` (gitignored) contains:

| File                       | Source                | Purpose                                   |
| -------------------------- | --------------------- | ----------------------------------------- |
| `config.json`              | HF Transformers       | Model configuration                       |
| `model.safetensors`        | training              | Weights (preferred over `pytorch_model.bin`) |
| `tokenizer_config.json`    | HF Transformers       | Tokenizer configuration                   |
| `vocab.txt`                | HF Transformers       | WordPiece vocabulary                      |
| `special_tokens_map.json`  | HF Transformers       | [CLS], [SEP], [PAD], etc.                 |
| `labels.json`              | training (ours)       | Ordered list of 77 intent strings         |
| `manifest.json`            | training (ours)       | `model_version`, training date, metrics   |

### Two files beyond standard HF format

- **`labels.json`** — `["card_arrival", "lost_or_stolen_card", ..., "wrong_amount_of_cash_received"]` in label-id order. Critical: model outputs class index `i`; we resolve to a string via `labels[i]`. Without this file we have integers without meaning.
- **`manifest.json`** — written by training, read at startup:

```json
{
  "model_version": "bert-base-uncased@v0.1.0-a1b2c3d",
  "base_model": "bert-base-uncased",
  "trained_at": "2026-04-25T12:34:56Z",
  "git_sha": "a1b2c3d",
  "metrics": { "macro_f1": 0.93, "top3_acc": 0.98, "top1_acc": 0.92 }
}
```

The `model_version` field is what populates the `tickets.model_version` audit column.

### Inference flow (single message)
classify(message)
-> tokenize(message, max_len=64, truncation, padding) -> input_ids, attention_mask
-> move tensors to device
-> model.forward(input_ids, attention_mask) -> logits [1, 77]
-> softmax(logits, dim=-1)                  -> probs  [1, 77]
-> topk(probs, k=3)                          -> top-3 indices and values
-> map indices to labels via labels.json
-> return Prediction(intent, confidence, top_k)

Expected wall time: ~10 ms on A100, ~200-400 ms on a modern laptop CPU.

### No batching in v0.1

Each request runs a single forward pass. Request batching (queue + flush) is a throughput optimization with implementation cost. Not justified for a local demo. Documented as a future improvement.

### Device selection

`Settings.device` is `"cuda" | "cpu" | "auto"`. `"auto"` uses CUDA if `torch.cuda.is_available()`, else CPU. Resolved once at startup; the same device is used for all requests.

### `/health` endpoint

Returns server and model status:

```json
{
  "status": "ok",
  "model_version": "bert-base-uncased@v0.1.0-a1b2c3d",
  "device": "cuda",
  "num_labels": 77,
  "uptime_seconds": 12345
}
```

Used by the frontend on load to detect a missing/down backend, and by future monitoring.

### Reproducibility

- Training: `Settings.seed` (default 42) seeds Python `random`, NumPy, PyTorch CPU and CUDA RNGs. Set once at the top of `train.py`.
- Inference: deterministic by construction (no dropout, no sampling). No seed needed.

---

## 9. Project Tooling

### Python project file: `pyproject.toml`

Single source of truth for Python packaging, dependencies, and tool config (ruff, pytest, mypy). Replaces `setup.py`, `setup.cfg`, and per-tool config files.

### Dependency management

- **Runtime deps:** declared in `pyproject.toml` under `[project.dependencies]`.
- **Dev deps:** `[project.optional-dependencies].dev` (pytest, ruff, mypy, pre-commit).
- **Pinning:** exact versions for reproducibility (matches what `requirements.txt` already has).
- **Install:** `pip install -e ".[dev]"` for an editable install with dev tools.
- **Virtual environment:** `.venv/` at project root, gitignored.

### Linting and formatting: `ruff`

One tool replaces `flake8`, `isort`, `black`, `pyupgrade`, etc. Configured in `pyproject.toml`.

- **Lint:** `ruff check .`
- **Format:** `ruff format .`
- **Auto-fix:** `ruff check --fix .`

Rules: pycodestyle, pyflakes, isort, bugbear, pep8-naming, simplify, comprehensions. Line length 100.

### Type checking: `mypy`

Strict mode for `src/` and `api/`. Untyped third-party libs ignored via overrides. Run as `mypy src api`.

### Testing: `pytest`

- Test discovery: `tests/` mirrors `src/` structure.
- Naming: `test_<module>.py`, functions `test_<behavior>`.
- Fixtures in `tests/conftest.py`.
- Coverage via `pytest-cov`; target 80% on `src/`.
- Slow tests (model loading, full training) marked with `@pytest.mark.slow` and skipped by default.

### Pre-commit hooks: `pre-commit`

Runs on every `git commit`. Configured in `.pre-commit-config.yaml`:

- `ruff check --fix`
- `ruff format`
- `mypy` (on staged files only, fast mode)
- trailing-whitespace, end-of-file-fixer, check-yaml, check-toml

Bad commits are blocked at commit time, not in CI.

### Frontend tooling

- **Build:** Vite (fast HMR, native ESM).
- **Lint:** ESLint with `@typescript-eslint` and `eslint-plugin-react`.
- **Format:** Prettier.
- **Type checking:** `tsc --noEmit` (TypeScript compiler in check mode).
- **Test:** Vitest (Vite-native, Jest-compatible API).

Configured in `frontend/package.json` and `frontend/eslint.config.js`.

### Type generation: backend -> frontend

After any change to Pydantic schemas:

```bash
# Backend serves OpenAPI schema; frontend regenerates TS types
npx openapi-typescript http://localhost:8000/openapi.json -o frontend/src/types/api.ts
```

Generated file is committed. Never hand-edited. Out-of-date types are caught by `tsc` before runtime.

### CI: GitHub Actions

`.github/workflows/ci.yml` runs on push and PR:

- **Backend job:** install deps, `ruff check`, `mypy`, `pytest -m "not slow"`.
- **Frontend job:** `npm ci`, `npm run lint`, `npm run typecheck`, `npm run test`.

No deployment. No model training in CI (too expensive, not deterministic enough).

### Editor config

`.editorconfig` at repo root: UTF-8, LF line endings, 2-space indent for YAML/JSON, 4-space for Python, trim trailing whitespace.

### Documentation tooling

- Specs and architecture in plain Markdown under `docs/`.
- API docs auto-generated by FastAPI at `/docs` (Swagger UI) and `/redoc`.
- No separate docs site for v0.1.

### Tool summary

| Concern              | Tool                  | Config location                |
| -------------------- | --------------------- | ------------------------------ |
| Python packaging     | `pyproject.toml`      | repo root                      |
| Python lint+format   | ruff                  | `pyproject.toml`               |
| Python type check    | mypy                  | `pyproject.toml`               |
| Python testing       | pytest, pytest-cov    | `pyproject.toml`               |
| Pre-commit           | pre-commit            | `.pre-commit-config.yaml`      |
| Frontend build       | Vite                  | `frontend/vite.config.ts`      |
| Frontend lint        | ESLint                | `frontend/eslint.config.js`    |
| Frontend format      | Prettier              | `frontend/.prettierrc`         |
| Frontend type check  | tsc                   | `frontend/tsconfig.json`       |
| Frontend testing     | Vitest                | `frontend/vite.config.ts`      |
| Type generation      | openapi-typescript    | npm script                     |
| DB migrations        | Alembic               | `alembic.ini`, `alembic/`      |
| CI                   | GitHub Actions        | `.github/workflows/ci.yml`     |

---
