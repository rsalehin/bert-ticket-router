# Implementation Tasks

**Version:** 0.1.0
**Status:** Draft, in progress
**Companions:** [specs.md](specs.md), [architecture.md](architecture.md)

Each task is atomic, independently testable, and small enough for one TDD
chat session. Tasks are listed in dependency order. Do not skip ahead;
later tasks assume earlier ones are complete and merged.

**Size legend:** S = ~30 min, M = ~1-2 hr, L = ~half day.

---

## Foundation (T-001 .. T-006)

### T-001: Initialize `pyproject.toml`

- **Size:** M
- **Touches:** `pyproject.toml`, `requirements.txt`
- **Dependencies:** none
- **Description:** Replace `requirements.txt` with `pyproject.toml` as the single source of truth. Declare project metadata, runtime deps (transformers, datasets, torch, scikit-learn, fastapi, uvicorn, pydantic, pydantic-settings, sqlalchemy, alembic, pyyaml, python-ulid, structlog), dev deps (pytest, pytest-cov, ruff, mypy, pre-commit, httpx). Configure ruff (line length 100, rule selection), pytest (testpaths, markers), mypy (strict for `src` and `api`).
- **Acceptance:**
  - `pip install -e ".[dev]"` succeeds in a fresh `.venv`.
  - `ruff check .` runs (no source files yet, so passes vacuously).
  - `pytest` runs and reports zero tests collected (no errors).
  - `mypy src api` runs (no source files yet, passes).
  - `requirements.txt` removed (or kept as a thin `-e .` shim — decide in task).
- **Notes:** Pin exact versions. Streamlit and psycopg2 from the old `requirements.txt` are dropped (not in scope per specs).

### T-002: Set up `.env.example`, `Settings` class, and `src/__init__.py`

- **Size:** M
- **Touches:** `src/config.py`, `src/__init__.py`, `.env.example`
- **Dependencies:** T-001
- **Description:** Implement `Settings` (Pydantic `BaseSettings`) per architecture section 6. All fields, types, and defaults as specified. `get_settings()` cached via `lru_cache`. Create `.env.example` listing every `APP_*` variable with a dummy value. Add empty `src/__init__.py`.
- **Acceptance:**
  - `from src.config import Settings, get_settings` works.
  - Instantiating `Settings()` with no env vars uses architecture-defined defaults.
  - Setting `APP_MAX_LEN=128` env var overrides the default.
  - `.env.example` exists, gitignored `.env` does not exist (real `.env` is per-developer).
  - Test file `tests/test_config.py` exercises defaults and env override.

### T-003: Set up `src/errors.py` (exception hierarchy)

- **Size:** S
- **Touches:** `src/errors.py`, `tests/test_errors.py`
- **Dependencies:** T-001
- **Description:** Implement `AppError`, `ValidationError`, `IntentNotInRoutingError`, `ModelNotLoadedError`, `PersistenceError` per architecture section 7. Each subclass has class-level `code: str` and `http_status: int` attributes.
- **Acceptance:**
  - All five classes defined with correct codes and statuses.
  - `AppError` is the base of all four subclasses.
  - Tests verify class hierarchy and attribute values.

### T-004: Set up `src/schemas.py` (Pydantic models)

- **Size:** M
- **Touches:** `src/schemas.py`, `tests/test_schemas.py`
- **Dependencies:** T-001
- **Description:** Implement `ClassifyRequest`, `IntentScore`, `Ticket`, `ClassifyResponse`, `Prediction`, `RoutingRule`, `ErrorResponse` per architecture section 4. Apply all field constraints (`Literal["P1","P2","P3"]`, length bounds, value ranges). UTC `datetime` with ISO 8601 serialization.
- **Acceptance:**
  - All seven models defined and importable from `src.schemas`.
  - Validation rejects empty messages, out-of-range confidences, invalid priorities, etc.
  - `Ticket.created_at` is timezone-aware UTC; serializes to `Z`-suffixed ISO 8601.
  - Tests cover happy path and at least one rejection per constraint.

### T-005: Configure pre-commit hooks

- **Size:** S
- **Touches:** `.pre-commit-config.yaml`
- **Dependencies:** T-001
- **Description:** Add pre-commit config per architecture section 9: `ruff check --fix`, `ruff format`, `mypy` (fast mode on staged), trailing-whitespace, end-of-file-fixer, check-yaml, check-toml.
- **Acceptance:**
  - `pre-commit install` registers the hook.
  - Committing a file with trailing whitespace triggers auto-fix.
  - Committing a Python file with a ruff violation blocks the commit.

### T-006: Set up logging (`src/logging_config.py`)

- **Size:** S
- **Touches:** `src/logging_config.py`, `tests/test_logging_config.py`
- **Dependencies:** T-002
- **Description:** Configure `structlog` with JSON renderer. `configure_logging(level: str)` function callable from FastAPI startup and from `train.py`. Adds correlation_id binding helper. Log level from `Settings.log_level`.
- **Acceptance:**
  - `configure_logging("INFO")` sets up structlog without errors.
  - Emitted log records are JSON with `level`, `event`, `timestamp` keys.
  - `bind_correlation_id(cid)` adds `correlation_id` to all subsequent log records in the context.

---

## Data (T-007 .. T-010)

### T-007: Implement `src/data.py` — dataset loading

- **Size:** M
- **Touches:** `src/data.py`, `tests/test_data.py`
- **Dependencies:** T-002, T-004
- **Description:** Implement `load_banking77()` returning a `DatasetDict` with `train` and `test` splits from `datasets.load_dataset("PolyAI/banking77")`. Implement `get_label_names()` returning the ordered 77-element list of intent strings (from the dataset's `ClassLabel` feature). Implement `get_num_labels()` returning 77.
- **Acceptance:**
  - `load_banking77()` returns a `DatasetDict` with `train` (10003 rows) and `test` (3080 rows).
  - Each row has keys `text` (str) and `label` (int in 0..76).
  - `get_label_names()` returns 77 strings, includes known examples like `card_arrival` and `lost_or_stolen_card`.
  - Tests use `pytest.mark.slow` (network-dependent) for the live load and a fixture-based fast test for the label-name function.

### T-008: Implement `src/data.py` — train/val split

- **Size:** S
- **Touches:** `src/data.py`, `tests/test_data.py`
- **Dependencies:** T-007
- **Description:** Implement `make_splits(dataset, val_ratio=0.1, seed=42)` returning a dict with `train`, `val`, `test`. Stratify by label so each split has the same class distribution. Deterministic given seed.
- **Acceptance:**
  - `make_splits(..., seed=42)` produces identical splits across runs.
  - Val split is ~10 percent of train (within rounding); test is unchanged.
  - Each label appears in train and val (no class disappears in val).
  - Test verifies determinism and stratification.

### T-009: Implement `src/data.py` — tokenization

- **Size:** S
- **Touches:** `src/data.py`, `tests/test_data.py`
- **Dependencies:** T-007
- **Description:** Implement `tokenize_batch(examples, tokenizer, max_len=64)` returning `{input_ids, attention_mask, label}` tensors. Use the BERT-base-uncased tokenizer (`AutoTokenizer.from_pretrained("bert-base-uncased")`). Padding to `max_length`, truncation enabled.
- **Acceptance:**
  - Input list of strings -> output dict of equal-length tensors with shape `[batch, 64]`.
  - Long strings are truncated to 64 tokens.
  - Short strings padded with `[PAD]` token id.
  - `attention_mask` has 1s for real tokens, 0s for padding.
  - Tests cover empty edge case, long input, batch of mixed lengths.

### T-010: Implement `src/data.py` — DataLoader factory

- **Size:** S
- **Touches:** `src/data.py`, `tests/test_data.py`
- **Dependencies:** T-008, T-009
- **Description:** Implement `make_dataloaders(splits, tokenizer, batch_size=64, max_len=64) -> {train, val, test}` returning a dict of PyTorch `DataLoader`s. Train shuffled, val/test not. Use `DataCollatorWithPadding` from transformers, or roll a simple collate. `num_workers=0` in tests; let production override via setting if needed (skip for v0.1).
- **Acceptance:**
  - Each loader yields batches with correct keys and tensor shapes.
  - Train loader yields different order on different epochs (shuffling).
  - Total samples across all batches equals the split size.
  - Tests use a 50-row subset to keep fast.

---

## Model (T-011 .. T-014)

### T-011: Implement `src/model.py` — `BertClassifier`

- **Size:** M
- **Touches:** `src/model.py`, `tests/test_model.py`
- **Dependencies:** T-002, T-004
- **Description:** Implement `BertClassifier` (`nn.Module`) with BERT-base-uncased backbone, dropout(0.1) on the [CLS] pooled output, `Linear(768, 77)` classification head. `forward(input_ids, attention_mask) -> logits` of shape `[batch, 77]`. Implement `from_pretrained(path)` classmethod and `save_pretrained(path)` instance method that persist the HF backbone + classifier head + `labels.json` + `manifest.json`.
- **Acceptance:**
  - Forward pass on a `[2, 64]` input returns logits of shape `[2, 77]`.
  - `save_pretrained(tmp)` followed by `from_pretrained(tmp)` reproduces logits exactly on the same input (within float tolerance).
  - The saved directory contains `config.json`, `model.safetensors`, tokenizer files, `labels.json`, `manifest.json`.
  - `manifest.json` contains `model_version`, `base_model`, `trained_at`, `git_sha`, `metrics`.
  - Tests use a tiny config (e.g. `prajjwal1/bert-tiny`) to keep fast; mark live-bert tests as `slow`.

### T-012: Implement `src/evaluate.py` — metrics

- **Size:** M
- **Touches:** `src/evaluate.py`, `tests/test_evaluate.py`
- **Dependencies:** T-011
- **Description:** Implement `compute_metrics(y_true, y_pred, y_probs, label_names) -> dict` returning accuracy, macro-F1, per-class P/R/F1, top-1/3/5 accuracy, and a confusion matrix (numpy array). Implement `save_report(metrics, dir)` writing `metrics.json` and `confusion_matrix.png`.
- **Acceptance:**
  - Given hand-crafted small inputs, returned metrics match expected values.
  - `top_k_accuracy(k=1)` equals plain accuracy.
  - `save_report` writes both files; `metrics.json` is valid JSON with all keys.
  - Tests cover correctness on a 4-class toy example.

### T-013: Implement `src/evaluate.py` — evaluation loop

- **Size:** S
- **Touches:** `src/evaluate.py`, `tests/test_evaluate.py`
- **Dependencies:** T-010, T-011, T-012
- **Description:** Implement `evaluate(model, dataloader, device) -> dict` that runs the model in eval mode + `torch.inference_mode()`, collects predictions and probabilities, calls `compute_metrics`. Returns the metrics dict.
- **Acceptance:**
  - Returns a dict with the same keys as `compute_metrics`.
  - Does not modify model parameters (verified by checksum before/after).
  - Test uses a tiny model + 50-row dataloader.

### T-014: Implement `src/train.py` — training loop

- **Size:** L
- **Touches:** `src/train.py`, `tests/test_train.py`, `notebooks/02_train.ipynb`
- **Dependencies:** T-010, T-011, T-013
- **Description:** Implement `train(settings: Settings) -> None`:
  - Seed all RNGs from `settings.seed`.
  - Load data, build dataloaders, build `BertClassifier`.
  - AdamW (lr=2e-5, wd=0.01), linear warmup (10 percent of total steps), cosine or linear decay.
  - bf16 mixed precision when CUDA available.
  - Loop over `num_epochs`. Per step: forward, cross-entropy loss, backward, optimizer step, scheduler step, log loss every N steps.
  - Per epoch: run `evaluate` on val split, log metrics, save checkpoint to `artifacts/checkpoints/epoch_{i}/` if val macro-F1 improved.
  - At end: copy best checkpoint to `artifacts/model/`, run final evaluation on test, write `artifacts/reports/metrics.json` and `confusion_matrix.png`.
  - Create `notebooks/02_train.ipynb` that imports `train` and runs it on Colab — minimal cells (env setup, mount, call `train`, push checkpoint to download).
- **Acceptance:**
  - `train(settings_with_overrides)` completes on a tiny config (1 epoch, 100 train rows) without error.
  - Best checkpoint exists at `artifacts/model/` after run.
  - `artifacts/reports/metrics.json` exists.
  - Notebook runs end-to-end on Colab A100 in under 15 minutes (NFR-5) with full settings.
  - Final test-set macro-F1 >= 0.90 and top-3 accuracy >= 0.97 (NFR-1, NFR-2).
- **Notes:** Two acceptance modes — code-correctness (tiny config, in CI) and target-metrics (full run, manual on Colab). Both must pass.

---

## Routing and Persistence (T-015 .. T-019)

### T-015: Author `configs/routing.yaml` — 77 routing rules

- **Size:** M
- **Touches:** `configs/routing.yaml`, `tests/test_routing_config.py`
- **Dependencies:** T-007 (need the label list)
- **Description:** Hand-author one routing rule per banking77 intent. Format per architecture section 6: `intent`, `department`, `priority` (P1/P2/P3), `sla_hours`, `tags` (2-4 strings). Use departments: Cards, Transfers, Account, Security, Top-ups, General. Priority by urgency: P1 = security/fraud (lost card, compromised, suspicious), P2 = blocked transactions and access issues, P3 = informational queries.
- **Acceptance:**
  - File parses as valid YAML.
  - Exactly 77 entries.
  - Every banking77 intent name (from T-007 `get_label_names`) is present exactly once.
  - Every entry has all five fields with correct types.
  - All `priority` values are in `{P1, P2, P3}`.
  - All `sla_hours` are positive integers.
  - Test verifies these invariants programmatically.

### T-016: Implement `src/routing.py` — config loader

- **Size:** S
- **Touches:** `src/routing.py`, `tests/test_routing.py`
- **Dependencies:** T-004, T-015
- **Description:** Implement `load_routing_rules(path: Path) -> dict[str, RoutingRule]`. Parse YAML, validate each entry into a `RoutingRule` Pydantic model, return as dict keyed by intent. On startup the API will assert all 77 banking77 intents are covered; missing intent raises `IntentNotInRoutingError`.
- **Acceptance:**
  - Loads `configs/routing.yaml` and returns 77 `RoutingRule` instances keyed by intent.
  - Invalid YAML raises a clear error.
  - Missing required field on any entry raises `ValidationError`.
  - Duplicate intent in YAML raises a clear error.
  - Tests cover happy path and each failure mode using temp YAML files.

### T-017: Implement `src/routing.py` — `build_ticket`

- **Size:** S
- **Touches:** `src/routing.py`, `tests/test_routing.py`
- **Dependencies:** T-016
- **Description:** Implement `build_ticket(prediction: Prediction, message: str, rules: dict[str, RoutingRule], model_version: str) -> Ticket`. Lookup rule by `prediction.intent`. Generate ULID for ticket id. Set `created_at = datetime.now(UTC)`. Raise `IntentNotInRoutingError` if intent missing.
- **Acceptance:**
  - Given a known intent, returns a `Ticket` with all fields populated correctly.
  - `id` is a 26-char ULID string.
  - `created_at` is timezone-aware UTC.
  - Unknown intent raises `IntentNotInRoutingError`.
  - Tests cover happy path, missing intent, ULID format, timestamp tz.

### T-018: Implement `src/persistence.py` — ORM and DB init

- **Size:** M
- **Touches:** `src/persistence.py`, `alembic.ini`, `alembic/env.py`, `alembic/versions/0001_create_tickets.py`, `tests/test_persistence.py`
- **Dependencies:** T-001, T-002, T-004
- **Description:** Define `Base = declarative_base()` and `TicketRow` (SQLAlchemy 2.x typed) matching the schema in architecture section 5. Configure Alembic from project root: `alembic init alembic`, point `alembic.ini` at `Settings.db_url`, generate baseline migration `0001_create_tickets`. Implement `init_db()` (run pending migrations programmatically for tests) and `make_engine(url) -> Engine`, `make_session(engine) -> sessionmaker`.
- **Acceptance:**
  - `alembic upgrade head` on a fresh SQLite DB creates the `tickets` table with all columns and indexes per architecture section 5.
  - `TicketRow` instances can be added, committed, queried.
  - `init_db()` is idempotent (safe to call twice).
  - Tests use an in-memory SQLite (`sqlite:///:memory:`) or a temp file.

### T-019: Implement `src/persistence.py` — CRUD

- **Size:** S
- **Touches:** `src/persistence.py`, `tests/test_persistence.py`
- **Dependencies:** T-018
- **Description:** Implement `save_ticket(session, ticket: Ticket) -> Ticket` (Pydantic -> ORM with JSON serialization for `top_k` and `tags`, insert, commit, return Pydantic). Implement `list_tickets(session, limit: int = 50) -> list[Ticket]` ordered by `created_at DESC`. Implement `get_ticket(session, ticket_id: str) -> Ticket | None`.
- **Acceptance:**
  - `save_ticket` round-trips: inserted Pydantic ticket equals loaded Pydantic ticket on `get_ticket`.
  - `top_k` and `tags` serialize to/from JSON correctly.
  - `list_tickets(limit=10)` returns at most 10, newest first.
  - `get_ticket("nonexistent")` returns `None`.
  - DB error during commit raises `PersistenceError`.

---

## API (T-020 .. T-024)

### T-020: Implement `src/predict.py` — `Classifier` class

- **Size:** M
- **Touches:** `src/predict.py`, `tests/test_predict.py`
- **Dependencies:** T-009, T-011
- **Description:** Implement `Classifier` per architecture section 8. Constructor loads tokenizer, model (`.eval()`), `labels.json`, `manifest.json`. `classify(message: str) -> Prediction` runs the inference flow (tokenize -> forward -> softmax -> top-3) decorated with `@torch.inference_mode()`. Properties: `model_version`, `num_labels`, `device`. Raises `ModelNotLoadedError` if checkpoint dir is missing required files.
- **Acceptance:**
  - Constructed from a saved checkpoint dir, returns sensible `Prediction` for a sample message.
  - `Prediction.top_k` has length 3, all confidences in [0, 1], sum of all 77 probabilities ~= 1.0.
  - `model_version` matches manifest content.
  - Missing `labels.json` or `manifest.json` raises `ModelNotLoadedError`.
  - Tests use the same tiny checkpoint as T-011.

### T-021: Implement `api/server.py` — FastAPI app + lifespan

- **Size:** M
- **Touches:** `api/__init__.py`, `api/server.py`, `tests/test_server_lifespan.py`
- **Dependencies:** T-002, T-006, T-016, T-018, T-020
- **Description:** Create FastAPI app. Implement `lifespan` context manager that on startup:
  - Calls `configure_logging(settings.log_level)`.
  - Loads routing rules (`load_routing_rules`).
  - Constructs `Classifier` and asserts all 77 labels are present in routing rules.
  - Initializes DB engine + sessionmaker, runs migrations.
  - Stores all on `app.state` (`classifier`, `routing_rules`, `engine`, `session_factory`, `started_at`).
  Implement dependency providers: `get_classifier`, `get_routing_rules`, `get_session`, `get_settings`.
- **Acceptance:**
  - `from api.server import app` imports without error.
  - With a valid checkpoint and routing config, `TestClient(app)` startup completes successfully.
  - Missing checkpoint causes startup to fail with a clear `ModelNotLoadedError`.
  - Routing config missing one of the 77 intents causes startup to fail with `IntentNotInRoutingError`.

### T-022: Implement `api/server.py` — middleware and exception handlers

- **Size:** M
- **Touches:** `api/server.py`, `api/middleware.py`, `tests/test_api_errors.py`
- **Dependencies:** T-003, T-021
- **Description:** Add middleware that generates a ULID correlation ID per request, binds it to structlog context, and sets `X-Correlation-ID` response header. Add CORS middleware allowing `http://localhost:5173` (Vite dev server). Register exception handlers:
  - `AppError` -> use subclass `code` and `http_status`, render `ErrorResponse`.
  - `RequestValidationError` (Pydantic) -> 422 with structured field errors mapped into `ErrorResponse`.
  - Generic `Exception` -> log full traceback, return generic 500 `ErrorResponse` with `code="INTERNAL_ERROR"`.
- **Acceptance:**
  - Every response has an `X-Correlation-ID` header.
  - Raising `AppError` subclass in a test handler returns the right status and JSON body.
  - Pydantic validation failure returns 422 with `ErrorResponse` shape.
  - Uncaught `RuntimeError` in a test handler returns 500 with `code="INTERNAL_ERROR"` and a correlation_id.
  - Tests cover all four cases.

### T-023: Implement `api/server.py` — `/health` endpoint

- **Size:** S
- **Touches:** `api/server.py`, `tests/test_health.py`
- **Dependencies:** T-021
- **Description:** Implement `GET /health` returning `{ status, model_version, device, num_labels, uptime_seconds }` per architecture section 8. `uptime_seconds = (now - app.state.started_at).total_seconds()`.
- **Acceptance:**
  - 200 response with all five fields.
  - `num_labels` equals 77 (or whatever `Classifier.num_labels` reports).
  - `uptime_seconds` is positive and increases across calls.

### T-024: Implement `api/server.py` — `POST /classify` endpoint

- **Size:** M
- **Touches:** `api/server.py`, `tests/test_classify.py`
- **Dependencies:** T-017, T-019, T-020, T-021, T-022
- **Description:** Implement `POST /classify` accepting `ClassifyRequest`, returning `ClassifyResponse`. Flow:
  1. Validate request (Pydantic — automatic).
  2. `prediction = classifier.classify(req.message)`
  3. `ticket = build_ticket(prediction, req.message, rules, classifier.model_version)`
  4. `saved = save_ticket(session, ticket)`
  5. Return `ClassifyResponse(intent, confidence, top_k, ticket=saved)`.
  Log structured event with intent, confidence, latency_ms.
- **Acceptance:**
  - 200 response with the full `ClassifyResponse` shape.
  - A row is persisted in the DB (verifiable by `list_tickets`).
  - Empty message -> 422.
  - Message > 1000 chars -> 422.
  - The persisted ticket's `model_version` matches the classifier's.
  - Latency on CPU < 500 ms p95 (NFR-4) — measured in the test with 20 calls.
  - Integration test uses `TestClient` end-to-end.

---

## Frontend (T-025 .. T-030)

### T-025: Scaffold Vite + React + TypeScript + Tailwind

- **Size:** M
- **Touches:** `frontend/`, `frontend/.env.example`, `frontend/package.json`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/eslint.config.js`, `frontend/.prettierrc`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/index.css`
- **Dependencies:** none (parallel to backend)
- **Description:** Run `npm create vite@latest frontend -- --template react-ts`, then add Tailwind (`tailwindcss`, `postcss`, `autoprefixer`), ESLint with `@typescript-eslint` and `eslint-plugin-react`, Prettier, Vitest. Configure Tailwind: content globs, JIT, base styles in `index.css`. Create `frontend/.env.example` with `VITE_API_BASE_URL=http://localhost:8000`. Replace default `App.tsx` with a "Hello, ticket router" placeholder.
- **Acceptance:**
  - `npm install` and `npm run dev` start the Vite dev server at `http://localhost:5173`.
  - The placeholder page renders with a Tailwind utility class applied (e.g. `bg-slate-100`).
  - `npm run lint`, `npm run typecheck`, `npm run test` all pass.
  - `npm run build` produces `frontend/dist/` without errors.

### T-026: Generate API types from backend OpenAPI

- **Size:** S
- **Touches:** `frontend/src/types/api.ts`, `frontend/package.json`
- **Dependencies:** T-024, T-025
- **Description:** Add `openapi-typescript` to dev deps. Add npm script `gen:types` that fetches `http://localhost:8000/openapi.json` and writes `frontend/src/types/api.ts`. Run it once to commit a baseline.
- **Acceptance:**
  - `npm run gen:types` regenerates the file from a running backend.
  - The committed `frontend/src/types/api.ts` exports types matching `ClassifyRequest`, `ClassifyResponse`, `IntentScore`, `Ticket`, `ErrorResponse`.
  - `tsc --noEmit` passes.

### T-027: Implement `frontend/src/lib/api.ts` — typed API client

- **Size:** M
- **Touches:** `frontend/src/lib/api.ts`, `frontend/src/lib/api.test.ts`
- **Dependencies:** T-026
- **Description:** Implement `classify(message: string): Promise<ClassifyResponse>` and `health(): Promise<HealthResponse>`. Wrap `fetch`. Read `VITE_API_BASE_URL`. Handle responses per architecture section 7:
  - 200 -> parse JSON, return.
  - 4xx -> throw `ApiError(code, message, correlationId?)`.
  - 5xx -> throw `ApiError("INTERNAL", message, correlationId)`.
  - Network failure -> throw `NetworkError`.
- **Acceptance:**
  - Vitest unit tests with `fetch` mocked cover all four cases.
  - Errors carry `code` and optional `correlation_id` fields.

### T-028: Implement `frontend/src/components/MessageInput.tsx`

- **Size:** S
- **Touches:** `frontend/src/components/MessageInput.tsx`, `frontend/src/components/MessageInput.test.tsx`
- **Dependencies:** T-025
- **Description:** Controlled `<textarea>` with submit button. Props: `onSubmit(message: string)`, `disabled: boolean`. Empty input disables submit. Max 1000 chars enforced client-side with a counter. Tailwind for styling.
- **Acceptance:**
  - Typing updates the character counter.
  - Submit fires `onSubmit` with the trimmed message.
  - Empty / whitespace-only input does not fire submit.
  - Vitest + React Testing Library cover typing, submit, disabled state.

### T-029: Implement `frontend/src/components/TicketCard.tsx`

- **Size:** M
- **Touches:** `frontend/src/components/TicketCard.tsx`, `frontend/src/components/TicketCard.test.tsx`
- **Dependencies:** T-026
- **Description:** Render a `ClassifyResponse`:
  - Top-1 intent in large type with confidence percentage.
  - Top-3 alternatives as a list with confidence bars (Tailwind `bg-blue-500` width = `confidence * 100%`).
  - Ticket fields: id (monospace), department, priority badge (color: P1=red, P2=amber, P3=slate), `sla_hours`, tags as pills, `created_at` formatted as local time.
- **Acceptance:**
  - Given a fixture `ClassifyResponse`, all fields render correctly.
  - Priority badge color depends on value.
  - Confidence bars width is proportional to confidence.
  - Tests cover P1/P2/P3 colorings and one full snapshot.

### T-030: Wire up `App.tsx` — main flow

- **Size:** M
- **Touches:** `frontend/src/App.tsx`, `frontend/src/App.test.tsx`, `frontend/src/components/ErrorBanner.tsx`
- **Dependencies:** T-027, T-028, T-029
- **Description:** App layout: header, `MessageInput`, conditional `TicketCard` for last result, `ErrorBanner` for errors. State: `loading`, `result | null`, `error | null`. On submit: clear error, set loading, call `api.classify`, set result on success, set error on failure. Show correlation ID for 5xx errors. Show network-error banner if backend is unreachable. On mount, call `health()` and show a "backend offline" banner if it fails.
- **Acceptance:**
  - End-to-end happy path test (mocked fetch): submit message -> see ticket card.
  - Error path test: API throws `ApiError` -> see error banner.
  - Network error path test: `health()` rejects on mount -> see "backend offline".
  - Manual test: with backend running, full flow works in the browser.

---

## Integration and Polish (T-031 .. T-034)

### T-031: End-to-end integration test

- **Size:** M
- **Touches:** `tests/test_e2e.py`, `tests/conftest.py`
- **Dependencies:** T-024
- **Description:** Pytest test that boots the FastAPI app via `TestClient` against a real (small) checkpoint and a real SQLite temp file. Sends 5 representative banking77 messages through `POST /classify`. Asserts:
  - All return 200.
  - Each response satisfies the `ClassifyResponse` schema.
  - DB contains 5 rows after the run.
  - The `model_version` on each row matches `/health`.
  - Mean per-request latency on CPU < 500 ms.
- **Acceptance:**
  - Test passes with a checkpoint produced by T-014's tiny-config run.
  - Marked `slow` so CI can opt out; runs locally on demand.

### T-032: GitHub Actions CI workflow

- **Size:** M
- **Touches:** `.github/workflows/ci.yml`
- **Dependencies:** T-001, T-025
- **Description:** Two-job workflow on push and pull_request:
  - **backend:** Ubuntu, Python 3.11, cache pip, `pip install -e ".[dev]"`, `ruff check`, `mypy src api`, `pytest -m "not slow"`.
  - **frontend:** Ubuntu, Node 20, cache npm, `cd frontend && npm ci`, `npm run lint`, `npm run typecheck`, `npm run test`, `npm run build`.
  No model training, no slow tests, no deployment.
- **Acceptance:**
  - Push to a feature branch triggers both jobs.
  - Both jobs green on a clean main.
  - Failing test or lint blocks the workflow.

### T-033: Expand README with setup, training, serving, demo

- **Size:** M
- **Touches:** `README.md`
- **Dependencies:** T-024, T-030
- **Description:** Replace the v0.1 placeholder README with sections:
  - **Overview** (one paragraph + diagram link).
  - **Prerequisites** (Python 3.11+, Node 20+, optional CUDA).
  - **Backend setup** (clone, venv, `pip install -e ".[dev]"`, `cp .env.example .env`, `alembic upgrade head`).
  - **Training** (Colab notebook link, expected outputs, where the checkpoint lands).
  - **Serving** (`uvicorn api.server:app --reload`, browse `/docs`).
  - **Frontend** (`cd frontend && npm install && npm run dev`).
  - **Running tests** (`pytest`, `pytest -m slow`, `npm run test`).
  - **Project layout** (tree).
  - **Roadmap to v0.2** (pipeline animation, hot-swap model, PostgreSQL).
  - **License**.
- **Acceptance:**
  - Every command in the README runs successfully on a fresh clone.
  - Internal links to `docs/specs.md` and `docs/architecture.md` work.

### T-034: Demo script and screenshots

- **Size:** S
- **Touches:** `docs/demo.md`, `docs/screenshots/` (gitignored or committed small PNGs)
- **Dependencies:** T-030, T-033
- **Description:** Write a short `docs/demo.md` walkthrough:
  1. Start backend (one command).
  2. Start frontend (one command).
  3. Try 3 prepared messages (one per priority tier).
  4. Inspect resulting tickets in the DB via `sqlite3 artifacts/tickets.db`.
  5. Hit `/health` to show model version.
  Capture 2-3 screenshots and link them inline.
- **Acceptance:**
  - Following `docs/demo.md` from a fresh clone reproduces the flow without surprises.
  - Screenshots committed (kept small, < 200 KB each).

---

## Task Summary

**Total:** 34 tasks. Roughly partitioned:

| Phase                   | Tasks     | Count |
| ----------------------- | --------- | ----- |
| Foundation              | T-001..006| 6     |
| Data                    | T-007..010| 4     |
| Model                   | T-011..014| 4     |
| Routing & Persistence   | T-015..019| 5     |
| API                     | T-020..024| 5     |
| Frontend                | T-025..030| 6     |
| Integration & Polish    | T-031..034| 4     |

**Critical path:** T-001 -> T-002, T-004 -> T-007 -> T-008..010 -> T-011 -> T-012, T-013 -> T-014 (training) -> T-020 -> T-021..024 (API) -> T-031 (E2E).

**Parallelizable:** T-025 (frontend scaffold) can start any time after T-001. T-015 (routing.yaml) needs T-007 only. T-032 (CI) needs T-001 and T-025.

**Definition of Done for v0.1:** all 34 tasks merged on `main`, NFR-1 and NFR-2 met by a real Colab training run, demo flow in `docs/demo.md` works end to end.

---
