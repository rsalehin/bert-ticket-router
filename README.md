# BERT Ticket Router

Customer-service intent classification on PolyAI/banking77, with full ticket
routing (department, priority, SLA, tags), a FastAPI service, SQL persistence,
and a React + Vite + Tailwind demo frontend.

See [docs/specs.md](docs/specs.md) for the full specification.

## Status

v0.1.0 — in development. Built in a human-in-the-loop SDLC:
requirements -> architecture -> tasks -> TDD -> review.

## Stack

- **Model:** BERT-base-uncased fine-tuned with a custom PyTorch loop
- **Dataset:** PolyAI/banking77 (77 banking intents)
- **Training:** Google Colab Pro, A100 GPU
- **Serving:** FastAPI + SQLAlchemy + SQLite (local)
- **Frontend:** React + Vite + TypeScript + Tailwind

## License

MIT
