# Crypto Trading

Solo retail crypto trading rig. Separate from the stocks/options system in
`../Trading`.

## Status

Day 0 — directory scaffolded, no code yet.

## Quick start

```powershell
cd C:\Users\dougn\Documents\CryptoTrading

# 1. Copy env template + add your keys
copy .env.example .env
notepad .env

# 2. Set up Python venv
python -m venv .venv
.venv\Scripts\activate
pip install ccxt requests python-dotenv pandas numpy

# 3. (When code exists) run the system
python core/broker.py --status
```

## See also

- [`CLAUDE.md`](CLAUDE.md) — architecture, key differences from stocks system,
  what to build first, what to avoid.
- `../Trading/CLAUDE.md` — sister stocks/options project (for reference only;
  do not import code from it without justification).

## License

Personal use. No redistribution.
