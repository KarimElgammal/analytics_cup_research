# CLAUDE.md

Project context for AI assistants.

## Project

SkillCorner X PySport Analytics Cup 2026 - Research Track

## Goal

Player similarity analysis: identify A-League players matching world-class archetypes using SkillCorner tracking data.

## Structure

```
app.py                 # Streamlit app
run.sh                 # App launcher
submission.ipynb       # Research notebook
src/
  core/                # Archetype, Similarity engine
  data/                # Data loading
  analysis/            # Entry, defender, goalkeeper profiles
  statsbomb/           # StatsBomb archetype factory
  utils/               # AI insights, rate limiter
  visualization/       # Plotting
docs/                  # MkDocs documentation
```

## Running

```bash
./run.sh
# or: uv run streamlit run app.py
```

## AI Insights

| Backend | Models | Token |
|---------|--------|-------|
| GitHub Models | Phi-4, GPT-4o Mini | `github_token.txt` or `GITHUB_TOKEN` |
| HuggingFace | Llama 3.1/3.2, Qwen 2.5, SmolLM3, Gemma 2 | `hf_token.txt` or `HF_TOKEN` |

Token files are gitignored. For HF Spaces deployment, use Secrets.

## Code Style

- polars, not pandas
- Modular: logic in src/, interface in app/notebook
- Clean code, minimal comments

## Key Files

- `src/utils/ai_insights.py` - LLM backends (GitHub Models, HuggingFace)
- `src/utils/rate_limiter.py` - Rate limiting ($2/month per backend)
- `src/core/similarity.py` - Similarity engine
- `src/core/archetype.py` - Archetype definitions

## Archetypes (12 total)

| Position | Archetypes | AUC |
|----------|------------|-----|
| Forwards | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri | 0.656 |
| Defenders | Gvardiol, Romero, Hakimi | 0.845 |
| Goalkeepers | Lloris, Livakovic, Bounou | 0.993 |

## Documentation

https://karimelgammal.github.io/analytics_cup_research/
