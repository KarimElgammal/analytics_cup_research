"""AI-powered player similarity insights using GitHub Models API (Phi-4)."""

import os
from pathlib import Path
import polars as pl

AVAILABLE_MODELS = {
    "phi-4": "microsoft/Phi-4",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

DEFAULT_MODEL = "phi-4"

SYSTEM_PROMPT = """You are a football scout (and here we are talking about football played by foot not the american football) and analyst writing player similarity reports.
Use Australian English (defence, centre, metre). Be practical and specific and avoid vague generalities and avoid overstatements.
Be concise and insightful. Focus on actionable scouting insights."""


def get_github_token() -> str | None:
    """Load GitHub token from file or environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    token_path = Path(__file__).parent.parent.parent / "github_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def set_github_token(token: str) -> None:
    """Set GitHub token in environment."""
    os.environ["GITHUB_TOKEN"] = token


def _call_model(prompt: str, model_key: str = DEFAULT_MODEL, max_tokens: int = 300) -> str:
    """Call GitHub Models API with Phi-4."""
    token = get_github_token()
    if not token:
        return "AI insights unavailable (no GitHub token provided)"

    model_id = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL])

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(token),
        )

        response = client.complete(
            messages=[
                SystemMessage(SYSTEM_PROMPT),
                UserMessage(prompt),
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            model=model_id,
        )

        return response.choices[0].message.content or "Empty response"

    except ImportError:
        import requests

        try:
            response = requests.post(
                "https://models.github.ai/inference/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"AI unavailable: {str(e)}"

    except Exception as e:
        return f"AI unavailable: {str(e)}"


def generate_similarity_insight(
    ranked_players: pl.DataFrame,
    alvarez_description: str,
    top_n: int = 5,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate AI insight comparing top candidates to Alvarez archetype."""
    from src.data.player_ages import get_player_age

    # Build player summaries with ages
    player_summaries = []
    for row in ranked_players.head(top_n).to_dicts():
        player_name = row.get('player_name', 'Unknown')
        age = row.get('age') or get_player_age(player_name)
        age_str = f"Age {age}, " if age else ""
        summary = (
            f"- {player_name} ({row.get('team_name', '')}): "
            f"{age_str}"
            f"Similarity {row.get('similarity_score', 0):.1f}%, "
            f"Danger rate {row.get('danger_rate', 0):.1f}%, "
            f"Central entries {row.get('central_pct', 0):.1f}%, "
            f"Separation {row.get('avg_separation', 0):.2f}m, "
            f"{row.get('total_entries', 0)} entries"
        )
        player_summaries.append(summary)

    prompt = f"""Analyse these A-League players compared to the Julian Alvarez archetype.

ALVAREZ ARCHETYPE:
{alvarez_description}

TOP CANDIDATES (ranked by similarity):
{chr(10).join(player_summaries)}

Write 3-4 paragraphs:
1. Which candidate most closely matches Alvarez's style and why (mention their age)
2. Key differences between the candidates and the archetype
3. Scouting recommendation: considering age and development potential, which player would you watch further

FORMATTING: Make all player names **bold** using markdown. Keep it practical and specific. Use Australian English. Be concise."""

    return _call_model(prompt, model, max_tokens=450)


def generate_player_report(
    player_name: str,
    player_profile: dict,
    similarity_score: float,
    alvarez_description: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate detailed scouting report for a single player."""

    prompt = f"""Write a brief scouting report comparing {player_name} to Julian Alvarez.

ALVAREZ ARCHETYPE:
{alvarez_description}

PLAYER PROFILE:
- Name: {player_name}
- Team: {player_profile.get('team_name', 'Unknown')}
- Similarity Score: {similarity_score:.1f}%
- Danger Rate: {player_profile.get('danger_rate', 0):.1f}%
- Central Entry %: {player_profile.get('central_pct', 0):.1f}%
- Half-Space Entry %: {player_profile.get('half_space_pct', 0):.1f}%
- Avg Separation: {player_profile.get('avg_separation', 0):.2f}m
- Avg Entry Speed: {player_profile.get('avg_entry_speed', 0):.1f} m/s
- Total Entries: {player_profile.get('total_entries', 0)}

Write 2 paragraphs: strengths that match Alvarez, and areas of difference.
Be specific and practical. Australian English."""

    return _call_model(prompt, model, max_tokens=250)


def has_valid_token() -> bool:
    """Check if a valid GitHub token exists."""
    token = get_github_token()
    return token is not None and len(token) > 10
