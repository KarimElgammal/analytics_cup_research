"""AI-powered player similarity insights using GitHub Models or HuggingFace Inference."""

import os
from pathlib import Path
import polars as pl

# Available models by backend
AVAILABLE_MODELS = {
    # GitHub Models (primary - better availability)
    "grok-3-mini": "xai/grok-3-mini",
    "phi-4": "microsoft/Phi-4",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # HuggingFace Inference Providers (fast/medium only)
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct-1M",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "gemma-2-2b": "google/gemma-2-2b-it",
}

# Backend mapping
HF_MODELS = {"llama-3.1-8b", "llama-3.2-3b", "qwen-2.5-7b", "smollm3-3b", "gemma-2-2b"}
GITHUB_MODELS = {"grok-3-mini", "phi-4", "gpt-4o-mini"}

# Defaults
DEFAULT_HF_MODEL = "llama-3.1-8b"
DEFAULT_GITHUB_MODEL = "phi-4"

# Model display names for UI
MODEL_DISPLAY_NAMES = {
    "grok-3-mini": "Grok 3 Mini (GitHub)",
    "phi-4": "Phi-4 (GitHub)",
    "gpt-4o-mini": "GPT-4o Mini (GitHub)",
    "llama-3.1-8b": "Llama 3.1 8B (HF)",
    "llama-3.2-3b": "Llama 3.2 3B (HF)",
    "qwen-2.5-7b": "Qwen 2.5 7B (HF)",
    "smollm3-3b": "SmolLM3 3B (HF)",
    "gemma-2-2b": "Gemma 2 2B (HF)",
}

SYSTEM_PROMPT = """You are a football scout (and here we are talking about football played by foot not the american football) and analyst writing player similarity reports.
Use Australian English (defence, centre, metre). Be practical and specific and avoid vague generalities and avoid overstatements.
Be concise and insightful. Focus on actionable scouting insights."""

# Position-specific metrics for player summaries
POSITION_METRICS = {
    "forward": {
        "metrics": ["danger_rate", "central_pct", "avg_separation", "avg_entry_speed", "carry_pct"],
        "count_field": "total_entries",
        "auc": 0.656,
        "criteria": "Forwards are valued for creating danger through intelligent movement, finding space between defenders, and clinical finishing. Key traits: high danger rate (shots per entry), good separation from markers, and effective use of central zones.",
    },
    "defender": {
        "metrics": ["stop_danger_rate", "reduce_danger_rate", "pressing_rate", "goal_side_rate", "avg_engagement_distance"],
        "count_field": "total_engagements",
        "auc": 0.845,
        "criteria": "Defenders are valued for stopping dangerous attacks, maintaining goal-side positioning, and intelligent pressing. Key traits: high stop/reduce danger rates, consistent goal-side positioning, and appropriate engagement distance.",
    },
    "goalkeeper": {
        "metrics": ["pass_success_rate", "avg_pass_distance", "long_pass_pct", "quick_distribution_pct"],
        "count_field": "total_distributions",
        "auc": 0.993,
        "criteria": "Goalkeepers are valued for distribution quality and decision-making under pressure. Key traits: high pass success rate, appropriate pass distance for team style, and quick distribution to initiate attacks.",
    },
}

# Metric display names for readability
METRIC_LABELS = {
    "danger_rate": "Danger Rate",
    "central_pct": "Central %",
    "avg_separation": "Separation",
    "avg_entry_speed": "Entry Speed",
    "carry_pct": "Carry %",
    "stop_danger_rate": "Stop Danger %",
    "reduce_danger_rate": "Reduce Danger %",
    "pressing_rate": "Pressing %",
    "goal_side_rate": "Goal-Side %",
    "avg_engagement_distance": "Engagement Dist",
    "pass_success_rate": "Pass Success %",
    "avg_pass_distance": "Pass Distance",
    "long_pass_pct": "Long Pass %",
    "quick_distribution_pct": "Quick Dist %",
}


def get_github_token() -> str | None:
    """Load GitHub token from file or environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    token_path = Path(__file__).parent.parent.parent / "github_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def get_hf_token() -> str | None:
    """Load HF token from environment (for Spaces) or file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    token_path = Path(__file__).parent.parent.parent / "hf_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def set_github_token(token: str) -> None:
    """Set GitHub token in environment."""
    os.environ["GITHUB_TOKEN"] = token


def set_hf_token(token: str) -> None:
    """Set HF token in environment."""
    os.environ["HF_TOKEN"] = token


def get_default_model() -> str:
    """Get default model based on available tokens.

    Prefers GitHub Models (better model availability) over HuggingFace.
    As of 2025, HF serverless focuses on CPU inference only.
    """
    if get_github_token():
        return DEFAULT_GITHUB_MODEL
    if get_hf_token():
        return DEFAULT_HF_MODEL
    return DEFAULT_GITHUB_MODEL


def get_available_backend() -> str | None:
    """Return which backend is available.

    Prefers GitHub Models over HuggingFace for text generation.
    """
    if get_github_token():
        return "github"
    if get_hf_token():
        return "huggingface"
    return None


def _call_github_model(prompt: str, model_key: str, max_tokens: int = 300) -> str:
    """Call GitHub Models API."""
    token = get_github_token()
    if not token:
        return "AI insights unavailable (no GitHub token)"

    model_id = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_GITHUB_MODEL])

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
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_tokens,
            model=model_id,
        )

        content = response.choices[0].message.content
        if not content:
            return f"Empty response from {model_id}"
        return content

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
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not content:
                return f"Empty response from {model_id}"
            return content
        except Exception as e:
            return f"AI unavailable: {str(e)}"

    except Exception as e:
        return f"AI unavailable: {str(e)}"


def _call_hf_model(prompt: str, model_id: str, max_tokens: int = 300) -> str:
    """Call HuggingFace Inference API."""
    token = get_hf_token()
    if not token:
        return "AI insights unavailable (no HF token)"

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)
        response = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or "Empty response"

    except ImportError:
        # fallback to requests
        import requests

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions",
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
                    "temperature": 0.7,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"AI unavailable: {str(e)}"

    except Exception as e:
        return f"AI unavailable: {str(e)}"


def _call_model(prompt: str, model_key: str | None = None, max_tokens: int = 300) -> str:
    """Route to appropriate backend based on model."""
    if model_key is None:
        model_key = get_default_model()

    if model_key in HF_MODELS:
        model_id = AVAILABLE_MODELS[model_key]
        return _call_hf_model(prompt, model_id, max_tokens)
    else:
        return _call_github_model(prompt, model_key, max_tokens)


def _format_metric_value(value, metric_name: str) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    if "pct" in metric_name or "rate" in metric_name:
        return f"{value:.1f}%"
    elif "distance" in metric_name or "separation" in metric_name:
        return f"{value:.1f}m"
    elif "speed" in metric_name:
        return f"{value:.1f}m/s"
    else:
        return f"{value:.1f}"


def _build_player_summary(row: dict, position_type: str) -> str:
    """Build a player summary string based on position type."""
    from src.data.player_ages import get_player_age

    config = POSITION_METRICS.get(position_type, POSITION_METRICS["forward"])

    player_name = row.get('player_name', 'Unknown')
    team = row.get('team_name', '')
    age = row.get('age') or get_player_age(player_name)
    age_str = f"Age {age}" if age else "Age unknown"
    similarity = row.get('similarity_score', 0)
    count = row.get(config["count_field"], 0)

    # Build metrics string
    metrics_parts = []
    for metric in config["metrics"]:
        value = row.get(metric)
        if value is not None:
            label = METRIC_LABELS.get(metric, metric)
            formatted = _format_metric_value(value, metric)
            metrics_parts.append(f"{label}: {formatted}")

    metrics_str = ", ".join(metrics_parts)
    count_label = config["count_field"].replace("total_", "")

    return f"- **{player_name}** ({team}): {age_str}, Similarity {similarity:.1f}%, {count} {count_label}. {metrics_str}"


def _compute_dataset_averages(profiles: pl.DataFrame, position_type: str) -> str:
    """Compute average metrics across the dataset for comparison."""
    config = POSITION_METRICS.get(position_type, POSITION_METRICS["forward"])

    avg_parts = []
    for metric in config["metrics"]:
        if metric in profiles.columns:
            avg_val = profiles[metric].mean()
            if avg_val is not None:
                label = METRIC_LABELS.get(metric, metric)
                formatted = _format_metric_value(avg_val, metric)
                avg_parts.append(f"{label}: {formatted}")

    return ", ".join(avg_parts) if avg_parts else "N/A"


def generate_similarity_insight(
    ranked_players: pl.DataFrame,
    archetype,
    top_n: int = 5,
    model: str | None = None,
    position_type: str = "forward",
) -> str:
    """Generate AI insight comparing top candidates to an archetype.

    Args:
        ranked_players: DataFrame with ranked player profiles
        archetype: Either an Archetype object (with .name and .description)
                   or a string description for backwards compatibility
        top_n: Number of top candidates to include
        model: Model to use for generation
        position_type: One of "forward", "defender", "goalkeeper"

    Returns:
        AI-generated scouting insight as markdown string
    """
    # Handle both Archetype objects and string descriptions
    if hasattr(archetype, 'name') and hasattr(archetype, 'description'):
        archetype_name = archetype.name.replace('_', ' ').title()
        archetype_description = archetype.description
    else:
        # Backwards compatibility: assume string description for Alvarez
        archetype_name = "Julian Alvarez"
        archetype_description = str(archetype)

    # Get position configuration
    config = POSITION_METRICS.get(position_type, POSITION_METRICS["forward"])

    # Build player summaries with position-specific metrics
    player_summaries = []
    for row in ranked_players.head(top_n).to_dicts():
        summary = _build_player_summary(row, position_type)
        player_summaries.append(summary)

    # Compute dataset averages for comparison
    dataset_averages = _compute_dataset_averages(ranked_players, position_type)

    # Get counts
    total_players = len(ranked_players)
    count_field = config["count_field"]
    total_events = ranked_players[count_field].sum() if count_field in ranked_players.columns else 0

    prompt = f"""Analyse these A-League {position_type}s compared to the {archetype_name} archetype.

ANALYSIS CONTEXT:
- Position: {position_type.title()}
- Dataset: {total_events} {count_field.replace('total_', '')} from {total_players} players across 10 A-League matches
- ML Model AUC: {config['auc']:.3f} (higher = more reliable ranking)

{archetype_name.upper()} ARCHETYPE:
{archetype_description}

WHAT MAKES A GREAT {position_type.upper()}:
{config['criteria']}

TOP {top_n} CANDIDATES (ranked by similarity):
{chr(10).join(player_summaries)}

DATASET AVERAGES (for comparison):
{dataset_averages}

Write 3-4 paragraphs:
1. Which candidate most closely matches {archetype_name}'s style and why (mention their age and specific metrics)
2. Key differences between the top candidates and the archetype (be specific about which metrics diverge)
3. Scouting recommendation: considering age, development potential, and sample size, which player deserves further observation

FORMATTING: Player names are already bold. Keep it practical and specific. Use Australian English. Be concise."""

    return _call_model(prompt, model, max_tokens=500)


def generate_player_report(
    player_name: str,
    player_profile: dict,
    similarity_score: float,
    archetype,
    model: str | None = None,
    position_type: str = "forward",
) -> str:
    """Generate detailed scouting report for a single player.

    Args:
        player_name: Name of the player
        player_profile: Dictionary with player metrics
        similarity_score: Similarity percentage
        archetype: Either an Archetype object or a string description
        model: Model to use for generation
        position_type: One of "forward", "defender", "goalkeeper"

    Returns:
        AI-generated scouting report as markdown string
    """
    # Handle both Archetype objects and string descriptions
    if hasattr(archetype, 'name') and hasattr(archetype, 'description'):
        archetype_name = archetype.name.replace('_', ' ').title()
        archetype_description = archetype.description
    else:
        archetype_name = "Julian Alvarez"
        archetype_description = str(archetype)

    # Get position configuration
    config = POSITION_METRICS.get(position_type, POSITION_METRICS["forward"])

    # Build metrics string
    metrics_lines = []
    for metric in config["metrics"]:
        value = player_profile.get(metric)
        if value is not None:
            label = METRIC_LABELS.get(metric, metric)
            formatted = _format_metric_value(value, metric)
            metrics_lines.append(f"- {label}: {formatted}")

    count_field = config["count_field"]
    count = player_profile.get(count_field, 0)
    metrics_lines.append(f"- Total {count_field.replace('total_', '')}: {count}")

    prompt = f"""Write a brief scouting report comparing {player_name} to {archetype_name}.

{archetype_name.upper()} ARCHETYPE:
{archetype_description}

PLAYER PROFILE:
- Name: {player_name}
- Team: {player_profile.get('team_name', 'Unknown')}
- Similarity Score: {similarity_score:.1f}%
{chr(10).join(metrics_lines)}

Write 2 paragraphs: strengths that match {archetype_name}, and areas of difference.
Be specific and practical. Australian English."""

    return _call_model(prompt, model, max_tokens=300)


def has_valid_token() -> bool:
    """Check if any valid token exists (GitHub or HF)."""
    gh_token = get_github_token()
    hf_token = get_hf_token()
    return (gh_token and len(gh_token) > 10) or (hf_token and len(hf_token) > 10)


def get_available_models() -> list[tuple[str, str]]:
    """Get list of available models based on tokens.

    Returns list of (model_key, display_name) tuples.
    """
    models = []
    gh_token = get_github_token()
    hf_token = get_hf_token()

    # Add GitHub models if token available
    if gh_token and len(gh_token) > 10:
        for key in GITHUB_MODELS:
            models.append((key, MODEL_DISPLAY_NAMES.get(key, key)))

    # Add HF models if token available
    if hf_token and len(hf_token) > 10:
        for key in HF_MODELS:
            models.append((key, MODEL_DISPLAY_NAMES.get(key, key)))

    return models
