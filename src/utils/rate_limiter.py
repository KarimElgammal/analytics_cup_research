"""Rate limiting for AI insights with per-backend budgets."""

import json
from pathlib import Path
from datetime import date

RATE_LIMIT_FILE = Path(__file__).parent.parent.parent / ".rate_limits.json"

# Configurable limits
MAX_CALLS_PER_SESSION = 5      # per visitor session
MAX_CALLS_PER_DAY = 100        # across all visitors (total)
MAX_MONTHLY_COST_PER_BACKEND = 2.00  # $2 per backend (github, huggingface)

BACKENDS = ["github", "huggingface"]


def load_limits() -> dict:
    """Load rate limit data from file."""
    if RATE_LIMIT_FILE.exists():
        try:
            return json.loads(RATE_LIMIT_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return _default_limits()


def _default_limits() -> dict:
    """Return default limits structure."""
    return {
        "daily_calls": 0,
        "last_reset": str(date.today()),
        "github_cost": 0.0,
        "huggingface_cost": 0.0,
    }


def save_limits(data: dict) -> None:
    """Save rate limit data to file."""
    try:
        RATE_LIMIT_FILE.write_text(json.dumps(data))
    except IOError:
        pass  # fail silently on HF Spaces if no write access


def check_daily_limit() -> tuple[bool, int]:
    """Check if daily limit reached. Returns (allowed, remaining)."""
    data = load_limits()

    # reset if new day
    if data.get("last_reset") != str(date.today()):
        data = _default_limits()
        # preserve monthly costs
        save_limits(data)

    remaining = MAX_CALLS_PER_DAY - data.get("daily_calls", 0)
    return remaining > 0, max(0, remaining)


def check_backend_budget(backend: str) -> tuple[bool, float]:
    """Check if backend budget exceeded. Returns (allowed, remaining)."""
    data = load_limits()
    cost_key = f"{backend}_cost"
    current_cost = data.get(cost_key, 0.0)
    remaining = MAX_MONTHLY_COST_PER_BACKEND - current_cost
    return remaining > 0, max(0.0, remaining)


def increment_call(backend: str = "github", estimated_cost: float = 0.001) -> None:
    """Record that a call was made for a specific backend."""
    data = load_limits()

    # reset daily calls if new day
    if data.get("last_reset") != str(date.today()):
        data["daily_calls"] = 0
        data["last_reset"] = str(date.today())

    data["daily_calls"] = data.get("daily_calls", 0) + 1

    # increment backend-specific cost
    cost_key = f"{backend}_cost"
    data[cost_key] = data.get(cost_key, 0.0) + estimated_cost

    save_limits(data)


def reset_monthly_costs() -> None:
    """Reset all monthly cost counters."""
    data = load_limits()
    data["github_cost"] = 0.0
    data["huggingface_cost"] = 0.0
    save_limits(data)


def get_usage_stats() -> dict:
    """Get current usage stats for display."""
    data = load_limits()
    return {
        "daily_calls": data.get("daily_calls", 0),
        "daily_limit": MAX_CALLS_PER_DAY,
        "github_cost": data.get("github_cost", 0.0),
        "huggingface_cost": data.get("huggingface_cost", 0.0),
        "budget_per_backend": MAX_MONTHLY_COST_PER_BACKEND,
        "last_reset": data.get("last_reset", str(date.today())),
    }


def get_backend_from_model(model_key: str) -> str:
    """Determine backend from model key."""
    from src.utils.ai_insights import GITHUB_MODELS
    return "github" if model_key in GITHUB_MODELS else "huggingface"
