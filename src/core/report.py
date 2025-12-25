"""AI-powered scouting report generation."""

from __future__ import annotations
import os
from pathlib import Path
from src.core.similarity import SimilarityEngine


class ScoutingReport:
    """Generate AI-powered scouting reports."""

    AVAILABLE_MODELS = {"phi-4": "microsoft/Phi-4", "gpt-4o-mini": "openai/gpt-4o-mini"}
    SYSTEM_PROMPT = "You are a football scout writing player similarity reports. Use Australian English. Be practical and specific."

    def __init__(self, engine: SimilarityEngine, model: str = "phi-4"):
        self.engine = engine
        self.model = model
        self._token: str | None = None

    def set_token(self, token: str) -> ScoutingReport:
        self._token = token
        os.environ["GITHUB_TOKEN"] = token
        return self

    def _get_token(self) -> str | None:
        if self._token:
            return self._token
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            return token
        token_path = Path(__file__).parent.parent.parent / "github_token.txt"
        if token_path.exists():
            return token_path.read_text().strip()
        return None

    def has_valid_token(self) -> bool:
        token = self._get_token()
        return token is not None and len(token) > 10

    def _call_model(self, prompt: str, max_tokens: int = 300) -> str:
        token = self._get_token()
        if not token:
            return "AI insights unavailable (no GitHub token provided)"
        import requests
        try:
            response = requests.post(
                "https://models.github.ai/inference/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": self.AVAILABLE_MODELS.get(self.model, "microsoft/Phi-4"),
                      "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                      "max_tokens": max_tokens},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"AI unavailable: {str(e)}"

    def generate(self, top_n: int = 5) -> str:
        if self.engine.rankings is None:
            self.engine.rank(top_n=top_n)
        rankings = self.engine.rankings.head(top_n)
        summaries = []
        for row in rankings.to_dicts():
            age = row.get("age")
            age_str = f"Age {age}, " if age else ""
            summaries.append(f"- {row.get('player_name', 'Unknown')} ({row.get('team_name', '')}): {age_str}Similarity {row.get('similarity_score', 0):.1f}%, Danger rate {row.get('danger_rate', 0):.1f}%")
        prompt = f"Analyse these A-League players compared to the {self.engine.archetype.name.title()} archetype.\n\nTOP CANDIDATES:\n" + "\n".join(summaries) + "\n\nWrite 3-4 paragraphs. Make player names **bold**. Use Australian English."
        return self._call_model(prompt, max_tokens=450)

    def player_report(self, player_name: str) -> str:
        profile = next((r for r in self.engine.profiles.to_dicts() if r.get("player_name") == player_name), {})
        prompt = f"Write a brief scouting report for {player_name} compared to the {self.engine.archetype.name} archetype. Danger Rate: {profile.get('danger_rate', 0):.1f}%, Separation: {profile.get('avg_separation', 0):.2f}m. Be specific. Australian English."
        return self._call_model(prompt, max_tokens=250)
