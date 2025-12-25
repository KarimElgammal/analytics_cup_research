# Computing Similarity

The `SimilarityEngine` computes weighted cosine similarity between players and an archetype.

## Basic Usage

```python
from src.core import Archetype, SimilarityEngine

archetype = Archetype.alvarez()
engine = SimilarityEngine(archetype)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)
```

## Explaining Scores

```python
explanation = engine.explain("Z. Clough")
print(f"Score: {explanation['similarity_score']}%")
```

## Feature Importance

```python
importance = engine.get_feature_importance()
for item in importance:
    print(f"{item['feature']}: {item['weight']:.2%}")
```
