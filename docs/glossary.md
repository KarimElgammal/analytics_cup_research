# Glossary

Key metrics and terms used in the player similarity analysis.

---

## Forward Metrics

| Metric | Description |
|--------|-------------|
| **danger_rate** | Percentage of final third entries that led to a shot. Higher values indicate more clinical finishing. |
| **central_pct** | Percentage of entries through the central zone (vs. wide areas). Central entries often indicate a striker-like role. |
| **half_space_pct** | Percentage of entries through half-spaces (between central and wide). Associated with creative playmakers. |
| **avg_separation** | Average distance (metres) from the nearest defender when entering the final third. Higher values suggest better movement to find space. |
| **avg_entry_speed** | Average speed (m/s) when crossing into the final third. Higher speeds indicate direct, penetrating runs. |
| **carry_pct** | Percentage of entries made by carrying the ball (dribbling) vs. receiving a pass. Low values indicate off-ball movement style. |
| **avg_passing_options** | Average number of teammates available for a pass when entering. Higher values suggest good timing of runs. |
| **avg_defensive_line_dist** | Average distance from the defensive line when entering. Indicates positioning depth. |

---

## Defender Metrics

| Metric | Description |
|--------|-------------|
| **stop_danger_rate** | Percentage of defensive engagements that completely stopped a dangerous attack. |
| **reduce_danger_rate** | Percentage of engagements that reduced but didn't eliminate danger. |
| **force_backward_rate** | Percentage of engagements forcing the attacker to play backwards. |
| **pressing_rate** | How often the defender engages proactively vs. reactively. High values indicate aggressive pressing. |
| **goal_side_rate** | Percentage of engagements where defender maintained goal-side position. |
| **avg_engagement_distance** | Average distance from own goal when engaging. Higher values indicate a high defensive line. |
| **beaten_by_possession_rate** | How often the defender is beaten while the opponent keeps possession. |
| **beaten_by_movement_rate** | How often the defender is beaten by off-ball movement. |

---

## Goalkeeper Metrics

| Metric | Description |
|--------|-------------|
| **pass_success_rate** | Percentage of distributions that successfully reach a teammate. |
| **avg_pass_distance** | Average distance of distributions (metres). Higher values indicate long distribution style. |
| **long_pass_pct** | Percentage of distributions that are long passes. |
| **short_pass_pct** | Percentage of distributions that are short passes. |
| **high_pass_pct** | Percentage of distributions that are aerial/high balls. |
| **quick_distribution_pct** | Percentage of distributions made quickly after gaining possession. Indicates tempo-setting ability. |
| **under_pressure_pct** | Percentage of distributions made while under pressure from opponents. |
| **to_attacking_third_pct** | Percentage of distributions reaching the attacking third directly. |

---

## Similarity Metrics

| Metric | Description |
|--------|-------------|
| **similarity_score** | Weighted cosine similarity (0-100%) between a player's profile and the target archetype. |

---

## Data Sources

| Source | File | Description |
|--------|------|-------------|
| **SkillCorner** | `dynamic_events.csv` | Game intelligence metrics from A-League broadcast tracking (10 matches). |
| **StatsBomb** | Free data API | Event data from World Cup 2022 used to compute archetype target profiles. |
