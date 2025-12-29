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
| **avg_passing_options** | Average number of teammates available for a pass when entering. Higher values suggest good timing of runs. |
| **avg_defensive_line_dist** | Average distance from the defensive line when entering. Indicates positioning depth. |
| **quick_break_pct** | Percentage of entries during quick break (counter-attack) phases. High values indicate transition threat. |
| **one_touch_pct** | Percentage of entries involving one-touch play. Indicates quick combination play. |
| **penalty_area_pct** | Percentage of entries ending in the penalty area. Higher = more central/dangerous finishing positions. |
| **avg_opponents_bypassed** | Average number of opponents bypassed per entry. Higher = more direct penetration. |
| **forward_momentum_pct** | Percentage of entries with forward momentum. Indicates attacking intent. |

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
| **avg_engagement_angle** | Average angle of engagement (degrees). Indicates positioning approach. |
| **avg_consecutive_engagements** | Average consecutive engagements per defensive action. Higher = sustained pressure. |
| **close_at_start_pct** | Percentage of engagements where defender was close at possession start. Indicates proactive positioning. |
| **avg_possession_danger** | Average possession danger level faced. Higher = defending more dangerous situations. |

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
| **to_attacking_third_pct** | Percentage of distributions reaching the attacking third directly. |
| **pass_ahead_pct** | Percentage of distributions that go forward. Higher = more progressive distribution. |
| **avg_targeted_xthreat** | Average expected threat created by passes. Higher = more dangerous distribution. |
| **avg_safe_dangerous_options** | Average number of safe but dangerous passing options available. |
| **forward_momentum_pct** | Percentage of distributions with forward momentum. Indicates proactive distribution. |

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
