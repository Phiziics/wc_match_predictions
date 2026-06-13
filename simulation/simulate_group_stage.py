from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from match_edge_v5_predictor import MatchEdgeV5Predictor


def simulate_group_stage(n_sims=1000, seed=42):
    rng = np.random.default_rng(seed)

    fixtures_path = ROOT / "data/processed/fixture_context_group_stage_2026.csv"
    if not fixtures_path.exists():
        raise FileNotFoundError("fixture_context_group_stage_2026.csv not found.")

    fixtures = pd.read_csv(fixtures_path)

    predictor = MatchEdgeV5Predictor(
        teams_csv=ROOT / "data/processed/wc2026_teams_model_input.csv",
        features_csv=ROOT / "data/processed/team_form_elo_latest.csv",
        model_path=ROOT / "models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl",
    )

    teams = predictor.valid_teams()[["team_id", "team_name", "group_code"]].copy()
    finish_counts = {team_id: {"points": [], "advance_count": 0} for team_id in teams["team_id"]}

    for _ in range(n_sims):
        table = {
            row.team_id: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
            for row in teams.itertuples()
        }

        for row in fixtures.itertuples():
            result = predictor.predict(
                row.team_a_name,
                row.team_b_name,
                venue_country=row.country if row.country in ["Mexico", "Canada", "United States"] else "Other",
                team_a_rest_days=None if pd.isna(row.team_a_rest_days) else row.team_a_rest_days,
                team_b_rest_days=None if pd.isna(row.team_b_rest_days) else row.team_b_rest_days,
                team_a_travel_km=None if pd.isna(row.team_a_travel_km) else row.team_a_travel_km,
                team_b_travel_km=None if pd.isna(row.team_b_travel_km) else row.team_b_travel_km,
            )

            probs = [
                result["team_b_win_probability"],
                result["draw_probability"],
                result["team_a_win_probability"]
            ]
            outcome = rng.choice([0, 1, 2], p=np.array(probs) / np.sum(probs))

            # Expected-goal rounded proxy for goals, enough for simulation scaffold.
            a_goals = max(0, int(round(result["team_a_expected_goals"])))
            b_goals = max(0, int(round(result["team_b_expected_goals"])))

            if outcome == 2:
                if a_goals <= b_goals:
                    a_goals = b_goals + 1
                table[row.team_a_id]["points"] += 3
            elif outcome == 0:
                if b_goals <= a_goals:
                    b_goals = a_goals + 1
                table[row.team_b_id]["points"] += 3
            else:
                avg = int(round((a_goals + b_goals) / 2))
                a_goals = b_goals = avg
                table[row.team_a_id]["points"] += 1
                table[row.team_b_id]["points"] += 1

            table[row.team_a_id]["gf"] += a_goals
            table[row.team_a_id]["ga"] += b_goals
            table[row.team_b_id]["gf"] += b_goals
            table[row.team_b_id]["ga"] += a_goals

        for tid in table:
            table[tid]["gd"] = table[tid]["gf"] - table[tid]["ga"]
            finish_counts[tid]["points"].append(table[tid]["points"])

        for group, group_teams in teams.groupby("group_code"):
            ordered = sorted(
                group_teams["team_id"],
                key=lambda tid: (table[tid]["points"], table[tid]["gd"], table[tid]["gf"]),
                reverse=True
            )
            # 48-team WC: top 2 definitely advance; best third-place teams require cross-group logic.
            for tid in ordered[:2]:
                finish_counts[tid]["advance_count"] += 1

    rows = []
    for _, row in teams.iterrows():
        tid = row["team_id"]
        rows.append({
            "team_id": tid,
            "team_name": row["team_name"],
            "group_code": row["group_code"],
            "avg_group_points": float(np.mean(finish_counts[tid]["points"])),
            "prob_top_two_group": finish_counts[tid]["advance_count"] / n_sims,
            "simulations": n_sims
        })

    out = pd.DataFrame(rows).sort_values(["group_code", "prob_top_two_group"], ascending=[True, False])
    output_path = ROOT / "reports/group_stage_simulation_results.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    simulate_group_stage()
