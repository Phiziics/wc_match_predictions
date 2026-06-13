import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from match_edge_v5_predictor import MatchEdgeV5Predictor


def optional_value(row, name, default=None):
    if name not in row or pd.isna(row[name]):
        return default
    return row[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with team_a, team_b, venue_country and optional context columns.")
    parser.add_argument("--output", default=str(ROOT / "reports/batch_predictions.csv"))
    args = parser.parse_args()

    predictor = MatchEdgeV5Predictor(
        teams_csv=ROOT / "data/processed/wc2026_teams_model_input.csv",
        features_csv=ROOT / "data/processed/team_form_elo_latest.csv",
        model_path=ROOT / "models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl",
    )

    matches = pd.read_csv(args.input)
    rows = []

    for _, row in matches.iterrows():
        result = predictor.predict(
            row["team_a"],
            row["team_b"],
            venue_country=row.get("venue_country", "Other"),
            team_a_rest_days=optional_value(row, "team_a_rest_days"),
            team_b_rest_days=optional_value(row, "team_b_rest_days"),
            team_a_travel_km=optional_value(row, "team_a_travel_km"),
            team_b_travel_km=optional_value(row, "team_b_travel_km"),
            temperature_c=optional_value(row, "temperature_c", 22.0),
            humidity_pct=optional_value(row, "humidity_pct", 50.0),
        )
        rows.append({k: v for k, v in result.items() if k != "specialist_breakdown"})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
