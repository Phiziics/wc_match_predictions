import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from match_edge_v5_predictor import MatchEdgeV5Predictor


st.set_page_config(page_title="Match Edge AI V5", layout="centered")

st.title("Match Edge AI V5")
st.subheader("Calibrated Specialist Ensemble")
st.caption("World Cup 2026 team-only predictor.")

predictor = MatchEdgeV5Predictor(
    teams_csv=ROOT / "data/processed/wc2026_teams_model_input.csv",
    features_csv=ROOT / "data/processed/team_form_elo_latest.csv",
    model_path=ROOT / "models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl",
)

teams = predictor.valid_teams()
team_names = teams["team_name"].tolist()

team_a = st.selectbox("Team A", team_names, index=team_names.index("United States"))
team_b = st.selectbox("Team B", team_names, index=team_names.index("Paraguay"))
venue_country = st.selectbox("Venue country", ["Other", "Mexico", "Canada", "United States"], index=3)

with st.expander("Optional fixture context"):
    team_a_rest_days = st.number_input("Team A rest days", min_value=0.0, max_value=14.0, value=0.0, step=0.5)
    team_b_rest_days = st.number_input("Team B rest days", min_value=0.0, max_value=14.0, value=0.0, step=0.5)
    team_a_travel_km = st.number_input("Team A travel km", min_value=0.0, max_value=6000.0, value=0.0, step=50.0)
    team_b_travel_km = st.number_input("Team B travel km", min_value=0.0, max_value=6000.0, value=0.0, step=50.0)
    temperature_c = st.number_input("Temperature C", min_value=-10.0, max_value=45.0, value=22.0, step=1.0)
    humidity_pct = st.number_input("Humidity %", min_value=0.0, max_value=100.0, value=50.0, step=5.0)

if st.button("Predict match"):
    result = predictor.predict(
        team_a,
        team_b,
        venue_country=venue_country,
        team_a_rest_days=team_a_rest_days,
        team_b_rest_days=team_b_rest_days,
        team_a_travel_km=team_a_travel_km,
        team_b_travel_km=team_b_travel_km,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
    )

    st.write(f"## {result['team_a_name']} vs {result['team_b_name']}")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{result['team_a_name']} win", f"{result['team_a_win_probability']:.1%}")
    col2.metric("Draw", f"{result['draw_probability']:.1%}")
    col3.metric(f"{result['team_b_name']} win", f"{result['team_b_win_probability']:.1%}")

    st.success(f"Final prediction: {result['predicted_result']}")
    st.info(f"Expected score: {result['team_a_name']} {result['team_a_expected_goals']} - {result['team_b_expected_goals']} {result['team_b_name']}")

    st.write("### Main signals")
    st.write(f"- Elo difference: `{result['elo_difference']}`")
    st.write(f"- FIFA rank: `{result['team_a_name']} {result['team_a_fifa_rank']}` vs `{result['team_b_name']} {result['team_b_fifa_rank']}`")
    st.write(f"- Rest days difference: `{result['rest_days_difference']}`")
    st.write(f"- Travel km difference: `{result['travel_km_difference']}`")
    st.write(f"- Calibration temperature: `{result['calibration_temperature']:.3f}`")

    st.write("### Specialist breakdown")
    breakdown = pd.DataFrame(result["specialist_breakdown"]).T
    st.dataframe(breakdown, use_container_width=True)

with st.expander("Valid teams"):
    st.dataframe(teams, use_container_width=True)
