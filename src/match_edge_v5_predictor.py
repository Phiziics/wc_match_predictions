from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


HOST_COUNTRIES = {"Mexico", "Canada", "United States"}

ALIASES = {
    "usa": "USA", "us": "USA", "u.s.": "USA",
    "united states": "USA", "united states of america": "USA",
    "south africa": "RSA",
    "bosnia": "BIH", "bosnia-herzegovina": "BIH", "bosnia and herzegovina": "BIH",
    "south korea": "KOR", "korea republic": "KOR",
    "czech republic": "CZE", "czechia": "CZE",
    "turkey": "TUR", "turkiye": "TUR", "türkiye": "TUR",
    "ivory coast": "CIV", "côte d’ivoire": "CIV", "côte d'ivoire": "CIV",
    "dr congo": "COD", "congo dr": "COD", "democratic republic of the congo": "COD",
    "curacao": "CUW", "curaçao": "CUW",
    "cape verde": "CPV", "saudi arabia": "KSA", "new zealand": "NZL",
}


def clean_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("’", "'").replace("ʻ", "'")
    value = re.sub(r"\s+", " ", value)
    return value




def patch_sklearn_logistic_compatibility(obj):
    """
    Fixes scikit-learn pickle compatibility for LogisticRegression objects.

    Some scikit-learn versions expect LogisticRegression.multi_class during
    predict_proba, while newer pickles may not store that attribute.
    """
    visited = set()

    def patch(current):
        obj_id = id(current)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if current.__class__.__name__ == "LogisticRegression":
            if not hasattr(current, "multi_class"):
                current.multi_class = "auto"
            return

        if hasattr(current, "steps"):
            for _, step in current.steps:
                patch(step)

        if hasattr(current, "estimators_"):
            for estimator in current.estimators_:
                patch(estimator)

        if isinstance(current, dict):
            for value in current.values():
                patch(value)
        elif isinstance(current, (list, tuple)):
            for value in current:
                patch(value)

    patch(obj)
    return obj


def force_patch_logistic_regression(obj):
    """
    Forces old/new scikit-learn LogisticRegression pickle compatibility.

    This runs directly before predict_proba, so it catches Pipeline objects
    even if the bundle-level patch did not run.
    """
    visited = set()

    def patch(current):
        obj_id = id(current)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if current.__class__.__name__ == "LogisticRegression":
            current.multi_class = getattr(current, "multi_class", "auto") or "auto"
            return

        if hasattr(current, "steps"):
            for _, step in current.steps:
                patch(step)

        if hasattr(current, "named_steps"):
            for step in current.named_steps.values():
                patch(step)

        if hasattr(current, "estimators_"):
            for estimator in current.estimators_:
                patch(estimator)

        if hasattr(current, "estimators"):
            for estimator in current.estimators:
                if isinstance(estimator, tuple) and len(estimator) >= 2:
                    patch(estimator[1])
                else:
                    patch(estimator)

        if isinstance(current, dict):
            for value in current.values():
                patch(value)
        elif isinstance(current, (list, tuple)):
            for value in current:
                patch(value)

    patch(obj)
    return obj


def predict_proba_aligned(model, X: pd.DataFrame) -> np.ndarray:
    # Patch every time before probability prediction.
    model = force_patch_logistic_regression(model)

    try:
        probs = model.predict_proba(X)
    except AttributeError as error:
        if "multi_class" not in str(error):
            raise

        # Patch again and retry once.
        model = force_patch_logistic_regression(model)
        probs = model.predict_proba(X)

    classes = list(model.classes_)
    out = np.zeros((len(X), 3))

    for idx, cls in enumerate(classes):
        out[:, int(cls)] = probs[:, idx]

    return out

def poisson_pmf(k, lam):
    lam = np.clip(lam, 0.05, 6.0)
    return np.exp(-lam) * np.power(lam, k) / math.factorial(k)


def poisson_1x2_from_lambdas(lambda_a, lambda_b, max_goals=7):
    rows = []
    for la, lb in zip(np.asarray(lambda_a), np.asarray(lambda_b)):
        a_probs = np.array([poisson_pmf(g, la) for g in range(max_goals + 1)])
        b_probs = np.array([poisson_pmf(g, lb) for g in range(max_goals + 1)])
        matrix = np.outer(a_probs, b_probs)
        a_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        b_win = float(np.triu(matrix, 1).sum())
        total = max(a_win + draw + b_win, 1e-9)
        rows.append([b_win / total, draw / total, a_win / total])
    return np.array(rows)


def draw_to_1x2_probs(draw_prob, anchor_probs):
    rows = []
    for d, anchor in zip(draw_prob, anchor_probs):
        d = float(np.clip(d, 0.05, 0.70))
        non_draw = 1.0 - d
        b_weight = max(float(anchor[0]), 1e-6)
        a_weight = max(float(anchor[2]), 1e-6)
        total = b_weight + a_weight
        rows.append([non_draw * b_weight / total, d, non_draw * a_weight / total])
    return np.array(rows)


def temperature_scale_probs(probs, temperature):
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def context_adjustment_probs(base_probs, rest_diff=0.0, travel_diff_km=0.0,
                             temperature_c=22.0, humidity_pct=50.0):
    edge = 0.0
    edge += np.clip(rest_diff, -4, 4) * 0.035
    edge += np.clip(-travel_diff_km / 1000.0, -3, 3) * 0.035

    draw_boost = 0.0
    if temperature_c >= 30:
        draw_boost += min((temperature_c - 30) * 0.006, 0.04)
    if humidity_pct >= 70:
        draw_boost += min((humidity_pct - 70) * 0.001, 0.03)

    edge = float(np.clip(edge, -0.15, 0.15))
    adjusted = []
    for p in np.asarray(base_probs):
        p = np.clip(p, 1e-8, 1.0)
        logits = np.log(p)
        logits[2] += edge
        logits[0] -= edge
        logits[1] += draw_boost
        logits = logits - logits.max()
        ep = np.exp(logits)
        adjusted.append(ep / ep.sum())
    return np.array(adjusted)


def fifa_rank_points_specialist(row):
    a_rank = float(row.get("team_a_fifa_rank", np.nan))
    b_rank = float(row.get("team_b_fifa_rank", np.nan))
    a_points = float(row.get("team_a_fifa_points", np.nan))
    b_points = float(row.get("team_b_fifa_points", np.nan))

    if np.isnan(a_rank) or np.isnan(b_rank):
        return np.array([0.33, 0.34, 0.33])

    rank_edge = (b_rank - a_rank) / 45.0
    points_edge = 0.0 if (np.isnan(a_points) or np.isnan(b_points)) else (a_points - b_points) / 300.0
    edge = np.clip(rank_edge + points_edge, -2.0, 2.0)
    vals = np.array([-edge, 0.35 - abs(edge) * 0.25, edge])
    vals = vals - vals.max()
    return np.exp(vals) / np.exp(vals).sum()


class MatchEdgeV5Predictor:
    def __init__(self, teams_csv: str | Path, features_csv: str | Path, model_path: str | Path):
        self.teams = pd.read_csv(teams_csv)
        self.features = pd.read_csv(features_csv)

        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)

        # Patch saved models for scikit-learn version compatibility.
        self.bundle = patch_sklearn_logistic_compatibility(self.bundle)
        self.bundle = force_patch_logistic_regression(self.bundle)

        self.specialists = self.bundle["specialists"]
        self.feature_groups = self.bundle["feature_groups"]
        self.specialist_order = self.bundle["specialist_order"]
        self.weights = np.asarray(self.bundle["weights"])
        self.temperature = float(self.bundle["temperature"])
        self.metadata = self.bundle["metadata"]

        self.teams["team_id"] = self.teams["team_id"].astype(str).str.upper()
        self.features["team_id"] = self.features["team_id"].astype(str).str.upper()

        self.team_table = self.teams.merge(
            self.features,
            on=["team_id", "team_name"],
            how="left",
            suffixes=("", "_feature")
        )
        self.team_lookup = self.team_table.set_index("team_id", drop=False).to_dict("index")
        self.alias_to_id = self._build_alias_map()

    def _build_alias_map(self) -> Dict[str, str]:
        alias_to_id = {}
        for _, row in self.team_table.iterrows():
            team_id = row["team_id"]
            for value in [team_id, row["team_name"], row.get("common_name", row["team_name"])]:
                alias_to_id[clean_text(value)] = team_id

        for alias, team_id in ALIASES.items():
            if team_id in self.team_lookup:
                alias_to_id[clean_text(alias)] = team_id

        return alias_to_id

    def valid_teams(self) -> pd.DataFrame:
        columns = ["team_id", "team_name", "group_code", "confederation", "fifa_rank", "fifa_points", "current_elo"]
        return self.team_table[columns].sort_values(["group_code", "team_name"])

    def resolve_team(self, team_name: str) -> str:
        key = clean_text(team_name)
        if key in self.alias_to_id:
            return self.alias_to_id[key]

        valid = ", ".join(self.valid_teams()["team_name"].tolist())
        raise ValueError(
            f"'{team_name}' is not one of the 48 World Cup teams in this model. "
            f"Valid teams are: {valid}"
        )

    def _host_advantage(self, team: Dict[str, Any], venue_country: str) -> int:
        return int(
            venue_country in HOST_COUNTRIES
            and int(team.get("is_host", 0)) == 1
            and str(team.get("host_country", "")).strip() == venue_country
        )

    def _feature_row(self, team_a_id: str, team_b_id: str, venue_country: str, importance: float) -> pd.DataFrame:
        a = self.team_lookup[team_a_id]
        b = self.team_lookup[team_b_id]

        a_host = self._host_advantage(a, venue_country)
        b_host = self._host_advantage(b, venue_country)

        row = {
            "neutral_flag": 0 if (a_host or b_host) else 1,
            "importance": float(importance),
            "team_a_pre_elo": float(a["current_elo"]),
            "team_b_pre_elo": float(b["current_elo"]),
            "pre_elo_difference": float(a["current_elo"]) - float(b["current_elo"]),
            "team_a_host_advantage": a_host,
            "team_b_host_advantage": b_host,
        }

        stats = [
            "matches_last_5", "wins_last_5", "points_avg_last_5",
            "goal_diff_avg_last_5", "goals_for_avg_last_10",
            "goals_against_avg_last_10", "goal_diff_avg_last_10",
            "avg_opponent_elo_last_10", "days_since_last_match",
        ]

        for col in stats:
            row[f"team_a_{col}"] = float(a[col])
            row[f"team_b_{col}"] = float(b[col])

        for col in [
            "wins_last_5", "points_avg_last_5", "goal_diff_avg_last_5",
            "goals_for_avg_last_10", "goals_against_avg_last_10",
            "goal_diff_avg_last_10", "avg_opponent_elo_last_10",
            "days_since_last_match",
        ]:
            row[f"{col}_difference"] = row[f"team_a_{col}"] - row[f"team_b_{col}"]

        return pd.DataFrame([row])

    def _specialist_probability_map(self, features: pd.DataFrame):
        strength_probs = predict_proba_aligned(
            self.specialists["strength_elo_logistic"],
            features[self.feature_groups["strength_elo_logistic"]]
        )
        form_probs = predict_proba_aligned(
            self.specialists["recent_form_random_forest"],
            features[self.feature_groups["recent_form_random_forest"]]
        )

        lambda_a = np.clip(
            self.specialists["poisson_team_a_goals"].predict(features[self.feature_groups["poisson_scoreline_model"]]),
            0.05, 6.0
        )
        lambda_b = np.clip(
            self.specialists["poisson_team_b_goals"].predict(features[self.feature_groups["poisson_scoreline_model"]]),
            0.05, 6.0
        )
        poisson_probs = poisson_1x2_from_lambdas(lambda_a, lambda_b)

        context_probs = predict_proba_aligned(
            self.specialists["context_xgboost"],
            features[self.feature_groups["context_xgboost"]]
        )
        full_xgb_probs = predict_proba_aligned(
            self.specialists["full_xgboost"],
            features[self.feature_groups["full_xgboost"]]
        )

        draw_raw = self.specialists["draw_risk_classifier"].predict_proba(
            features[self.feature_groups["draw_risk_classifier"]]
        )
        draw_prob = draw_raw[:, 1] if draw_raw.shape[1] > 1 else np.repeat(0.25, len(features))
        draw_probs = draw_to_1x2_probs(draw_prob, full_xgb_probs)

        return {
            "strength_elo_logistic": strength_probs,
            "recent_form_random_forest": form_probs,
            "poisson_scoreline_model": poisson_probs,
            "context_xgboost": context_probs,
            "full_xgboost": full_xgb_probs,
            "draw_risk_classifier": draw_probs,
        }, lambda_a, lambda_b

    def _blend(self, prob_map):
        probs = np.zeros_like(next(iter(prob_map.values())))
        for name, weight in zip(self.specialist_order, self.weights):
            probs += float(weight) * prob_map[name]
        return temperature_scale_probs(probs, self.temperature)

    def predict(
        self,
        team_a: str,
        team_b: str,
        venue_country: str = "Other",
        importance: float = 60.0,
        team_a_rest_days: float | None = None,
        team_b_rest_days: float | None = None,
        team_a_travel_km: float | None = None,
        team_b_travel_km: float | None = None,
        temperature_c: float = 22.0,
        humidity_pct: float = 50.0,
        apply_context_adjustment: bool = True,
    ) -> Dict[str, Any]:
        team_a_id = self.resolve_team(team_a)
        team_b_id = self.resolve_team(team_b)

        if team_a_id == team_b_id:
            raise ValueError("Team A and Team B must be different teams.")

        if venue_country not in {"Other", "Mexico", "Canada", "United States"}:
            venue_country = "Other"

        features = self._feature_row(team_a_id, team_b_id, venue_country, importance)
        prob_map, lambda_a, lambda_b = self._specialist_probability_map(features)
        final_probs = self._blend(prob_map)

        rest_diff = 0.0
        if team_a_rest_days is not None and team_b_rest_days is not None:
            rest_diff = float(team_a_rest_days) - float(team_b_rest_days)

        travel_diff = 0.0
        if team_a_travel_km is not None and team_b_travel_km is not None:
            travel_diff = float(team_a_travel_km) - float(team_b_travel_km)

        if apply_context_adjustment:
            final_probs = context_adjustment_probs(
                final_probs,
                rest_diff=rest_diff,
                travel_diff_km=travel_diff,
                temperature_c=float(temperature_c),
                humidity_pct=float(humidity_pct),
            )

        team_b_win, draw, team_a_win = final_probs[0]

        a = self.team_lookup[team_a_id]
        b = self.team_lookup[team_b_id]

        fifa_row = {
            "team_a_fifa_rank": a.get("fifa_rank", np.nan),
            "team_b_fifa_rank": b.get("fifa_rank", np.nan),
            "team_a_fifa_points": a.get("fifa_points", np.nan),
            "team_b_fifa_points": b.get("fifa_points", np.nan),
        }
        fifa_probs = fifa_rank_points_specialist(fifa_row)

        labels = {
            f"{a['team_name']} win": team_a_win,
            "Draw": draw,
            f"{b['team_name']} win": team_b_win,
        }

        breakdown = {}
        for name, probs in prob_map.items():
            breakdown[name] = {
                "team_b_win_probability": round(float(probs[0, 0]), 4),
                "draw_probability": round(float(probs[0, 1]), 4),
                "team_a_win_probability": round(float(probs[0, 2]), 4),
                "blend_weight": round(float(self.weights[list(self.specialist_order).index(name)]), 4),
            }

        breakdown["fifa_rank_points_specialist_untrained"] = {
            "team_b_win_probability": round(float(fifa_probs[0]), 4),
            "draw_probability": round(float(fifa_probs[1]), 4),
            "team_a_win_probability": round(float(fifa_probs[2]), 4),
            "blend_weight": 0.0,
        }

        return {
            "team_a_id": team_a_id,
            "team_a_name": a["team_name"],
            "team_b_id": team_b_id,
            "team_b_name": b["team_name"],
            "venue_country": venue_country,
            "team_a_win_probability": round(float(team_a_win), 4),
            "draw_probability": round(float(draw), 4),
            "team_b_win_probability": round(float(team_b_win), 4),
            "predicted_result": max(labels, key=labels.get),
            "team_a_expected_goals": round(float(lambda_a[0]), 2),
            "team_b_expected_goals": round(float(lambda_b[0]), 2),
            "team_a_current_elo": round(float(a["current_elo"]), 2),
            "team_b_current_elo": round(float(b["current_elo"]), 2),
            "elo_difference": round(float(a["current_elo"]) - float(b["current_elo"]), 2),
            "team_a_fifa_rank": None if pd.isna(a.get("fifa_rank", np.nan)) else int(a.get("fifa_rank")),
            "team_b_fifa_rank": None if pd.isna(b.get("fifa_rank", np.nan)) else int(b.get("fifa_rank")),
            "team_a_fifa_points": None if pd.isna(a.get("fifa_points", np.nan)) else round(float(a.get("fifa_points")), 2),
            "team_b_fifa_points": None if pd.isna(b.get("fifa_points", np.nan)) else round(float(b.get("fifa_points")), 2),
            "team_a_points_avg_last_5": round(float(a["points_avg_last_5"]), 2),
            "team_b_points_avg_last_5": round(float(b["points_avg_last_5"]), 2),
            "team_a_goal_diff_avg_last_10": round(float(a["goal_diff_avg_last_10"]), 2),
            "team_b_goal_diff_avg_last_10": round(float(b["goal_diff_avg_last_10"]), 2),
            "team_a_host_advantage": self._host_advantage(a, venue_country),
            "team_b_host_advantage": self._host_advantage(b, venue_country),
            "rest_days_difference": round(rest_diff, 2),
            "travel_km_difference": round(travel_diff, 2),
            "temperature_c": float(temperature_c),
            "humidity_pct": float(humidity_pct),
            "calibration_temperature": self.temperature,
            "specialist_breakdown": breakdown,
            "model_version": self.bundle["model_version"],
        }


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Predict any 2026 World Cup team matchup with Match Edge AI V5.")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--venue-country", default="Other", choices=["Other", "Mexico", "Canada", "United States"])
    parser.add_argument("--importance", type=float, default=60.0)
    parser.add_argument("--team-a-rest-days", type=float, default=None)
    parser.add_argument("--team-b-rest-days", type=float, default=None)
    parser.add_argument("--team-a-travel-km", type=float, default=None)
    parser.add_argument("--team-b-travel-km", type=float, default=None)
    parser.add_argument("--temperature-c", type=float, default=22.0)
    parser.add_argument("--humidity-pct", type=float, default=50.0)
    args = parser.parse_args()

    predictor = MatchEdgeV5Predictor(
        teams_csv=root / "data/processed/wc2026_teams_model_input.csv",
        features_csv=root / "data/processed/team_form_elo_latest.csv",
        model_path=root / "models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl",
    )

    result = predictor.predict(
        args.team_a,
        args.team_b,
        venue_country=args.venue_country,
        importance=args.importance,
        team_a_rest_days=args.team_a_rest_days,
        team_b_rest_days=args.team_b_rest_days,
        team_a_travel_km=args.team_a_travel_km,
        team_b_travel_km=args.team_b_travel_km,
        temperature_c=args.temperature_c,
        humidity_pct=args.humidity_pct,
    )

    print("\nMatch Edge AI V5")
    print("-" * 70)
    print(f"{result['team_a_name']} win: {result['team_a_win_probability']:.1%}")
    print(f"Draw: {result['draw_probability']:.1%}")
    print(f"{result['team_b_name']} win: {result['team_b_win_probability']:.1%}")
    print(f"Predicted result: {result['predicted_result']}")
    print(f"Expected score: {result['team_a_name']} {result['team_a_expected_goals']} - {result['team_b_expected_goals']} {result['team_b_name']}")

    print("\nMain signals")
    print(f"Elo difference: {result['elo_difference']}")
    print(f"FIFA rank: {result['team_a_name']} {result['team_a_fifa_rank']} vs {result['team_b_name']} {result['team_b_fifa_rank']}")
    print(f"Rest days difference: {result['rest_days_difference']}")
    print(f"Travel km difference: {result['travel_km_difference']}")

    print("\nSpecialist breakdown")
    for model_name, probs in result["specialist_breakdown"].items():
        print(f"{model_name}: Team A {probs['team_a_win_probability']:.1%}, "
              f"Draw {probs['draw_probability']:.1%}, "
              f"Team B {probs['team_b_win_probability']:.1%}, "
              f"Weight {probs['blend_weight']:.2f}")


if __name__ == "__main__":
    main()
