
# -*- coding: utf-8 -*-
"""
match_importance_json.py
========================
Schedule-aware "pivotal match" analysis directly from a JSON file like:

{
  "gameweek_1": [
    {"team_A_name":"成都蓉城","team_B_name":"武汉三镇","score_A":"1","score_B":"0"},
    ...
  ],
  "gameweek_2": [...],
  ...
}

Usage:
------
python match_importance_json.py \
  --json matches.json \
  --season 2024-25 \
  --current_round 20 \
  --team "青岛海牛" \
  --relegation_slots 3 \
  --n_sims 20000

Outputs:
--------
- Prints baseline relegation probability for target team
- Prints target team's remaining fixtures ranked by Expected impact and Swing
- Saves CSV: match_importance_<team>.csv
- Saves JSON: match_importance_<team>.json

Notes:
------
- A match is treated as "played" if both score_A and score_B are present and parseable as integers.
- Future fixtures should have score fields missing/empty. They will be simulated.
- Dates are optional. If you have per-match dates, add a "date" key in each item (ISO string); otherwise they're blank.
"""

import argparse
import json
import math
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def parse_int_or_none(x):
    if x is None: return None
    if isinstance(x, (int, float)) and not math.isnan(x):
        try: return int(x)
        except: return None
    if isinstance(x, str):
        xs = x.strip()
        if xs == "": return None
        try: return int(xs)
        except: return None
    return None

def load_matches_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for key, matches in data.items():
        # extract round number, e.g. "gameweek_1" -> 1
        # tolerate other patterns by picking the last integer found
        import re
        nums = re.findall(r'\d+', key)
        if not nums:
            raise ValueError(f"Cannot infer round number from key: {key}")
        round_num = int(nums[-1])
        for m in matches:
            # support both the provided schema and a few common variants
            home = m.get("team_A_name") or m.get("home") or m.get("teamA") or m.get("team_a")
            away = m.get("team_B_name") or m.get("away") or m.get("teamB") or m.get("team_b")
            sA = parse_int_or_none(m.get("score_A") if "score_A" in m else m.get("home_goals"))
            sB = parse_int_or_none(m.get("score_B") if "score_B" in m else m.get("away_goals"))
            date = m.get("date", "")
            if not home or not away:
                raise ValueError(f"Missing team names in {key}: {m}")
            rows.append({
                "season": None,  # fill later
                "round": round_num,
                "date": date,
                "home": str(home),
                "away": str(away),
                "home_goals": sA,
                "away_goals": sB,
            })
    df = pd.DataFrame(rows)
    # sort numeric round, then date (if any)
    df.sort_values(by=["round","date"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def infer_teams(df: pd.DataFrame) -> List[str]:
    teams = pd.unique(pd.concat([df["home"], df["away"]]))
    return sorted([t for t in teams if pd.notna(t)])

def compute_standings(points_rows: pd.DataFrame, teams: List[str]) -> pd.DataFrame:
    pts = {t:0 for t in teams}
    gd = {t:0 for t in teams}
    for _, r in points_rows.iterrows():
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        h, a = r["home"], r["away"]
        if hg > ag:
            pts[h]+=3
        elif hg < ag:
            pts[a]+=3
        else:
            pts[h]+=1; pts[a]+=1
        gd[h] += (hg-ag); gd[a] += (ag-hg)
    table = pd.DataFrame({
        "team": teams,
        "points": [pts[t] for t in teams],
        "gd": [gd[t] for t in teams],
    })
    table.sort_values(by=["points","gd","team"], ascending=[False,False,True], inplace=True, kind="mergesort")
    table.reset_index(drop=True, inplace=True)
    return table

def initialize_elo(teams: List[str], base: float = 1500.0) -> Dict[str, float]:
    return {t: float(base) for t in teams}

def update_elo(elo: Dict[str,float], row: Dict[str,Any], k: float=20.0, home_adv: float=60.0) -> Dict[str,float]:
    h, a = row["home"], row["away"]
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    Ra = elo[h] + home_adv
    Rb = elo[a]
    Ea = 1.0/(1.0 + 10**((Rb - Ra)/400.0))
    Eb = 1.0 - Ea
    if hg > ag: Sa, Sb = 1.0, 0.0
    elif hg < ag: Sa, Sb = 0.0, 1.0
    else: Sa, Sb = 0.5, 0.5
    elo[h] = elo[h] + k*(Sa - Ea)
    elo[a] = elo[a] + k*(Sb - Eb)
    return elo

def draw_probability(ediff: float, base_draw: float=0.25, decay: float=400.0) -> float:
    # higher draw chance near equal strength; clip to sensible bounds
    p = base_draw * math.e**(-abs(ediff)/decay)
    return float(max(0.05, min(0.35, p)))

def wl_probabilities(ediff: float, base_draw: float=0.25, decay: float=400.0) -> Tuple[float,float,float]:
    p_home_no_draw = 1.0/(1.0 + 10**(-ediff/400.0))
    p_away_no_draw = 1.0 - p_home_no_draw
    p_d = draw_probability(ediff, base_draw, decay)
    p_h = (1 - p_d) * p_home_no_draw
    p_a = (1 - p_d) * p_away_no_draw
    total = p_h + p_d + p_a
    return (p_h/total, p_d/total, p_a/total)

def simulate_season(df_hist: pd.DataFrame, df_future: pd.DataFrame, teams: List[str],
                    relegation_slots: int, elo_init: Dict[str,float],
                    n_sims: int=10000, k_elo: float=20.0, home_adv: float=60.0,
                    base_draw: float=0.25, decay: float=400.0, rng=None) -> Dict[str,float]:
    if rng is None:
        rng = RNG
    relegated_counts = {t:0 for t in teams}
    for _ in range(n_sims):
        elo = elo_init.copy()
        for _, r in df_hist.iterrows():
            elo = update_elo(elo, r, k=k_elo, home_adv=home_adv)
        hist_table = compute_standings(df_hist, teams)
        pts = dict(zip(hist_table["team"], hist_table["points"]))
        gd = dict(zip(hist_table["team"], hist_table["gd"]))
        for _, r in df_future.iterrows():
            h, a = r["home"], r["away"]
            ediff = (elo[h] + home_adv) - elo[a]
            p_h, p_d, p_a = wl_probabilities(ediff, base_draw, decay)
            outcome = rng.choice(["H","D","A"], p=[p_h, p_d, p_a])
            if outcome == "H":
                pts[h]+=3; gd[h]+=1; gd[a]-=1
                elo = update_elo(elo, {"home":h,"away":a,"home_goals":1,"away_goals":0}, k=k_elo, home_adv=home_adv)
            elif outcome == "A":
                pts[a]+=3; gd[a]+=1; gd[h]-=1
                elo = update_elo(elo, {"home":h,"away":a,"home_goals":0,"away_goals":1}, k=k_elo, home_adv=home_adv)
            else:
                pts[h]+=1; pts[a]+=1
                elo = update_elo(elo, {"home":h,"away":a,"home_goals":1,"away_goals":1}, k=k_elo, home_adv=home_adv)
        table = pd.DataFrame({"team":teams, "points":[pts[t] for t in teams], "gd":[gd[t] for t in teams]})
        table.sort_values(by=["points","gd","team"], ascending=[False,False,True], inplace=True, kind="mergesort")
        bottom = table.tail(relegation_slots)["team"].tolist()
        for t in bottom:
            relegated_counts[t]+=1
    return {t: relegated_counts[t]/n_sims for t in teams}

def match_importance_from_json(json_path: str, season: str, current_round: int, target_team: str,
                               relegation_slots: int=3, n_sims: int=20000,
                               k_elo: float=20.0, home_adv: float=60.0,
                               base_draw: float=0.25, decay: float=400.0) -> Tuple[pd.DataFrame, float]:
    df = load_matches_json(json_path)
    df["season"] = season
    # split history vs future by round and presence of goals
    df.sort_values(by=["round","date"], inplace=True)
    # History: <= current_round AND scores present
    df_hist = df[(df["round"]<=current_round) & df["home_goals"].notna() & df["away_goals"].notna()].copy()
    # Future: > current_round OR scores missing in current_round block
    df_future = df[(df["round"]>current_round) | df["home_goals"].isna() | df["away_goals"].isna()].copy()
    # But ensure the same match isn't in both (if current round has some played and some unplayed)
    df_future = df_future[~df_future.index.isin(df_hist.index)].copy()

    teams = infer_teams(df)
    # Seed Elo with history
    elo = initialize_elo(teams, base=1500.0)
    for _, r in df_hist.iterrows():
        elo = update_elo(elo, r, k=k_elo, home_adv=home_adv)
    elo_init = elo.copy()

    base_prob = simulate_season(df_hist, df_future, teams, relegation_slots, elo_init,
                                n_sims=n_sims, k_elo=k_elo, home_adv=home_adv,
                                base_draw=base_draw, decay=decay)
    base_p_target = base_prob.get(target_team)
    if base_p_target is None:
        raise ValueError(f"Target team '{target_team}' not found in teams: {teams}")

    # Remaining matches of target team
    mask_target = (df_future["home"]==target_team) | (df_future["away"]==target_team)
    rem = df_future[mask_target].copy()

    rows = []
    for _, m in rem.iterrows():
        h, a = m["home"], m["away"]
        ediff = (elo_init[h] + home_adv) - elo_init[a]
        p_h, p_d, p_a = wl_probabilities(ediff, base_draw, decay)
        impacts = {}
        for forced in ["H","D","A"]:
            df_future_forced = df_future.copy()
            # pick the exact row to force by round+teams (assuming unique per round)
            key = (df_future_forced["home"]==h) & (df_future_forced["away"]==a) & (df_future_forced["round"]==m["round"])
            fixed_row = df_future_forced[key].copy()
            if forced=="H": hg, ag = 1, 0
            elif forced=="A": hg, ag = 0, 1
            else: hg, ag = 1, 1
            fixed_row.loc[:, "home_goals"] = hg
            fixed_row.loc[:, "away_goals"] = ag
            new_hist = pd.concat([df_hist, fixed_row], ignore_index=True)
            new_future = df_future_forced[~key].copy()
            prob = simulate_season(new_hist, new_future, teams, relegation_slots, elo_init,
                                   n_sims=n_sims, k_elo=k_elo, home_adv=home_adv,
                                   base_draw=base_draw, decay=decay)
            impacts[forced] = prob[target_team] - base_p_target

        if h==target_team:
            pW, pD, pL = p_h, p_d, p_a
            opp = a
            hoa = "Home"
            delta_win = impacts["H"]; delta_loss = impacts["A"]
        else:
            pW, pD, pL = p_a, p_d, p_h
            opp = h
            hoa = "Away"
            delta_win = impacts["A"]; delta_loss = impacts["H"]

        expected_impact = pW*abs(delta_win) + pD*abs(impacts["D"]) + pL*abs(delta_loss)
        swing = max(impacts.values()) - min(impacts.values())

        rows.append({
            "round": int(m["round"]),
            "date": m["date"],
            "opponent": opp,
            "home_away": hoa,
            "p_win": round(pW, 4),
            "p_draw": round(pD, 4),
            "p_loss": round(pL, 4),
            "delta_if_win": round(delta_win, 5),
            "delta_if_draw": round(impacts["D"], 5),
            "delta_if_loss": round(delta_loss, 5),
            "expected_impact": round(expected_impact, 5),
            "swing": round(swing, 5),
        })

    out = pd.DataFrame(rows).sort_values(by=["expected_impact","swing"], ascending=[False, False]).reset_index(drop=True)
    return out, base_p_target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to matches.json")
    ap.add_argument("--season", required=True, help="Season label, e.g., 2024-25")
    ap.add_argument("--current_round", type=int, required=True, help="Current finished round number (e.g., 20)")
    ap.add_argument("--team", required=True, help="Target team name exactly as in JSON")
    ap.add_argument("--relegation_slots", type=int, default=3)
    ap.add_argument("--n_sims", type=int, default=20000)
    ap.add_argument("--k_elo", type=float, default=20.0)
    ap.add_argument("--home_adv", type=float, default=60.0)
    ap.add_argument("--base_draw", type=float, default=0.25)
    ap.add_argument("--decay", type=float, default=400.0)
    args = ap.parse_args()

    df_out, base_p = match_importance_from_json(
        json_path=args.json,
        season=args.season,
        current_round=args.current_round,
        target_team=args.team,
        relegation_slots=args.relegation_slots,
        n_sims=args.n_sims,
        k_elo=args.k_elo,
        home_adv=args.home_adv,
        base_draw=args.base_draw,
        decay=args.decay,
    )

    print(f"Baseline relegation probability for {args.team}: {base_p:.4f}")
    print(df_out.to_string(index=False))

    out_csv = f"match_importance_{args.team.replace(' ','_')}.csv"
    out_json = f"match_importance_{args.team.replace(' ','_')}.json"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_out.to_json(out_json, orient="records", force_ascii=False, indent=2)
    print(f"\nSaved: {out_csv}\nSaved: {out_json}")

if __name__ == "__main__":
    main()
