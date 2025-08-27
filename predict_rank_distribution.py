# -*- coding: utf-8 -*-
"""
predict_rank_distribution.py
============================
This module contains functions to predict final rank distribution for teams
based on Elo ratings and Monte Carlo simulations.

Usage:
------
python predict_rank_distribution.py --json matches.json --season 2024-25 --current_round 20 --n_sims 20000

Outputs:
--------
- CSV/JSON: rank_distribution.csv / .json (teams x rank probability, ranks are 1..N)
"""

import argparse
import json
import math
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def parse_int_or_none(x):
    if x is None: 
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        try: 
            return int(x)
        except: 
            return None
    if isinstance(x, str):
        xs = x.strip()
        if xs == "": 
            return None
        try: 
            return int(xs)
        except: 
            return None
    return None

def load_matches_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for key, matches in data.items():
        import re
        nums = re.findall(r'\d+', key)
        if not nums:
            raise ValueError(f"Cannot infer round number from key: {key}")
        round_num = int(nums[-1])
        for m in matches:
            home = m.get("team_A_name") or m.get("home") or m.get("teamA") or m.get("team_a")
            away = m.get("team_B_name") or m.get("away") or m.get("teamB") or m.get("team_b")
            sA = parse_int_or_none(m.get("score_A") if "score_A" in m else m.get("home_goals"))
            sB = parse_int_or_none(m.get("score_B") if "score_B" in m else m.get("away_goals"))
            date = m.get("date", "")
            if not home or not away:
                raise ValueError(f"Missing team names in {key}: {m}")
            rows.append({
                "season": None,
                "round": round_num,
                "date": date,
                "home": str(home),
                "away": str(away),
                "home_goals": sA,
                "away_goals": sB,
            })
    df = pd.DataFrame(rows)
    df.sort_values(by=["round","date"], inplace=True, kind="mergesort" )
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
    if hg > ag: 
        Sa, Sb = 1.0, 0.0
    elif hg < ag: 
        Sa, Sb = 0.0, 1.0
    else: 
        Sa, Sb = 0.5, 0.5
    elo[h] = elo[h] + k*(Sa - Ea)
    elo[a] = elo[a] + k*(Sb - Eb)
    return elo

def draw_probability(ediff: float, base_draw: float=0.25, decay: float=400.0) -> float:
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

def last5_form_zscores(df_hist: pd.DataFrame, teams: List[str]) -> Dict[str, float]:
    ppg5 = {}
    for t in teams:
        rows = df_hist[(df_hist["home"]==t) | (df_hist["away"]==t)].sort_values(by=["round","date"])
        results = []
        for _, r in rows.tail(5).iterrows():
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            if r["home"]==t:
                if hg>ag: results.append(3)
                elif hg<ag: results.append(0)
                else: results.append(1)
            else:
                if ag>hg: results.append(3)
                elif ag<hg: results.append(0)
                else: results.append(1)
        if results:
            ppg5[t] = sum(results)/len(results)
        else:
            ppg5[t] = None
    vals = [v for v in ppg5.values() if v is not None]
    if not vals or len(set(vals))==1:
        return {t: 0.0 for t in teams}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals)>1 else 1.0
    z = {t: ((ppg5[t]-mean)/std if ppg5[t] is not None else 0.0) for t in teams}
    return z

def simulate_season(df_future: pd.DataFrame, teams: List[str],
                    relegation_slots: int, elo_init: Dict[str,float],
                    hist_points: Dict[str,int], hist_gd: Dict[str,int],
                    n_sims: int=200, k_elo: float=20.0, home_adv: float=60.0,
                    base_draw: float=0.25, decay: float=400.0,
                    form_bonus: Dict[str,float]=None,
                    progress_every: int=0, progress_label: str="", verbose: bool=False, rng=None):
    if rng is None:
        rng = RNG
    relegated_counts = {t:0 for t in teams}
    N = len(teams)
    rank_counts = {t: np.zeros(N, dtype=np.int64) for t in teams}
    step_print = max(1, progress_every) if progress_every and progress_every>0 else None

    for sim in range(1, n_sims+1):
        elo = elo_init.copy()
        pts = hist_points.copy()
        gd = hist_gd.copy()

        for _, r in df_future.iterrows():
            h, a = r["home"], r["away"]
            fb_h = form_bonus.get(h, 0.0) if form_bonus else 0.0
            fb_a = form_bonus.get(a, 0.0) if form_bonus else 0.0
            ediff = (elo[h] + home_adv + fb_h) - (elo[a] + fb_a)
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
        table = table.reset_index(drop=True)
        for idx, row in table.iterrows():
            rank = idx+1
            rank_counts[row["team"]][rank-1] += 1

        if step_print and verbose and (sim % step_print == 0 or sim == n_sims):
            label = f"[{progress_label}] " if progress_label else ""
            percent = (sim / n_sims) * 100
            print(f"{label}sim {sim}/{n_sims} ({percent:.1f}%)")

        # 如果没有设置 progress_every 但 verbose 为 True，则每 10% 进度显示一次
        elif verbose and progress_every == 0 and (sim % max(1, n_sims // 10) == 0 or sim == n_sims):
            label = f"[{progress_label}] " if progress_label else ""
            percent = (sim / n_sims) * 100
            print(f"{label}sim {sim}/{n_sims} ({percent:.1f}%)")

    relegation_prob = {t: relegated_counts[t]/n_sims for t in teams}
    rank_prob = {}
    for t, arr in rank_counts.items():
        rank_prob[t] = (arr / float(n_sims)).tolist()
    rank_df = pd.DataFrame(rank_prob).T
    rank_df.columns = [f"rank_{i}" for i in range(1, N+1)]
    rank_df.index.name = "team"
    rank_df.reset_index(inplace=True)

    return relegation_prob, rank_df

def predict_rank_distribution_from_json(json_path: str, season: str, current_round: int,
                                       relegation_slots: int=3, n_sims: int=20000,
                                       k_elo: float=20.0, home_adv: float=60.0,
                                       base_draw: float=0.25, decay: float=400.0,
                                       form_coeff: float=40.0, progress_every: int=0, verbose: bool=False):
    df = load_matches_json(json_path)
    df["season"] = season
    df.sort_values(by=["round","date"], inplace=True)
    df_hist = df[(df["round"]<=current_round) & df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df_future = df[(df["round"]>current_round) | df["home_goals"].isna() | df["away_goals"].isna()].copy()
    df_future = df_future[~df_future.index.isin(df_hist.index)].copy()

    teams = infer_teams(df)
    elo = initialize_elo(teams, base=1500.0)
    for _, r in df_hist.iterrows():
        elo = update_elo(elo, r, k=k_elo, home_adv=home_adv)
    elo_init = elo.copy()

    z = last5_form_zscores(df_hist, teams)
    form_bonus = {t: form_coeff * z[t] for t in teams}

    # 预计算历史积分和净胜球
    hist_table = compute_standings(df_hist, teams)
    hist_points = dict(zip(hist_table["team"], hist_table["points"]))
    hist_gd = dict(zip(hist_table["team"], hist_table["gd"]))
    
    _, rank_df = simulate_season(
        df_future, teams, relegation_slots, elo_init, hist_points, hist_gd,
        n_sims=n_sims, k_elo=k_elo, home_adv=home_adv,
        base_draw=base_draw, decay=decay, form_bonus=form_bonus,
        progress_every=progress_every, progress_label="baseline", verbose=verbose, rng=RNG
    )
    
    return rank_df, elo_init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to matches.json")
    ap.add_argument("--season", required=True, help="Season label, e.g., 2024-25")
    ap.add_argument("--current_round", type=int, required=True, help="Current finished round number (e.g., 20)")
    ap.add_argument("--relegation_slots", type=int, default=3)
    ap.add_argument("--n_sims", type=int, default=20000)
    ap.add_argument("--k_elo", type=float, default=20.0)
    ap.add_argument("--home_adv", type=float, default=60.0)
    ap.add_argument("--base_draw", type=float, default=0.25)
    ap.add_argument("--decay", type=float, default=400.0)
    ap.add_argument("--form_coeff", type=float, default=40.0, help="Elo points per 1 std dev of last-5 PPG")
    ap.add_argument("--progress_every", type=int, default=0, help="Print progress every N simulations (0=off, suggested ~ n_sims/10)")
    ap.add_argument("--verbose", type=int, default=1, help="1=show baseline progress; 0=quiet")
    args = ap.parse_args()

    rank_df, elo_scores = predict_rank_distribution_from_json(
        json_path=args.json,
        season=args.season,
        current_round=args.current_round,
        relegation_slots=args.relegation_slots,
        n_sims=args.n_sims,
        k_elo=args.k_elo,
        home_adv=args.home_adv,
        base_draw=args.base_draw,
        decay=args.decay,
        form_coeff=args.form_coeff,
        progress_every=args.progress_every,
        verbose=bool(args.verbose),
    )

    rank_df.to_csv("rank_distribution.csv", index=False, encoding="utf-8-sig")
    rank_df.to_json("rank_distribution.json", orient="records", force_ascii=False, indent=2)

    # 输出Elo分数
    elo_df = pd.DataFrame(list(elo_scores.items()), columns=["team", "elo"])
    elo_df.sort_values(by="elo", ascending=False, inplace=True)
    elo_df.to_csv("elo_scores.csv", index=False, encoding="utf-8-sig")
    elo_df.to_json("elo_scores.json", orient="records", force_ascii=False, indent=2)

    print(f"\nSaved: rank_distribution.csv\nSaved: rank_distribution.json\nSaved: elo_scores.csv\nSaved: elo_scores.json")

if __name__ == "__main__":
    main()