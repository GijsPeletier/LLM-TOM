"""Analyse panel experiment results.

Usage:
    python analyze_panel.py <results_dir>
    python analyze_panel.py results/panel_results/openrouter/seed_6
"""

import json
import sys
from pathlib import Path


def _game_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_games(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("game_*.json"), key=_game_number)
    games: list[dict] = []
    for path in paths:
        with open(path) as f:
            g = json.load(f)
        g["meta"]["_file"] = path.name
        games.append(g)
    return games


def player_label(g: dict, idx: int) -> str:
    return "Human" if g["players"][idx]["type"] == "human" else "LLM"


def summarize(results_dir: Path) -> None:
    games = load_games(results_dir)
    if not games:
        print("No game JSON files found.")
        return

    human_deltas: list[float] = []
    llm_deltas: list[float] = []
    per_game: list[dict] = []

    for g in games:
        p0_type = g["players"][0]["type"]
        p1_type = g["players"][1]["type"]
        human_idx = 0 if p0_type == "human" else 1
        llm_idx = 1 - human_idx

        outcome = g["outcome"]
        deltas = outcome["deltas"]
        final_utils = outcome["final_utilities"]
        human_delta = deltas[human_idx]
        llm_delta = deltas[llm_idx]
        human_deltas.append(human_delta)
        llm_deltas.append(llm_delta)

        human_reached = final_utils[human_idx] >= 800
        llm_reached = final_utils[llm_idx] >= 800

        filename = Path(g["meta"].get("_file", "")).name
        per_game.append(
            {
                "game": g["meta"]["board_index"],
                "config": g["meta"]["config"],
                "filename": filename,
                "result": outcome["result"],
                "rounds": outcome["rounds"],
                "agreement": outcome["agreement"],
                "human_delta": human_delta,
                "llm_delta": llm_delta,
                "human_reached": human_reached,
                "llm_reached": llm_reached,
            }
        )

    header = (
        f"{'Game':>5}  {'Config':>7}  {'Result':>10}  {'Rounds':>7}  "
        f"{'Agreement':>10}  {'Human Delta':>12}  {'LLM Delta':>12}  "
        f"{'Human Goal':>11}  {'LLM Goal':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in per_game:
        print(
            f"{row['game']:>5}  {row['config']:>7}  {row['result']:>10}  "
            f"{row['rounds']:>7}  {str(row['agreement']):>10}  "
            f"{row['human_delta']:>12.1f}  {row['llm_delta']:>12.1f}  "
            f"{'YES' if row['human_reached'] else 'no':>11}  "
            f"{'YES' if row['llm_reached'] else 'no':>9}"
        )

    print()
    print("--- Summary ---")
    print(f"Games: {len(games)}")
    agreements = sum(1 for r in per_game if r["agreement"])
    print(f"Agreements: {agreements}/{len(games)}")
    human_goals = sum(1 for r in per_game if r["human_reached"])
    llm_goals = sum(1 for r in per_game if r["llm_reached"])
    print(f"Human reached goal: {human_goals}/{len(games)}")
    print(f"LLM   reached goal: {llm_goals}/{len(games)}")

    def stats(vals: list[float], label: str) -> None:
        mean = sum(vals) / len(vals)
        print(f"{label}  mean: {mean:+.1f}  total: {sum(vals):+.1f}")

    stats(human_deltas, "Human")
    stats(llm_deltas, "LLM   ")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_panel.py <results_dir>")
        print(
            "Example: python analyze_panel.py results/panel_results/openrouter/seed_6"
        )
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}")
        sys.exit(1)

    print(f"Results dir: {results_dir}")
    print()
    summarize(results_dir)


if __name__ == "__main__":
    main()
