"""Analyse panel experiment results across multiple seeds.

Usage:
    python analyze_panel.py openrouter
    python analyze_panel.py openrouter --seeds 6 11 18
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

VALID_GOAL_COORDS = [
    (0, 0),
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 4),
    (3, 0),
    (3, 4),
    (4, 0),
    (4, 1),
    (4, 3),
    (4, 4),
]

ATL_NOTATION_TO_GOAL: dict[str, int] = {
    "TLTL": 0,
    "TLTR": 1,
    "TRTL": 2,
    "TRTR": 3,
    "TLBL": 4,
    "TRBR": 5,
    "BLTL": 6,
    "BRTR": 7,
    "BLBL": 8,
    "BLBR": 9,
    "BRBL": 10,
    "BRBR": 11,
}

CHESS_TO_GOAL: dict[str, int] = {
    "A5": 0,
    "B5": 1,
    "D5": 2,
    "E5": 3,
    "A4": 4,
    "E4": 5,
    "A2": 6,
    "E2": 7,
    "A1": 8,
    "B1": 9,
    "D1": 10,
    "E1": 11,
}

WIN_CATEGORIES = [
    "Neither reached goal",
    "Only Human reached goal",
    "Only LLM reached goal",
    "Both reached goal",
]


def _square_to_goal_indices(square_str: str) -> set[int]:
    result: set[int] = set()
    for token in square_str.strip().split():
        token_upper = token.upper()
        if token_upper in ATL_NOTATION_TO_GOAL:
            result.add(ATL_NOTATION_TO_GOAL[token_upper])
        elif re.match(r"^[A-E][1-5]$", token_upper):
            if token_upper in CHESS_TO_GOAL:
                result.add(CHESS_TO_GOAL[token_upper])
    return result


def _parse_human_square(thoughts: str | None) -> str | None:
    if not thoughts:
        return None
    if " - " not in thoughts:
        return None
    square = thoughts.split(" - ", 1)[0].strip()
    if square == "?":
        return None
    if not square:
        return None
    return square


def _parse_game_file(path: Path) -> dict | None:
    parts = path.stem.split("_")
    try:
        game_idx = int(parts[1])
    except (IndexError, ValueError):
        return None
    board_type = parts[2]
    deception = parts[3] == "dec"
    config = parts[4]

    with open(path) as f:
        g = json.load(f)

    outcome = g.get("outcome")
    if outcome is None or "result" not in outcome:
        return None

    llm_idx = 0 if g["players"][0]["type"] != "human" else 1
    human_idx = 1 - llm_idx

    llm_goal = g["players"][llm_idx]["goal"]
    final_utils = outcome.get("final_utilities")
    rounds = outcome.get("rounds", 0)
    llm_reached = final_utils is not None and final_utils[llm_idx] >= 800 - rounds
    human_reached = final_utils is not None and final_utils[human_idx] >= 800 - rounds

    category: int
    if not human_reached and not llm_reached:
        category = 0
    elif human_reached and not llm_reached:
        category = 1
    elif not human_reached and llm_reached:
        category = 2
    else:
        category = 3

    num_messages = len(g.get("messages", []))
    turns = (num_messages + 1) // 2

    goal_correct = False
    has_prediction = False
    llm_goal_set = {llm_goal}
    for msg in g.get("messages", []):
        if msg["player_id"] != human_idx:
            continue
        square = _parse_human_square(msg.get("thoughts"))
        if square is None:
            continue
        has_prediction = True
        predicted = _square_to_goal_indices(square)
        if predicted & llm_goal_set:
            goal_correct = True

    if not has_prediction:
        goal_correct = False

    return {
        "game_idx": game_idx,
        "board_type": board_type,
        "deception": deception,
        "config": config,
        "category": category,
        "llm_reached": llm_reached,
        "human_reached": human_reached,
        "turns": turns,
        "goal_correct": goal_correct,
    }


def _load_all_games(base_dir: Path, seeds: list[int] | None = None) -> list[dict]:
    all_games: list[dict] = []
    seed_dirs = sorted(base_dir.glob("seed_*"))
    for sd in seed_dirs:
        try:
            seed_num = int(sd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if seeds is not None and seed_num not in seeds:
            continue
        for gf in sorted(sd.glob("game_*.json")):
            parsed = _parse_game_file(gf)
            if parsed is None:
                continue
            parsed["seed"] = seed_num
            all_games.append(parsed)
    return all_games


def _category_breakdown(games: list[dict]) -> list[tuple[int, int, float]]:
    total = len(games)
    if total == 0:
        return [(c, 0, 0.0) for c in range(4)]
    counts = [sum(1 for g in games if g["category"] == c) for c in range(4)]
    return [(c, counts[c], counts[c] / total * 100) for c in range(4)]


def _win_rate_llm(games: list[dict]) -> float:
    if not games:
        return 0.0
    return sum(1 for g in games if g["llm_reached"]) / len(games) * 100


def _goal_accuracy(games: list[dict]) -> float:
    if not games:
        return 0.0
    return sum(1 for g in games if g["goal_correct"]) / len(games) * 100


def _print_category_table(title: str, games: list[dict]) -> None:
    print(f"\n  {title}")
    breakdown = _category_breakdown(games)
    print(f"{'':>28} {'Count':>6} {'%':>8}")
    print(f"  {'-' * 40}")
    for cat, count, pct in breakdown:
        label = WIN_CATEGORIES[cat]
        print(f"  {label:>26}  {count:>5}  {pct:>5.1f}%")
    total = len(games)
    print(f"  {'Total':>26}  {total:>5}")
    llm_wr = _win_rate_llm(games)
    print(
        f"  LLM reached goal: {llm_wr:.1f}% ({sum(1 for g in games if g['llm_reached'])}/{total})"
    )
    acc = _goal_accuracy(games)
    print(
        f"  Goal est. correct: {acc:.1f}% ({sum(1 for g in games if g['goal_correct'])}/{total})"
    )


def _print_summary(all_games: list[dict]) -> None:
    seeds = sorted(set(g["seed"] for g in all_games))

    print("=" * 45)
    print(f"  PANEL ANALYSIS — {len(all_games)} games across {len(seeds)} seeds")
    print("=" * 45)

    _print_category_table("OVERALL", all_games)

    print(f"\n\n\n{'=' * 45}")
    print(" " * 18, "PER SEED")
    print("=" * 45)
    for seed in seeds:
        seed_games = [g for g in all_games if g["seed"] == seed]
        _print_category_table(f"Seed {seed} ({len(seed_games)} games)", seed_games)

    print(f"\n\n\n{'=' * 45}")
    print(" " * 9, "DECEPTION vs NON-DECEPTION")
    print("=" * 45)
    for label, dec_val in [("DECEPTION ON", True), ("DECEPTION OFF", False)]:
        subset = [g for g in all_games if g["deception"] == dec_val]
        _print_category_table(f"{label} ({len(subset)} games)", subset)

    print(f"\n\n\n{'=' * 45}")
    print(" " * 9, "NORMAL vs ADVANTAGE BOARDS")
    print("=" * 45)
    for btype in ("normal", "advantage"):
        subset = [g for g in all_games if g["board_type"] == btype]
        _print_category_table(f"{btype.upper()} ({len(subset)} games)", subset)

    print(f"\n\n\n{'=' * 45}")
    print(" " * 10, "DECEPTION x BOARD TYPE")
    print("=" * 45)
    for btype in ("normal", "advantage"):
        for label, dec_val in [("dec", True), ("nodec", False)]:
            subset = [
                g
                for g in all_games
                if g["board_type"] == btype and g["deception"] == dec_val
            ]
            _print_category_table(f"{btype} / {label} ({len(subset)} games)", subset)


def _build_turn_plot(all_games: list[dict], output_path: str) -> None:
    games_ok = [g for g in all_games if g["llm_reached"] is not None]
    turn_max = max((g["turns"] for g in games_ok), default=4)
    turns = list(range(1, min(turn_max, 6) + 1))
    if turn_max >= 7:
        turns.append(7)
    loss_rates: list[float] = []
    turn_counts: list[int] = []
    for t in turns:
        if t < 7:
            at_turn = [g for g in games_ok if g["turns"] == t]
        else:
            at_turn = [g for g in games_ok if g["turns"] >= 7]
        turn_counts.append(len(at_turn))
        if not at_turn:
            loss_rates.append(0.0)
        else:
            loss_rates.append(
                sum(1 for g in at_turn if not g["llm_reached"]) / len(at_turn)
            )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(turns, loss_rates, color="#d62728", edgecolor="black")
    ax.set_xlabel("Turns")
    ax.set_ylabel("LLM Loss Rate")
    ax.set_title("LLM Loss Rate by Game Length (Turns)")
    ax.set_xticks(turns)
    ax.set_xticklabels([str(t) if t < 7 else "7+" for t in turns])
    ax.set_ylim(0, 1)

    for bar, rate, count in zip(bars, loss_rates, turn_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved bar plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse panel experiment results across seeds."
    )
    parser.add_argument(
        "controller_type", type=str, help="Controller type (e.g. openrouter)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Specific seeds to include (default: all)",
    )
    args = parser.parse_args()

    results_dir = Path("results") / "panel_results" / args.controller_type
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}")
        sys.exit(1)

    print(f"Loading games from {results_dir} ...")
    all_games = _load_all_games(results_dir, seeds=args.seeds)
    if not all_games:
        print("No games found.")
        sys.exit(1)

    _print_summary(all_games)

    plot_path = Path("results") / "images" / "turn_loss_plot.png"
    _build_turn_plot(all_games, str(plot_path))


if __name__ == "__main__":
    main()
