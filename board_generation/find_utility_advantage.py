"""
find_utility_advantage.py

Searches for Colored Trails boards where a higher-order ToM agent
outperforms a lower-order ToM agent by at least a given utility threshold.

Both the baseline ToM level (--tom-low, default 0) and the upgraded
ToM level (--tom-high, default 1) are configurable, so you can search
for boards that differentiate any pair: 0 vs 1, 0 vs 2, 1 vs 2, etc.

Parallelism
-----------
Board evaluation (the slow part) is distributed across multiple CPU
cores using Python's multiprocessing.  Board generation is done
sequentially first (it's fast and needs deterministic seeding), then
all boards are evaluated in parallel.

Use --workers to control the number of parallel processes (default:
all available CPUs).

Methodology
-----------
For every randomly generated board we play four matchups:

    LvL  —  ToM-Low  (initiator) vs ToM-Low  (responder)  [baseline]
    HvL  —  ToM-High (initiator) vs ToM-Low  (responder)
    LvH  —  ToM-Low  (initiator) vs ToM-High (responder)
    HvH  —  ToM-High (initiator) vs ToM-High (responder)

To control for positional (seat) advantage we repeat all four matchups
on a **mirrored** copy of the board where chips and goal locations are
swapped between P0 and P1.  This ensures that each physical position
(chip set + goal) appears once as initiator and once as responder.

The primary metric — avg_self_mirrored — averages the per-seat
advantage of ToM-High over ToM-Low across both orientations
(4 measurements total).  Only boards exceeding the threshold are kept.

Usage
-----
    # Default: ToM-1 vs ToM-0, threshold 100, all CPUs
    python find_utility_advantage.py --threshold 100

    # ToM-2 vs ToM-1 with 8 workers
    python find_utility_advantage.py --tom-low 1 --tom-high 2 --workers 8

    # ToM-2 vs ToM-0, 16 workers, search 500 boards
    python find_utility_advantage.py --tom-low 0 --tom-high 2 --workers 16 --max-search 500

Output
------
    board_generation/tom<high>_vs_tom<low>_advantage_mirrored_<threshold>.json
"""

import random
import json
import sys
import os
import multiprocessing
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.colored_trails import Game_ct
from agents.tom_agent import Agent_ct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_NAMES = ["White", "Black", "Magenta", "Grey", "Yellow"]
MAX_NEGOTIATION_ROUNDS = 10
COST_PER_ROUND = 1
LEARNING_SPEED = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_matchup_result(result):
    """Format a single matchup result as 'D(+x,+y)' or 'F(+x,+y)'."""
    tag = "D" if result["agreed"] else "F"
    return f"{tag}({result['p0_gain']:+.0f},{result['p1_gain']:+.0f})"


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def run_negotiation(game, initiator, responder):
    """Play one full negotiation between two agents on the given game board."""
    initiator_start_utility = game.utility_function[game.locations[0]][
        game.chip_sets[0]
    ]
    responder_start_utility = game.utility_function[game.locations[1]][
        game.chip_sets[1]
    ]

    initiator.init(game)
    responder.init(game)
    agents = [initiator, responder]

    current_player = 0
    offer_on_table = None
    agreement_reached = False

    for round_number in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        if offer_on_table is None:
            offer, _ = agents[0].make_offer()
        else:
            offer, _ = agents[current_player].make_offer(
                game.flip_array[offer_on_table]
            )

        if offer == game.chip_sets[current_player]:
            break

        if offer_on_table is not None and offer == game.flip_array[offer_on_table]:
            agreement_reached = True
            break

        offer_on_table = offer
        current_player = 1 - current_player

    if agreement_reached:
        proposer = 1 - current_player
        final_chips_p0 = (
            offer_on_table if proposer == 0 else game.flip_array[offer_on_table]
        )
        final_chips_p1 = (
            game.flip_array[offer_on_table] if proposer == 0 else offer_on_table
        )
    else:
        final_chips_p0 = game.chip_sets[0]
        final_chips_p1 = game.chip_sets[1]

    negotiation_cost = round_number * COST_PER_ROUND
    initiator_gain = (
        game.utility_function[game.locations[0]][final_chips_p0]
        - negotiation_cost
        - initiator_start_utility
    )
    responder_gain = (
        game.utility_function[game.locations[1]][final_chips_p1]
        - negotiation_cost
        - responder_start_utility
    )

    return initiator_gain, responder_gain, agreement_reached, round_number


# ---------------------------------------------------------------------------
# Single matchup wrapper
# ---------------------------------------------------------------------------


def play_matchup(
    board, chips_p0, chips_p1, goal_p0, goal_p1, tom_level_p0, tom_level_p1
):
    """Set up a fresh game and run one negotiation."""
    game = Game_ct()
    game.load_setting(board, chips_p0[:], chips_p1[:], goal_p0, goal_p1)

    agent_p0 = Agent_ct(tom_level_p0, 0, learning_speed=LEARNING_SPEED)
    agent_p1 = Agent_ct(tom_level_p1, 1, learning_speed=LEARNING_SPEED)

    p0_gain, p1_gain, agreed, rounds = run_negotiation(game, agent_p0, agent_p1)

    return {
        "p0_gain": p0_gain,
        "p1_gain": p1_gain,
        "agreed": agreed,
        "rounds": rounds,
        "total_gain": p0_gain + p1_gain,
    }


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


def compute_advantage(original_results, mirrored_results, label_high, label_low):
    """
    Measure how much better the high-order agent performs than the low-order
    agent, averaged across both seat orientations.
    """
    key_low_v_low = f"{label_low}v{label_low}"
    key_high_v_low = f"{label_high}v{label_low}"
    key_low_v_high = f"{label_low}v{label_high}"

    orig = original_results
    mirr = mirrored_results

    orig_initiator_adv = (
        orig[key_high_v_low]["p0_gain"] - orig[key_low_v_low]["p0_gain"]
    )
    orig_responder_adv = (
        orig[key_low_v_high]["p1_gain"] - orig[key_low_v_low]["p1_gain"]
    )

    mirr_initiator_adv = (
        mirr[key_high_v_low]["p0_gain"] - mirr[key_low_v_low]["p0_gain"]
    )
    mirr_responder_adv = (
        mirr[key_low_v_high]["p1_gain"] - mirr[key_low_v_low]["p1_gain"]
    )

    avg_self_advantage = (
        orig_initiator_adv
        + orig_responder_adv
        + mirr_initiator_adv
        + mirr_responder_adv
    ) / 4

    orig_surplus_adv = (
        (orig[key_high_v_low]["total_gain"] - orig[key_low_v_low]["total_gain"])
        + (orig[key_low_v_high]["total_gain"] - orig[key_low_v_low]["total_gain"])
    ) / 2
    mirr_surplus_adv = (
        (mirr[key_high_v_low]["total_gain"] - mirr[key_low_v_low]["total_gain"])
        + (mirr[key_low_v_high]["total_gain"] - mirr[key_low_v_low]["total_gain"])
    ) / 2
    avg_surplus_advantage = (orig_surplus_adv + mirr_surplus_adv) / 2

    return {
        "orig_initiator": orig_initiator_adv,
        "orig_responder": orig_responder_adv,
        "mirr_initiator": mirr_initiator_adv,
        "mirr_responder": mirr_responder_adv,
        "avg_self_mirrored": avg_self_advantage,
        "avg_surplus_mirrored": avg_surplus_advantage,
    }


# ---------------------------------------------------------------------------
# Board evaluation (runs in worker processes)
# ---------------------------------------------------------------------------


def evaluate_single_board(task):
    """
    Evaluate one board across all 8 matchups (4 configs x 2 orientations).

    Designed to be called by multiprocessing.Pool.map().
    Receives a single dict with all needed info, returns a result dict
    or None if the board does not meet the threshold.
    """
    board = task["board"]
    chips_p0 = task["chips_p0"]
    chips_p1 = task["chips_p1"]
    goal_p0 = task["goal_p0"]
    goal_p1 = task["goal_p1"]
    tom_low = task["tom_low"]
    tom_high = task["tom_high"]
    threshold = task["threshold"]
    board_index = task["board_index"]

    label_low = f"tom{tom_low}"
    label_high = f"tom{tom_high}"

    matchup_configs = [
        (f"{label_low}v{label_low}", tom_low, tom_low),
        (f"{label_high}v{label_low}", tom_high, tom_low),
        (f"{label_low}v{label_high}", tom_low, tom_high),
        (f"{label_high}v{label_high}", tom_high, tom_high),
    ]

    # --- Original orientation ---
    original_results = {}
    for key, tom_p0, tom_p1 in matchup_configs:
        original_results[key] = play_matchup(
            board, chips_p0, chips_p1, goal_p0, goal_p1, tom_p0, tom_p1
        )

    # --- Mirrored orientation: swap chips and goals ---
    mirrored_results = {}
    for key, tom_p0, tom_p1 in matchup_configs:
        mirrored_results[key] = play_matchup(
            board, chips_p1, chips_p0, goal_p1, goal_p0, tom_p0, tom_p1
        )

    advantage = compute_advantage(
        original_results, mirrored_results, label_high, label_low
    )

    if advantage["avg_self_mirrored"] < threshold:
        return None

    # --- Compute starting utilities for metadata ---
    game_ref = Game_ct()
    game_ref.load_setting(board, chips_p0[:], chips_p1[:], goal_p0, goal_p1)
    p0_start_utility = game_ref.utility_function[game_ref.locations[0]][
        game_ref.chip_sets[0]
    ]
    p1_start_utility = game_ref.utility_function[game_ref.locations[1]][
        game_ref.chip_sets[1]
    ]

    return {
        "board_index": board_index,
        "board": board,
        "chips1": chips_p0,
        "chips2": chips_p1,
        "location1": goal_p0,
        "location2": goal_p1,
        "p0_initial_utility": p0_start_utility,
        "p1_initial_utility": p1_start_utility,
        "original_matchups": original_results,
        "mirrored_matchups": mirrored_results,
        "advantage": advantage,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_advantage_boards(
    tom_low: int = 1,
    tom_high: int = 2,
    threshold: float = 100,
    target: int = 50,
    max_search: int = 2000,
    seed: int = 42,
    workers: int | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Search for boards where a higher-order ToM agent outperforms a lower-order one.

    Returns a list of dicts with fields: board, chips1, chips2, location1, location2,
    p0_initial_utility, p1_initial_utility, original_matchups, mirrored_matchups, advantage.
    """
    random.seed(seed)
    num_workers = workers or multiprocessing.cpu_count()

    label_low = f"tom{tom_low}"
    label_high = f"tom{tom_high}"
    key_lvl = f"{label_low}v{label_low}"
    key_hvl = f"{label_high}v{label_low}"
    key_lvh = f"{label_low}v{label_high}"

    if not quiet:
        print(
            f"Searching for {target} boards where ToM-{tom_high} "
            f"beats ToM-{tom_low} by >= {threshold} utility"
        )
        print(f"Using MIRRORED matchups to control for seat advantage")
        print(f"Workers: {num_workers} | Seed: {seed} | Max search: {max_search}\n")

    # Phase 1: Generate candidates sequentially
    if not quiet:
        print(f"Phase 1: Generating {max_search} candidate boards...")

    tasks = []
    stage1_rejected_reachability = 0
    stage1_rejected_same_goal = 0

    for board_index in range(max_search):
        game = Game_ct()
        valid = game.init()

        if valid is False:
            stage1_rejected_reachability += 1
            continue

        if game.locations[0] == game.locations[1]:
            stage1_rejected_same_goal += 1
            continue

        tasks.append(
            {
                "board_index": board_index,
                "board": [row[:] for row in game.board],
                "chips_p0": game.chips[0][:],
                "chips_p1": game.chips[1][:],
                "goal_p0": game.locations[0],
                "goal_p1": game.locations[1],
                "tom_low": tom_low,
                "tom_high": tom_high,
                "threshold": threshold,
            }
        )

    if not quiet:
        print(f"  Attempted:                  {max_search}")
        print(f"  Rejected (reachability):    {stage1_rejected_reachability}")
        print(f"  Rejected (identical goals): {stage1_rejected_same_goal}")
        print(f"  Passed to phase 2:          {len(tasks)}\n")

    # Phase 2: Evaluate boards in parallel
    if not quiet:
        print(f"Phase 2: Evaluating across {num_workers} workers...")

    qualifying_boards = []
    completed = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(evaluate_single_board, tasks):
            completed += 1

            if result is None:
                if not quiet and (completed % 25 == 0 or completed == len(tasks)):
                    print(
                        f"  [{completed}/{len(tasks)} evaluated, "
                        f"{len(qualifying_boards)} qualifying so far]"
                    )
                continue

            qualifying_boards.append(result)

            if not quiet:
                avg_adv = result["advantage"]["avg_self_mirrored"]
                original = result["original_matchups"]
                mirrored = result["mirrored_matchups"]

                print(
                    f"  [{len(qualifying_boards):3d}/{target}] "
                    f"Board {result['board_index']} "
                    f"({completed}/{len(tasks)} evaled): "
                    f"avg_adv={avg_adv:+.0f} | "
                    f"Orig {key_lvl}:{format_matchup_result(original[key_lvl])} "
                    f"{key_hvl}:{format_matchup_result(original[key_hvl])} "
                    f"{key_lvh}:{format_matchup_result(original[key_lvh])} | "
                    f"Mirr {key_lvl}:{format_matchup_result(mirrored[key_lvl])} "
                    f"{key_hvl}:{format_matchup_result(mirrored[key_hvl])} "
                    f"{key_lvh}:{format_matchup_result(mirrored[key_lvh])}"
                )

            if len(qualifying_boards) >= target:
                pool.terminate()
                break

    qualifying_boards.sort(
        key=lambda b: b["advantage"]["avg_self_mirrored"],
        reverse=True,
    )

    for entry in qualifying_boards:
        entry.pop("board_index", None)

    if not quiet:
        print(f"\n{'=' * 60}")
        print("SEARCH COMPLETE")
        print(f"{'=' * 60}")
        print(f"Comparison:                 ToM-{tom_high} vs ToM-{tom_low}")
        print(f"Attempted:                  {max_search}")
        print(f"Rejected (reachability):    {stage1_rejected_reachability}")
        print(f"Rejected (identical goals): {stage1_rejected_same_goal}")
        print(f"Passed to phase 2:          {len(tasks)}")
        print(f"Evaluated:                  {completed}")
        print(f"Qualifying (>= {threshold}): {len(qualifying_boards)}")
        if completed > 0:
            print(
                f"Hit rate:                   {len(qualifying_boards) / completed * 100:.1f}%"
            )

        if qualifying_boards:
            advantages = [
                b["advantage"]["avg_self_mirrored"] for b in qualifying_boards
            ]
            print(f"\nAdvantage distribution (mirrored, seat-controlled):")
            print(f"  Min:    {min(advantages):+.0f}")
            print(f"  Median: {sorted(advantages)[len(advantages) // 2]:+.0f}")
            print(f"  Max:    {max(advantages):+.0f}")
            print(f"  Mean:   {sum(advantages) / len(advantages):+.1f}")

    return qualifying_boards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find Colored Trails boards where a higher-order ToM agent "
        "outperforms a lower-order one by a specified utility margin "
        "(seat-controlled via mirroring)."
    )
    parser.add_argument(
        "--tom-low",
        type=int,
        default=0,
        help="ToM level for the baseline (lower) agent (default: 0)",
    )
    parser.add_argument(
        "--tom-high",
        type=int,
        default=1,
        help="ToM level for the upgraded (higher) agent (default: 1)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="How many qualifying boards to collect (default: 50)",
    )
    parser.add_argument(
        "--max-search",
        type=int,
        default=2000,
        help="Maximum random boards to generate (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100,
        help="Minimum avg_self_mirrored advantage to keep a board (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="board_generation",
        help="Directory for output JSON (default: board_generation)",
    )
    args = parser.parse_args()

    if args.tom_high <= args.tom_low:
        parser.error(
            f"--tom-high ({args.tom_high}) must be greater than "
            f"--tom-low ({args.tom_low})"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    boards = find_advantage_boards(
        tom_low=args.tom_low,
        tom_high=args.tom_high,
        threshold=args.threshold,
        target=args.target,
        max_search=args.max_search,
        seed=args.seed,
        workers=args.workers,
        quiet=False,
    )

    label_low = f"tom{args.tom_low}"
    label_high = f"tom{args.tom_high}"

    output = {
        "metadata": {
            "scenario": f"{label_high}_vs_{label_low}_advantage_mirrored",
            "description": (
                f"Boards where ToM-{args.tom_high} outperforms "
                f"ToM-{args.tom_low} by >= {args.threshold} avg utility, "
                f"measured across mirrored seat assignments to control "
                f"for positional advantage."
            ),
            "tom_high": args.tom_high,
            "tom_low": args.tom_low,
            "count": len(boards),
            "candidates_attempted": args.max_search,
            "threshold": args.threshold,
            "seed": args.seed,
            "max_negotiation_rounds": MAX_NEGOTIATION_ROUNDS,
            "learning_speed": LEARNING_SPEED,
            "timestamp": datetime.now().isoformat(),
        },
        "boards": boards,
    }

    filename = (
        f"{label_high}_vs_{label_low}_advantage_mirrored_{int(args.threshold)}.json"
    )
    output_path = os.path.join(args.out_dir, filename)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(boards)} boards to {output_path}")
