"""
generate_boards.py

Generates a collection of valid Colored Trails boards that pass the
game's built-in pruning rules:

  Rule 1 (Reachability): Each player's goal must be reachable when
            holding all chips in the game (utility >= 800).
  Rule 2 (Challenge):    Each player's goal must NOT be reachable
            with their starting chips alone (utility < 800).

Goals must also be at Manhattan distance > 2 from the centre (the start
position), and both players must have distinct goals.

Usage
-----
    python generate_boards.py --target 100
    python generate_boards.py --target 200 --seed 0 --max-search 5000

Output
------
    board_generation/boards_<count>_seed<seed>.json
"""

import random
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.colored_trails import Game_ct


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate valid Colored Trails boards satisfying the game's pruning rules."
    )
    parser.add_argument("--target", type=int, default=100,
                        help="Number of boards to collect (default: 100)")
    parser.add_argument("--max-search", type=int, default=10000,
                        help="Maximum boards to attempt before stopping (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out-dir", type=str, default="board_generation",
                        help="Directory for the output JSON (default: board_generation)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    boards = []
    attempted = 0
    rejected_reachability = 0
    rejected_same_goal = 0

    print(f"Generating {args.target} valid boards  "
          f"(seed={args.seed}, max-search={args.max_search})\n")

    for _ in range(args.max_search):
        if len(boards) >= args.target:
            break

        attempted += 1
        game = Game_ct()
        valid = game.init()

        if valid is False:
            rejected_reachability += 1
            continue

        if game.locations[0] == game.locations[1]:
            rejected_same_goal += 1
            continue

        p0_start_utility = game.utility_function[game.locations[0]][game.chip_sets[0]]
        p1_start_utility = game.utility_function[game.locations[1]][game.chip_sets[1]]

        boards.append({
            "board":              [row[:] for row in game.board],
            "chips1":             game.chips[0][:],
            "chips2":             game.chips[1][:],
            "location1":          game.locations[0],
            "location2":          game.locations[1],
            "p0_initial_utility": p0_start_utility,
            "p1_initial_utility": p1_start_utility,
        })

        if len(boards) % 50 == 0:
            print(f"  [{len(boards)}/{args.target}] boards collected so far ...")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Attempted:                  {attempted}")
    print(f"Rejected (reachability):    {rejected_reachability}")
    print(f"Rejected (identical goals): {rejected_same_goal}")
    print(f"Collected:                  {len(boards)}")
    if attempted > 0:
        print(f"Acceptance rate:            {len(boards) / attempted * 100:.1f}%")

    output = {
        "metadata": {
            "description": (
                "Colored Trails boards satisfying both pruning rules: "
                "goal reachable with all chips (utility >= 800) and "
                "goal unreachable with starting chips alone (utility < 800)."
            ),
            "count":                 len(boards),
            "candidates_attempted":  attempted,
            "rejected_reachability": rejected_reachability,
            "rejected_same_goal":    rejected_same_goal,
            "seed":                  args.seed,
            "max_search":            args.max_search,
            "timestamp":             datetime.now().isoformat(),
        },
        "boards": boards,
    }

    filename = f"boards_{len(boards)}_seed{args.seed}.json"
    output_path = os.path.join(args.out_dir, filename)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(boards)} boards to {output_path}")
