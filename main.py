"""
main.py

Main execution script for a Colored Trails Agent match.
This script orchestrates a 2-player negotiation game across a series of pre-generated boards,
capable of evaluating various AI agents (ToM Baselines, LLMs via API, and open-source LLMs).
It also features checkpointing for safe resumption of interrupted matches.

Usage:
python main.py <player_0> <player_1>
"""

import argparse
import json
import os

from agents.human_agent import Human
from game.colored_trails import Game_ct
from game.history import GameHistory
from agents.tom_agent import Agent_ct
from agents.api_agent import APIAgent
from agents.hf_agent import HFAgent
from agents.shadow_agent import ShadowAgent, RandomShadowAgent
from results.analysis import run_analysis_pipeline
from dotmap import DotMap


# FILE IO & CHECKPOINT HELPERS


def load_boards_from_json(filepath: str) -> list[dict]:
    """Load pre-generated boards from a JSON file."""
    with open(filepath, "r") as f:
        return [
            {
                "board": b["board"],
                "chips1": b["chips1"],
                "chips2": b["chips2"],
                "location1": b["location1"],
                "location2": b["location2"],
            }
            for b in json.load(f)["boards"]
        ]


def load_checkpoint(filepath, agents_tuple, shadows_tuple):
    """Restores tournament state and agent models from a checkpoint file."""
    if not os.path.exists(filepath):
        return None

    print(f"\n[!] Found checkpoint: {filepath}. Resuming tournament...")
    with open(filepath, "r") as f:
        ckpt = json.load(f)

    # Restore playing agents
    for agent, state_key in zip(agents_tuple, ["p0_c1", "p1_c1", "p0_c2", "p1_c2"]):
        if hasattr(agent, "load_state"):
            agent.load_state(ckpt["agents_state"][state_key])

    # Restore shadow agents
    for group, key in zip(shadows_tuple, ["shadow_states_c1", "shadow_states_c2"]):
        if key in ckpt:
            for name, shadow in group.items():
                if hasattr(shadow, "load_state"):
                    shadow.load_state(ckpt[key].get(name))

    return ckpt


def save_checkpoint(filepath, ckpt_data, agents_tuple, shadows_tuple):
    """Saves tournament state and agent models to a checkpoint file."""
    agents_state = {
        key: agent.get_state() if hasattr(agent, "get_state") else None
        for agent, key in zip(agents_tuple, ["p0_c1", "p1_c1", "p0_c2", "p1_c2"])
    }
    ckpt_data["agents_state"] = agents_state
    ckpt_data["shadow_states_c1"] = {
        k: v.get_state() for k, v in shadows_tuple[0].items() if hasattr(v, "get_state")
    }
    ckpt_data["shadow_states_c2"] = {
        k: v.get_state() for k, v in shadows_tuple[1].items() if hasattr(v, "get_state")
    }

    with open(filepath, "w") as f:
        json.dump(ckpt_data, f)


# AGENT SETUP & GAME LOGIC


def create_agent(agent_type, player_id, learning_speed=0.8, debug=True):
    """Factory pattern for instantiating game agents cleanly."""
    agent_type = agent_type.lower()

    if agent_type.startswith("tom"):
        return Agent_ct(
            int(agent_type.replace("tom", "")),
            player_id,
            learning_speed=learning_speed,
            debug=debug,
        )

    match agent_type:
        case "claude":
            return APIAgent(
                player_id, provider="anthropic", model_name="claude-sonnet-4-6"
            )
        case "gemini":
            return APIAgent(player_id, provider="google", model_name="gemini-2.5-pro")
        case "qwen":
            return HFAgent(player_id, model_path="Qwen/Qwen2.5-72B-Instruct")
        case "llama":
            return HFAgent(player_id, model_path="meta-llama/Meta-Llama-3-8B-Instruct")
        case "gpt-oss":
            return HFAgent(player_id, model_path="openai/gpt-oss-120b")
        case "human":
            return Human(player_id)
        case _:
            print(
                f"[{player_id}] Unknown agent type '{agent_type}'. "
                f"Defaulting to Human. (Results will be stored under this name.)"
            )
            return Human(player_id)


def _evaluate_shadows(
    shadow_agents, current_player, incoming_offer, offer, game, game_evs
):
    """Updates shadow models and caches raw EVs."""
    actual_incoming = (
        incoming_offer if incoming_offer is not None else game.chip_sets[current_player]
    )

    for name, shadow in shadow_agents.items():
        if actual_incoming != game.chip_sets[current_player]:
            shadow.receive_offer(actual_incoming)

        # Capture the EVs
        chosen_offer, evs = shadow.get_evs(offer, actual_incoming)

        # Cache raw data for post-game analysis
        if evs is not None:
            game_evs[name].append({"chosen": chosen_offer, "evs": evs})
        else:
            # Handle RandomAgent probability natively
            num_possible_offers = len(game.utility_function[0])
            prob = 1.0 / num_possible_offers
            game_evs[name].append({"chosen": chosen_offer, "prob": prob})

        shadow.send_offer(offer)


def run_single_game(
    game, agent_init, agent_resp, history=None, shadow_agents=None, tracked_player=None
):
    """Executes a single game of Colored Trails between two agents.

    If `history` is provided, it is populated in place with negotiation messages
    and the final outcome. The return value is a 4-tuple
    `(delta_0, delta_1, game_evs, history)` where `delta_*` are the per-player
    utility changes. If the game errors (invalid LLM response), `None` is returned
    after stamping an "error" outcome into the supplied history.
    """
    shadow_agents = shadow_agents or {}
    u_init = [
        game.utility_function[game.locations[i]][game.chip_sets[i]] for i in (0, 1)
    ]

    for a in (agent_init, agent_resp):
        a.init(game)
    for shadow in shadow_agents.values():
        shadow.init(game)

    game_evs = {name: [] for name in shadow_agents}

    hide_opponent_goal = (
        isinstance(agent_init, Human) or isinstance(agent_resp, Human)
    )

    for row in game.board:
        print(str(row))
    for i, role in enumerate(["Initiator", "Responder"]):
        if hide_opponent_goal and i == 1:
            print(
                f"P{i} ({role}): Chips {Game_ct.convert_code(game.chip_sets[i], game.bin_max)}, Starting utility: {u_init[i]}"
            )
        else:
            print(
                f"P{i} ({role}): Goal {game.locations[i]}, Chips {Game_ct.convert_code(game.chip_sets[i], game.bin_max)}, Starting utility: {u_init[i]}"
            )

    current_player, incoming_offer = 0, None
    incoming_message: str | None = None
    agreement_reached, final_chips = False, list(game.chip_sets)
    rounds_played = 0

    for rounds in range(100):
        rounds_played = rounds
        role = "initiating" if rounds == 0 else "responding"
        print(f"\n[Round {rounds}] P{current_player} ({role}):")

        active_agent = agent_init if current_player == 0 else agent_resp
        try:
            offer, message = active_agent.make_offer(
                incoming_offer,
                round_idx=rounds,
                role=role,
                incoming_message=incoming_message,
            )
        except ValueError as e:
            print(f"  -> INVALID: {e}")
            if history is not None:
                history.outcome = DotMap({
                    "result": "error",
                    "rounds": rounds,
                    "error": str(e),
                })
            return None

        if history is not None:
            history.messages.append(message)

        incoming_message = message.message

        if current_player == tracked_player and shadow_agents:
            _evaluate_shadows(
                shadow_agents, current_player, incoming_offer, offer, game, game_evs
            )

        if offer == game.chip_sets[current_player]:
            print("  -> WITHDREW")
            break

        if rounds > 0 and offer == incoming_offer:
            agreement_reached = True
            final_chips[current_player] = offer
            final_chips[1 - current_player] = game.flip_array[offer]
            print("  -> ACCEPTED")
            break

        print(f"  -> Offers: {offer} {Game_ct.convert_code(offer, game.bin_max)}")
        incoming_offer = game.flip_array[offer]
        current_player = 1 - current_player

    cost = (rounds_played + 1) * 1
    s_end = [
        game.utility_function[game.locations[i]][final_chips[i]] - cost for i in (0, 1)
    ]

    print(
        f"\nGame Ended | Rounds: {rounds_played + 1} | Cost: {cost} | Agreement: {agreement_reached}"
    )
    print(f"Final Utilities: P0={s_end[0]:.1f}, P1={s_end[1]:.1f}")

    if history is not None:
        if agreement_reached:
            result = "accepted"
        elif offer == game.chip_sets[0] or offer == game.chip_sets[1]:
            result = "withdrawn"
        else:
            result = "max_rounds"
        history.rounds = rounds_played + 1
        history.outcome = DotMap({
            "result": result,
            "rounds": rounds_played + 1,
            "agreement": agreement_reached,
            "cost": cost,
            "final_chips": [
                Game_ct.convert_code(final_chips[0], game.bin_max),
                Game_ct.convert_code(final_chips[1], game.bin_max),
            ],
            "final_utilities": [s_end[0], s_end[1]],
            "deltas": [s_end[0] - u_init[0], s_end[1] - u_init[1]],
        })

    return (s_end[0] - u_init[0]), (s_end[1] - u_init[1]), game_evs, history


def _record_game_stats(
    stats, deltas, evidence_matrix, result, p0_name, p1_name, config_key, board_idx
):
    """Updates global tracking dictionaries with the results of a completed game."""
    init_score, resp_score, game_evs, _history = result

    stats[config_key]["init_score"] += init_score
    stats[config_key]["resp_score"] += resp_score
    stats[config_key]["count"] += 1

    if config_key == "config1":
        deltas[p0_name].append(init_score)
        deltas[p1_name].append(resp_score)
    else:
        deltas[p1_name].append(init_score)
        deltas[p0_name].append(resp_score)

    # Ensure game_evs isn't empty before appending
    if any(len(ev_list) > 0 for ev_list in game_evs.values()):
        cfg_id = "C1" if config_key == "config1" else "C2"
        evidence_matrix.append(
            {"game_id": f"Board_{board_idx}_{cfg_id}", "evs": game_evs}
        )


# MAIN EXECUTION


def main():
    """Initializes the agents, runs the games, calls the post-hoc analyses and saves the results."""
    parser = argparse.ArgumentParser(description="Run Colored Trails tournament.")
    agent_choices = [
        "tom0",
        "tom1",
        "tom2",
        "qwen",
        "llama",
        "gpt-oss",
        "gemini",
        "claude",
        "human",
    ]

    parser.add_argument("player_0", type=str)
    parser.add_argument("player_1", type=str)
    parser.add_argument("--boards-file", type=str, default="boards_20.json")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-speed", type=float, default=0.8)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # --- Directory Setup ---
    dir_checkpoints = os.path.join("results", "checkpoints")
    dir_deltas = os.path.join("results", "delta_scores")
    dir_evidence = os.path.join("results", "evidence_matrices")
    dir_raw_ev = os.path.join("results", "raw_evs")
    dir_games = os.path.join("results", "games", f"{args.player_0}_vs_{args.player_1}")
    for dir_path in [dir_checkpoints, dir_deltas, dir_evidence, dir_raw_ev, dir_games]:
        os.makedirs(dir_path, exist_ok=True)
    # -----------------------

    # Initialize Agents (Config 1 & Config 2)
    p0_c1 = create_agent(args.player_0, 0, args.learning_speed, args.debug)
    p1_c1 = create_agent(args.player_1, 1, args.learning_speed, args.debug)
    # shadows_c1 = {f"tom{i}": ShadowAgent(i, 0) for i in range(3)} | {"random": RandomShadowAgent(0)}
    shadows_c1 = {}  # Run without shadow agent tracking

    p0_c2 = create_agent(args.player_1, 0, args.learning_speed, args.debug)
    p1_c2 = create_agent(args.player_0, 1, args.learning_speed, args.debug)
    # shadows_c2 = {f"tom{i}": ShadowAgent(i, 1) for i in range(3)} | {"random": RandomShadowAgent(1)}
    shadows_c2 = {}  # Run without shadow agent tracking

    agents_tuple = (p0_c1, p1_c1, p0_c2, p1_c2)
    shadows_tuple = (shadows_c1, shadows_c2)

    # Tournament State Variables
    game = Game_ct()
    stats = {
        "config1": {"init_score": 0, "resp_score": 0, "count": 0},
        "config2": {"init_score": 0, "resp_score": 0, "count": 0},
    }
    deltas = {args.player_0: [], args.player_1: []}
    evidence_matrix = []
    skipped_boards = 0
    start_index = 0

    if args.boards_file:
        print(f"Loading pre-generated boards from {args.boards_file}...")
        boards_data = load_boards_from_json(args.boards_file)
        checkpoint_file = os.path.join(
            dir_checkpoints, f"checkpoint_{args.player_0}_vs_{args.player_1}.json"
        )

        # Resume from checkpoint
        ckpt = load_checkpoint(checkpoint_file, agents_tuple, shadows_tuple)
        if ckpt:
            stats = ckpt["stats"]
            deltas = ckpt["delta_scores"]
            evidence_matrix = ckpt["evidence_matrix"]
            skipped_boards = ckpt.get("skipped_boards", 0)
            start_index = ckpt["board_idx"] + 1

        # Tournament Loop
        for i, b in enumerate(boards_data[start_index:], start=start_index):
            # Config 1
            print(
                f"\n========== Board {i + 1}/{len(boards_data)} | Config 1: {args.player_0} vs {args.player_1} =========="
            )
            game.load_setting(
                b["board"], b["chips1"], b["chips2"], b["location1"], b["location2"]
            )
            history_c1 = GameHistory.from_game(
                game,
                p0_type=args.player_0,
                p1_type=args.player_1,
                initiator=0,
                board_index=i + 1,
                config="config1",
            )
            result_c1 = run_single_game(
                game, p0_c1, p1_c1, history=history_c1,
                shadow_agents=shadows_c1, tracked_player=0,
            )
            path_c1 = os.path.join(dir_games, f"board_{i + 1}_c1.json")
            if os.path.exists(path_c1):
                print(f"  -> Skipping save (already exists): {path_c1}")
            else:
                history_c1.save(path_c1)

            # Config 2
            print(
                f"\n========== Board {i + 1}/{len(boards_data)} | Config 2: {args.player_1} vs {args.player_0} =========="
            )
            game.load_setting(
                b["board"], b["chips1"], b["chips2"], b["location1"], b["location2"]
            )
            history_c2 = GameHistory.from_game(
                game,
                p0_type=args.player_1,
                p1_type=args.player_0,
                initiator=1,
                board_index=i + 1,
                config="config2",
            )
            result_c2 = run_single_game(
                game, p0_c2, p1_c2, history=history_c2,
                shadow_agents=shadows_c2, tracked_player=1,
            )
            path_c2 = os.path.join(dir_games, f"board_{i + 1}_c2.json")
            if os.path.exists(path_c2):
                print(f"  -> Skipping save (already exists): {path_c2}")
            else:
                history_c2.save(path_c2)

            # Skip & Record Tracking
            if result_c1 is None or result_c2 is None:
                skipped_boards += 1
                print(f"  -> Board {i + 1} excluded from results.")
                continue

            _record_game_stats(
                stats,
                deltas,
                evidence_matrix,
                result_c1,
                args.player_0,
                args.player_1,
                "config1",
                i + 1,
            )
            _record_game_stats(
                stats,
                deltas,
                evidence_matrix,
                result_c2,
                args.player_0,
                args.player_1,
                "config2",
                i + 1,
            )

            # Save Checkpoint
            ckpt_data = {
                "board_idx": i,
                "skipped_boards": skipped_boards,
                "stats": stats,
                "delta_scores": deltas,
                "evidence_matrix": evidence_matrix,
            }
            save_checkpoint(checkpoint_file, ckpt_data, agents_tuple, shadows_tuple)

    # FINAL REPORTING AND ANALYSIS

    c1, c2 = stats["config1"], stats["config2"]
    n1, n2 = max(1, c1["count"]), max(1, c2["count"])  # Prevent Div by 0

    if skipped_boards:
        print(f"  Skipped {skipped_boards} board(s) due to invalid agent responses.\n")

    i1, r1 = c1["init_score"] / n1, c1["resp_score"] / n1
    i2, r2 = c2["init_score"] / n2, c2["resp_score"] / n2

    print(f"\n{'=' * 80}\nFINAL RESULTS\n{'=' * 80}")
    print(
        f"Config 1 (Init: {args.player_0}, Resp: {args.player_1}) | {args.player_0} Avg: {i1:.2f} | {args.player_1} Avg: {r1:.2f}"
    )
    print(
        f"Config 2 (Init: {args.player_1}, Resp: {args.player_0}) | {args.player_1} Avg: {i2:.2f} | {args.player_0} Avg: {r2:.2f}"
    )
    print("-" * 80)
    print(
        f"Overall ({c1['count'] + c2['count']} Games) | {args.player_0} Avg: {(i1 + r2) / 2:.2f} | {args.player_1} Avg: {(r1 + i2) / 2:.2f}"
    )

    # File Exports
    output_path = os.path.join(
        dir_deltas, f"delta_scores_{args.player_0}_vs_{args.player_1}_lo.json"
    )
    with open(output_path, "w") as f:
        json.dump(deltas, f, indent=2)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(
            f"\n[!] Tournament files saved securely. Removed checkpoint: {checkpoint_file}"
        )

    # Offload MLE Fitting & BMS to the analysis module
    if evidence_matrix:
        csv_path = os.path.join(
            dir_evidence, f"evidence_matrix_{args.player_0}_vs_{args.player_1}_lo.csv"
        )
        json_path = os.path.join(
            dir_raw_ev, f"raw_ev_cache_{args.player_0}_vs_{args.player_1}_lo.json"
        )
        shadow_names = list(
            shadows_c1.keys()
        )  # Note that only <player 0> is analyzed by default
        run_analysis_pipeline(evidence_matrix, shadow_names, csv_path, json_path)


if __name__ == "__main__":
    main()
