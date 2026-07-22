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
import random
from datetime import datetime

from rich.console import Console

from agents.human_agent import Human
from agents.utils import format_chips
from game.colored_trails import Game_ct
from game.history import GameHistory
from agents.tom_agent import Agent_ct
from agents.api_agent import APIAgent
from agents.openrouter_agent import OpenRouterAgent
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


def create_agent(agent_type, player_id, learning_speed=0.8, debug=False, verbose=False):
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
                player_id,
                provider="anthropic",
                model_name="claude-sonnet-4-6",
                debug=debug,
            )
        case "gemini":
            return APIAgent(
                player_id,
                provider="google",
                model_name="gemini-2.5-pro",
                debug=debug,
            )
        case "qwen":
            return HFAgent(
                player_id,
                model_path="Qwen/Qwen2.5-72B-Instruct",
                debug=debug,
            )
        case "llama":
            return HFAgent(
                player_id,
                model_path="meta-llama/Meta-Llama-3-8B-Instruct",
                debug=debug,
            )
        case "gpt-oss":
            return HFAgent(
                player_id,
                model_path="openai/gpt-oss-120b",
                debug=debug,
            )
        case "qwen-small":
            return HFAgent(
                player_id,
                model_path="Qwen/Qwen3-0.6B",
                debug=debug,
            )
        case "openrouter":
            return OpenRouterAgent(
                player_id,
                model_name="openai/gpt-oss-120b",
                debug=debug,
                verbose=verbose,
            )
        case "human":
            return Human(player_id, debug=debug)
        case _:
            print(
                f"[{player_id}] Unknown agent type '{agent_type}'. "
                f"Defaulting to Human. (Results will be stored under this name.)"
            )
            return Human(player_id, debug=debug)


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
    game,
    agent_init,
    agent_resp,
    history=None,
    shadow_agents=None,
    tracked_player=None,
    chat_history=True,
):
    """Executes a single game of Colored Trails between two agents.

    If `history` is provided, it is populated in place with negotiation messages
    and the final outcome. The return value is a 4-tuple
    `(delta_0, delta_1, game_evs, history)` where `delta_*` are the per-player
    utility changes. If the game errors (invalid LLM response), `None` is returned
    after stamping an "error" outcome into the supplied history.

    When `chat_history` is True (default), each agent receives the full chat log
    of all messages exchanged so far. When False, each agent only receives the
    opponent's most recent message (legacy single-message mode).
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

    hide_opponent_goal = isinstance(agent_init, Human) or isinstance(agent_resp, Human)

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
    outgoing_by_player: list[list[dict]] = [[], []]
    agreement_reached, final_chips = False, list(game.chip_sets)
    rounds_played = 0

    for rounds in range(100):
        rounds_played = rounds
        role = "initiating" if rounds == 0 else "responding"
        print(f"\n[Round {rounds}] P{current_player} ({role}):")

        active_agent = agent_init if current_player == 0 else agent_resp
        opp_id = 1 - current_player
        if chat_history:
            chat_log_for_active = [
                {**entry, "speaker": "opponent"} for entry in outgoing_by_player[opp_id]
            ]
            incoming_for_active = None
        else:
            chat_log_for_active = None
            incoming_for_active = incoming_message

        try:
            offer, message = active_agent.make_offer(
                incoming_offer,
                round_idx=rounds,
                role=role,
                incoming_message=incoming_for_active,
                chat_log=chat_log_for_active,
            )
        except ValueError as e:
            print(f"  -> INVALID: {e}")
            if history is not None:
                history.outcome = DotMap(
                    {
                        "result": "error",
                        "rounds": rounds,
                        "error": str(e),
                    }
                )
            return None

        if history is not None:
            history.messages.append(message)

        if message.message:
            outgoing_by_player[current_player].append(
                {
                    "round": rounds,
                    "speaker": "you",
                    "text": message.message,
                }
            )

        incoming_message = message.message

        if current_player == tracked_player and shadow_agents:
            _evaluate_shadows(
                shadow_agents, current_player, incoming_offer, offer, game, game_evs
            )

        if message.action == "withdraw":
            print("  -> WITHDREW")
            break
        elif message.action == "accept":
            agreement_reached = True
            final_chips[current_player] = offer
            final_chips[1 - current_player] = game.flip_array[offer]
            print("  -> ACCEPTED")
            break
        else:
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
        elif message.action == "withdraw":
            result = "withdrawn"
        elif offer == game.chip_sets[0] or offer == game.chip_sets[1]:
            result = "withdrawn"
        else:
            result = "max_rounds"
        history.rounds = rounds_played + 1
        history.outcome = DotMap(
            {
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
            }
        )

    return (s_end[0] - u_init[0]), (s_end[1] - u_init[1]), game_evs, history


def run_single_human_game(
    game,
    agent_init,
    agent_resp,
    history=None,
    chat_history=True,
    deception=None,
):
    """Lean-stdout variant of `run_single_game` for matches involving a Human.

    The board rows are suppressed (the human sees the board in the rich Live
    UI); the per-round `[Round N] / -> Offers / -> ACCEPTED / -> WITHDREW`
    prints are replaced by a single `rich`-formatted line per move:

        P0 proposes  White:1, Black:0, Magenta:1, Gray:1, Yellow:0 — "msg"
        P1 accepts
        P1 withdraws

    Starting-info lines, `Game Ended | …`, and `Final Utilities: …` are kept.
    Returns the same 4-tuple as `run_single_game`. If the game errors,
    `None` is returned after stamping an "error" outcome into `history`.

    If `deception` is not None, it is set on any agent that has a `deception`
    attribute (i.e. LLM agents) after `init` is called.
    """
    console = Console()
    u_init = [
        game.utility_function[game.locations[i]][game.chip_sets[i]] for i in (0, 1)
    ]

    for a in (agent_init, agent_resp):
        a.init(game)
        if deception is not None and hasattr(a, "deception"):
            a.deception = deception

    init_id = agent_init.player_id
    resp_id = agent_resp.player_id
    print(
        f"P{init_id} (Initiator): Chips {Game_ct.convert_code(game.chip_sets[init_id], game.bin_max)}, "
        f"Starting utility: {u_init[init_id]}"
    )
    print(
        f"P{resp_id} (Responder): Chips {Game_ct.convert_code(game.chip_sets[resp_id], game.bin_max)}, "
        f"Starting utility: {u_init[resp_id]}"
    )

    header = " " * 12 + "   ".join(
        f"{c:>7}" for c in ["White", "Black", "Magenta", "Gray", "Yellow"]
    )
    print(header)

    current_player, incoming_offer = 0, None
    incoming_message: str | None = None
    outgoing_by_player: list[list[dict]] = [[], []]
    agreement_reached, final_chips = False, list(game.chip_sets)
    rounds_played = 0

    for rounds in range(100):
        rounds_played = rounds
        role = "initiating" if rounds == 0 else "responding"

        active_agent = agent_init if current_player == 0 else agent_resp
        opp_id = 1 - current_player
        if chat_history:
            chat_log_for_active = [
                {**entry, "speaker": "opponent"} for entry in outgoing_by_player[opp_id]
            ]
            incoming_for_active = None
        else:
            chat_log_for_active = None
            incoming_for_active = incoming_message

        try:
            offer, message = active_agent.make_offer(
                incoming_offer,
                round_idx=rounds,
                role=role,
                incoming_message=incoming_for_active,
                chat_log=chat_log_for_active,
            )
        except ValueError as e:
            print(f"  -> INVALID: {e}")
            if history is not None:
                history.outcome = DotMap(
                    {
                        "result": "error",
                        "rounds": rounds,
                        "error": str(e),
                    }
                )
            return None

        if history is not None:
            history.messages.append(message)

        if message.message:
            outgoing_by_player[current_player].append(
                {
                    "round": rounds,
                    "speaker": "you",
                    "text": message.message,
                }
            )

        incoming_message = message.message

        offer_vec = Game_ct.convert_code(offer, game.bin_max)
        chat = message.message
        if message.action == "withdraw":
            console.print(f"[bold]P{current_player}[/bold] [red]withdraws[/red]")
            break
        elif message.action == "accept":
            console.print(f"[bold]P{current_player}[/bold] [green]accepts[/green]")
            agreement_reached = True
            final_chips[current_player] = offer
            final_chips[1 - current_player] = game.flip_array[offer]
            break
        else:
            nums = "   ".join(f"{n:>7}" for n in offer_vec)
            suffix = f'  — "{chat}"' if chat else ""
            console.print(
                f"[bold]P{current_player}[/bold] [cyan]proposes[/cyan] {nums}{suffix}"
            )

            if offer == game.chip_sets[current_player]:
                break

            if rounds > 0 and offer == incoming_offer:
                agreement_reached = True
                final_chips[current_player] = offer
                final_chips[1 - current_player] = game.flip_array[offer]
                break

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
        elif message.action == "withdraw":
            result = "withdrawn"
        elif offer == game.chip_sets[0] or offer == game.chip_sets[1]:
            result = "withdrawn"
        else:
            result = "max_rounds"
        history.rounds = rounds_played + 1
        history.outcome = DotMap(
            {
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
            }
        )

    return (s_end[0] - u_init[0]), (s_end[1] - u_init[1]), {}, history


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


# PANEL EXPERIMENT

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


def _augment_board(board_data: dict, seed_augment: int | str) -> dict:
    """Return a colour-permuted, randomly-flipped copy of *board_data*.

    Deterministic given *seed_augment*.  Permutes the 5 colours, remaps
    chips accordingly, then applies a random horizontal and/or vertical
    flip.  Goal-location indices are remapped to the correct entries
    after flips.
    """
    local_rng = random.Random(str(seed_augment))

    colours = list(range(5))
    local_rng.shuffle(colours)

    new_board = [[colours[cell] for cell in row] for row in board_data["board"]]

    new_chips1 = [0] * 5
    new_chips2 = [0] * 5
    for i in range(5):
        new_chips1[colours[i]] = board_data["chips1"][i]
        new_chips2[colours[i]] = board_data["chips2"][i]

    h_flip = local_rng.choice([True, False])
    v_flip = local_rng.choice([True, False])

    if h_flip or v_flip:
        flipped = [[0] * 5 for _ in range(5)]
        for r in range(5):
            for c in range(5):
                src_r = (4 - r) if v_flip else r
                src_c = (4 - c) if h_flip else c
                flipped[r][c] = new_board[src_r][src_c]
        new_board = flipped

    def _remap_loc(loc_idx):
        r, c = VALID_GOAL_COORDS[loc_idx]
        if v_flip:
            r = 4 - r
        if h_flip:
            c = 4 - c
        return VALID_GOAL_COORDS.index((r, c))

    return {
        "board": new_board,
        "chips1": new_chips1,
        "chips2": new_chips2,
        "location1": _remap_loc(board_data["location1"]),
        "location2": _remap_loc(board_data["location2"]),
    }


def run_panel_experiment(
    llm_type: str,
    *,
    seed: int = 42,
    normal_boards_count: int = 5,
    advantage_boards_count: int = 5,
    advantage_threshold: float = 100,
    advantage_boards_file: str | None = None,
    boards_file: str = "boards_20.json",
    debug: bool = False,
    verbose: bool = False,
):
    """Run a panel experiment: Human vs LLM across 20 games.

    Loads *normal_boards_count* + *advantage_boards_count* unique boards.
    Each board is played **twice** — once with deception ON, once OFF —
    exactly 10 games apart.  The two plays use the same base board but
    the second play applies a deterministic colour permutation and random
    flips so the board geometry is not trivially recognisable.

    The order of boards and whether deception precedes non-deception for
    each board are both randomised (conditioned on *seed*).  Config 1/2
    alternates across all 20 games, giving each board one config-1 and
    one config-2 play.

    Advantage boards are generated on-the-fly using
    ``board_generation/find_utility_advantage.py`` (ToM-2 vs ToM-1).

    Results save to ``results/panel_results/{llm_type}/seed_{seed}/``.
    """
    random.seed(seed)

    normal_boards = load_boards_from_json(boards_file)
    if len(normal_boards) < normal_boards_count:
        print(f"Warning: only {len(normal_boards)} normal boards available, using all.")
        normal_boards_count = len(normal_boards)
    normal_boards = normal_boards[:normal_boards_count]

    try:
        from board_generation.find_utility_advantage import find_advantage_boards
    except ImportError:
        print(
            "Error: cannot import board_generation.find_utility_advantage."
            " Make sure the project root is on PYTHONPATH."
        )
        return

    advantage_boards: list[dict] = []

    if advantage_boards_file and os.path.exists(advantage_boards_file):
        print(f"Loading pre-generated advantage boards from {advantage_boards_file}...")
        try:
            loaded = load_boards_from_json(advantage_boards_file)
        except Exception:
            loaded = []
            print(
                f"  -> Failed to parse {advantage_boards_file}, will generate instead."
            )
        if len(loaded) >= advantage_boards_count:
            advantage_boards = loaded[:advantage_boards_count]
            print(
                f"  -> Loaded {len(advantage_boards)}/{advantage_boards_count} boards."
            )
        else:
            print(
                f"  -> Only {len(loaded)} boards in file (need "
                f"{advantage_boards_count}); will generate instead."
            )

    if not advantage_boards:
        advantage_boards = find_advantage_boards(
            tom_low=1,
            tom_high=2,
            threshold=advantage_threshold,
            target=advantage_boards_count,
            max_search=max(500, advantage_boards_count * 100),
            seed=seed,
            quiet=True,
        )

        advantage_boards = [
            {
                "board": b["board"],
                "chips1": b["chips1"],
                "chips2": b["chips2"],
                "location1": b["location1"],
                "location2": b["location2"],
            }
            for b in advantage_boards
        ]
        advantage_boards_count = len(advantage_boards)

        if advantage_boards and advantage_boards_file:
            os.makedirs(os.path.dirname(advantage_boards_file) or ".", exist_ok=True)
            output = {
                "metadata": {
                    "description": (
                        f"Advantage boards where ToM-2 outperforms ToM-1 "
                        f"by >= {advantage_threshold} avg utility."
                    ),
                    "tom_high": 2,
                    "tom_low": 1,
                    "count": len(advantage_boards),
                    "threshold": advantage_threshold,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                },
                "boards": advantage_boards,
            }
            with open(advantage_boards_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  -> Saved to {advantage_boards_file}")

    board_pairs = [(b, "normal") for b in normal_boards] + [
        (b, "advantage") for b in advantage_boards
    ]
    rng = random.Random(seed)
    rng.shuffle(board_pairs)

    num_boards = len(board_pairs)
    deception_first_list: list[bool] = []
    for board_idx in range(num_boards):
        pair_rng = random.Random(f"{seed}_{board_idx}")
        deception_first_list.append(pair_rng.choice([True, False]))

    schedule: list[dict] = []
    for slot in range(num_boards * 2):
        board_idx = slot % num_boards
        is_second = slot >= num_boards
        board_data, board_type = board_pairs[board_idx]

        deception_first = deception_first_list[board_idx]
        deception = (not deception_first) if is_second else deception_first

        config = "config1" if slot % 2 == 0 else "config2"

        used_board = board_data
        if is_second:
            used_board = _augment_board(board_data, f"{seed}_{board_idx}_aug")

        schedule.append(
            {
                "board": used_board,
                "deception": deception,
                "config": config,
                "board_type": board_type,
            }
        )

    human_p0 = Human(0, debug=debug)
    human_p1 = Human(1, debug=debug)
    llm_p0 = create_agent(llm_type, 0, debug=debug, verbose=verbose)
    llm_p1 = create_agent(llm_type, 1, debug=debug, verbose=verbose)

    output_dir = os.path.join("results", "panel_results", llm_type, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    boards_used_path = os.path.join(output_dir, "boards_used.json")
    with open(boards_used_path, "w") as f:
        json.dump(
            {
                "normal": normal_boards,
                "advantage": advantage_boards,
                "schedule": schedule,
            },
            f,
            indent=2,
        )
    print(
        f"Saved {len(normal_boards)} normal + {len(advantage_boards)} advantage "
        f"base boards + {len(schedule)} full board configs to {boards_used_path}\n"
    )

    deltas_human: list[float] = []
    deltas_llm: list[float] = []
    results_by_cell: dict[tuple, list[float]] = {}
    game = Game_ct()
    for game_idx, entry in enumerate(schedule):
        board_data = entry["board"]
        deception_val = entry["deception"]
        config = entry["config"]
        board_type = entry["board_type"]

        if config == "config1":
            agent_init, agent_resp = human_p0, llm_p1
            human_is_p0 = True
        else:
            agent_init, agent_resp = human_p1, llm_p0
            human_is_p0 = False

        decept_label = "dec" if deception_val else "nodec"
        print(f"\n========== Game {game_idx + 1}/{len(schedule)} ==========")

        label = f"{board_type}_{decept_label}_{config}"
        path = os.path.join(output_dir, f"game_{game_idx + 1}_{label}.json")

        if os.path.exists(path):
            print("  -> Game already completed; loading deltas from existing result")
            try:
                existing = GameHistory.load(path)
                if existing.outcome is not None and "deltas" in existing.outcome:
                    delta_0 = existing.outcome.deltas[0]
                    delta_1 = existing.outcome.deltas[1]
                    if human_is_p0:
                        human_delta, llm_delta = delta_0, delta_1
                    else:
                        llm_delta, human_delta = delta_0, delta_1
                    deltas_human.append(human_delta)
                    deltas_llm.append(llm_delta)
                    cell_key = (deception_val, board_type)
                    results_by_cell.setdefault(cell_key, []).append(human_delta)
                    continue
                else:
                    raise ValueError("missing deltas in outcome")
            except Exception as e:
                print(f"  -> Corrupt result file ({e}); re-playing game")
                os.remove(path)

        game.load_setting(
            board_data["board"],
            board_data["chips1"],
            board_data["chips2"],
            board_data["location1"],
            board_data["location2"],
        )

        history = GameHistory.from_game(
            game,
            p0_type="human" if human_is_p0 else llm_type,
            p1_type=llm_type if human_is_p0 else "human",
            initiator=0 if human_is_p0 else 1,
            board_index=game_idx + 1,
            config=config,
        )

        result = run_single_human_game(
            game,
            agent_init,
            agent_resp,
            history=history,
            chat_history=True,
            deception=deception_val,
        )

        if result is None:
            print("  -> Error, skipping — game not saved (re-run to retry)")
            continue

        history.save(path)

        delta_0, delta_1, _, _ = result
        if human_is_p0:
            human_delta, llm_delta = delta_0, delta_1
        else:
            llm_delta, human_delta = delta_0, delta_1

        deltas_human.append(human_delta)
        deltas_llm.append(llm_delta)

        cell_key = (deception_val, board_type)
        results_by_cell.setdefault(cell_key, []).append(human_delta)

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print(f"PANEL RESULTS — {llm_type} (seed={seed})")
    print(f"{'=' * 80}")
    print(f"{'':>20} {'Deception ON':>15} {'Deception OFF':>15}")
    print(f"{'-' * 50}")

    for btype in ("normal", "advantage"):
        row = f"{btype.capitalize()} boards"
        for dec in (True, False):
            vals = results_by_cell.get((dec, btype), [])
            avg = sum(vals) / len(vals) if vals else float("nan")
            row += f"  {avg:>+8.1f}"
        print(row)

    print(f"{'-' * 50}")
    row_all = f"{'Overall':>20}"
    for dec in (True, False):
        vals = [
            v for (d, bt), vlist in results_by_cell.items() if d == dec for v in vlist
        ]
        avg = sum(vals) / len(vals) if vals else float("nan")
        row_all += f"  {avg:>+8.1f}"
    print(row_all)
    print()

    h_avg = sum(deltas_human) / len(deltas_human) if deltas_human else float("nan")
    l_avg = sum(deltas_llm) / len(deltas_llm) if deltas_llm else float("nan")
    print(f"Human avg delta ({len(deltas_human)} games): {h_avg:+.1f}")
    print(f"LLM avg delta   ({len(deltas_llm)} games): {l_avg:+.1f}")

    delta_path = os.path.join(output_dir, "delta_scores.json")
    with open(delta_path, "w") as f:
        json.dump({"human": deltas_human, "llm": deltas_llm}, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


# MAIN EXECUTION


def main(args):
    """Initializes the agents, runs the games, calls the post-hoc analyses and saves the results."""

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
    p0_c1 = create_agent(
        args.player_0, 0, args.learning_speed, args.debug, args.verbose
    )
    p1_c1 = create_agent(
        args.player_1, 1, args.learning_speed, args.debug, args.verbose
    )
    # shadows_c1 = {f"tom{i}": ShadowAgent(i, 0) for i in range(3)} | {"random": RandomShadowAgent(0)}
    shadows_c1 = {}  # Run without shadow agent tracking

    p0_c2 = create_agent(
        args.player_1, 0, args.learning_speed, args.debug, args.verbose
    )
    p1_c2 = create_agent(
        args.player_0, 1, args.learning_speed, args.debug, args.verbose
    )
    # shadows_c2 = {f"tom{i}": ShadowAgent(i, 0) for i in range(3)} | {"random": RandomShadowAgent(0)}
    shadows_c2 = {}  # Run without shadow agent tracking

    agents_tuple = (p0_c1, p1_c1, p0_c2, p1_c2)
    shadows_tuple = (shadows_c1, shadows_c2)

    human_in_matchup = isinstance(p0_c1, Human) or isinstance(p1_c1, Human)

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

        if human_in_matchup:
            if len(boards_data) % 2 == 1:
                print(
                    f"  -> Human tournament has an odd number of boards "
                    f"({len(boards_data)}); ignoring the last one."
                )
            paired = []
            for k in range(0, len(boards_data) - 1, 2):
                paired.append((k, boards_data[k], boards_data[k + 1]))

            for i, b1, b2 in paired:
                if i < start_index:
                    continue

                # Config 1 on b1
                print(
                    f"\n========== Board {i + 1}/{len(paired) * 2} | Config 1: {args.player_0} vs {args.player_1} =========="
                )
                game.load_setting(
                    b1["board"],
                    b1["chips1"],
                    b1["chips2"],
                    b1["location1"],
                    b1["location2"],
                )
                history_c1 = GameHistory.from_game(
                    game,
                    p0_type=args.player_0,
                    p1_type=args.player_1,
                    initiator=0,
                    board_index=i + 1,
                    config="config1",
                )
                result_c1 = run_single_human_game(
                    game,
                    p0_c1,
                    p1_c1,
                    history=history_c1,
                    chat_history=args.full_chat_history,
                )
                path_c1 = os.path.join(dir_games, f"board_{i + 1}_c1.json")
                if os.path.exists(path_c1):
                    print(f"  -> Skipping save (already exists): {path_c1}")
                else:
                    history_c1.save(path_c1)

                # Config 2 on b2 (different board, swapped initiator)
                print(
                    f"\n========== Board {i + 2}/{len(paired) * 2} | Config 2: {args.player_1} vs {args.player_0} =========="
                )
                game.load_setting(
                    b2["board"],
                    b2["chips1"],
                    b2["chips2"],
                    b2["location1"],
                    b2["location2"],
                )
                history_c2 = GameHistory.from_game(
                    game,
                    p0_type=args.player_1,
                    p1_type=args.player_0,
                    initiator=1,
                    board_index=i + 2,
                    config="config2",
                )
                result_c2 = run_single_human_game(
                    game,
                    p0_c2,
                    p1_c2,
                    history=history_c2,
                    chat_history=args.full_chat_history,
                )
                path_c2 = os.path.join(dir_games, f"board_{i + 2}_c2.json")
                if os.path.exists(path_c2):
                    print(f"  -> Skipping save (already exists): {path_c2}")
                else:
                    history_c2.save(path_c2)

                if result_c1 is None or result_c2 is None:
                    skipped_boards += 1
                    print(f"  -> Boards {i + 1}/{i + 2} excluded from results.")
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
                    i + 2,
                )

                ckpt_data = {
                    "board_idx": i + 1,
                    "skipped_boards": skipped_boards,
                    "stats": stats,
                    "delta_scores": deltas,
                    "evidence_matrix": evidence_matrix,
                }
                save_checkpoint(checkpoint_file, ckpt_data, agents_tuple, shadows_tuple)
        else:
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
                    game,
                    p0_c1,
                    p1_c1,
                    history=history_c1,
                    shadow_agents=shadows_c1,
                    tracked_player=0,
                    chat_history=args.full_chat_history,
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
                    game,
                    p0_c2,
                    p1_c2,
                    history=history_c2,
                    shadow_agents=shadows_c2,
                    tracked_player=1,
                    chat_history=args.full_chat_history,
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
    parser = argparse.ArgumentParser(description="Run Colored Trails.")
    parser.add_argument(
        "--panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run panel experiment (Human vs LLM). Use --no-panel for tournament mode.",
    )
    parser.add_argument(
        "player_0",
        type=str,
        nargs="?",
        default=None,
        help="LLM type in panel mode (e.g. claude, qwen); first player in tournament mode.",
    )
    parser.add_argument(
        "player_1",
        type=str,
        nargs="?",
        default=None,
        help="Second player (tournament mode).",
    )
    parser.add_argument(
        "--boards-file",
        type=str,
        default="boards_20.json",
        help="Path to the JSON file with pre-generated boards.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--full-chat-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--learning-speed", type=float, default=0.8)
    parser.add_argument(
        "--normal-boards-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--advantage-boards-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--advantage-threshold",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--advantage-boards-file",
        type=str,
        default="board_generation/tom2_vs_tom1_advantage_mirrored_100.json",
        help="JSON file with pre-generated advantage boards. "
        "If it exists and has enough boards, "
        "they are loaded instead of being generated on-the-fly. "
        "When missing, boards are generated and saved to this path.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print retry attempts and token counts for OpenRouter agent.",
    )
    cli_args = parser.parse_args()

    if cli_args.panel:
        if not cli_args.player_0:
            parser.error("player_0 (LLM type) is required in panel mode")
        run_panel_experiment(
            cli_args.player_0,
            seed=cli_args.seed,
            normal_boards_count=cli_args.normal_boards_count,
            advantage_boards_count=cli_args.advantage_boards_count,
            advantage_threshold=cli_args.advantage_threshold,
            advantage_boards_file=cli_args.advantage_boards_file,
            boards_file=cli_args.boards_file,
            debug=cli_args.debug,
            verbose=cli_args.verbose,
        )
    else:
        if not cli_args.player_0 or not cli_args.player_1:
            parser.error("player_0 and player_1 are required in tournament mode")
        main(cli_args)
