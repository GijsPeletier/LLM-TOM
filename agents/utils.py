"""
utils.py
Shared utility functions for LLM agents in Colored Trails.
"""

import json
import re
from game.colored_trails import Game_ct

MAX_MESSAGE_LENGTH = 200

CHAT_HISTORY_MAX_MESSAGES = 50

DECEPTION_PROMPT = """

Note that it is allowed to deceive your opponent, and sometimes it is beneficial. Your opponent might infer your goal location based on the proposal you make. Consider which belief you want your opponent to hold, and how it serves your strategy."""


def format_board(board, goal_r, goal_c):
    """Formats the board with row/col indices and color names."""
    color_map = {0: "White", 1: "Black", 2: "Magenta", 3: "Gray", 4: "Yellow"}
    header = "     " + "   ".join([f"{c:^8}" for c in range(5)])
    lines = [header]

    for r in range(5):
        row_cells = []
        for c in range(5):
            tile_color_idx = board[r][c]
            color_name = color_map.get(tile_color_idx, "Unknown")
            if r == 2 and c == 2:
                cell_str = "[Start]"
            else:
                cell_str = color_name
            if r == goal_r and c == goal_c:
                cell_str = f"*{cell_str}*"
            row_cells.append(f"{cell_str:^8}")
        lines.append(f"{r}:  " + "   ".join(row_cells))
    return "\n".join(lines)


def format_chips(chip_list):
    """Converts list [1, 0, 2...] into string 'White:1, Black:0, Magenta:2...'"""
    color_names = ["White", "Black", "Magenta", "Gray", "Yellow"]
    parts = []
    for i, count in enumerate(chip_list):
        parts.append(f"{color_names[i]}:{count}")
    return ", ".join(parts)


def parse_response(text):
    """Extracts and parses the JSON object from the LLM's response text. Returns None on failure."""
    try:
        # First attempt: Look for JSON explicitly fenced in markdown blocks (most reliable)
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return json.loads(match.group(1))

        # Second attempt: Look for a JSON object that specifically contains our required "action" key.
        # This prevents matching random {1 Yellow} chip references in the reasoning text.
        match_fallback = re.search(
            r'\{\s*"action"\s*:.*\}', text, re.DOTALL | re.IGNORECASE
        )
        if match_fallback:
            return json.loads(match_fallback.group(0))

    except Exception:
        pass

    print(f"Parse failure on text: {text}")
    return None


def extract_thoughts(raw_response: str) -> str | None:
    """Extract the chain-of-thought text that appears before the JSON block.

    If the response contains a ```json fence, returns everything before it.
    Otherwise, returns the full stripped response (the whole text is reasoning).
    Returns None if the response is empty.
    """
    if not raw_response:
        return None
    match = re.search(r"```(?:json)?", raw_response, re.IGNORECASE)
    if match:
        return raw_response[: match.start()].strip() or None
    return raw_response.strip() or None


def _format_chat_history_block(chat_log, fallback_incoming_message=None):
    """Render the chat-history block for the LLM prompt.

    `chat_log` is a list of `{"round": int, "speaker": "you"|"opponent", "text": str|None}`
    entries in chronological order. Capped to the most recent
    `CHAT_HISTORY_MAX_MESSAGES` entries.

    Falls back to a single-message block (legacy behavior) when `chat_log` is
    empty/None and `fallback_incoming_message` is provided.
    """
    if chat_log:
        trimmed = chat_log[-CHAT_HISTORY_MAX_MESSAGES:]
        lines = []
        any_text = False
        for entry in trimmed:
            text = (entry.get("text") or "").strip()
            if text:
                any_text = True
                lines.append(f'- [R{entry["round"]}, {entry["speaker"]}] "{text}"')
            else:
                lines.append(f"- [R{entry['round']}, {entry['speaker']}] (no message)")
        if not any_text:
            return ""
        return "CHAT HISTORY WITH OPPONENT:\n" + "\n".join(lines) + "\n"

    if fallback_incoming_message is not None and str(fallback_incoming_message).strip():
        return (
            f"PREVIOUS MESSAGE FROM OPPONENT:\n"
            f'"{str(fallback_incoming_message).strip()}"\n'
        )
    return ""


def construct_prompt(
    player_id,
    game,
    history,
    offer_received,
    incoming_message=None,
    chat_log=None,
    deception=False,
):
    """Constructs the prompt sent to the LLM based on the current game state.

    `chat_log` (preferred) is a list of `{"round", "speaker", "text"}` dicts in
    chronological order representing every free-text message exchanged so far
    in the negotiation (both directions). When provided, it supersedes the
    legacy single-message `incoming_message`.
    """
    my_goal_idx = game.locations[player_id]
    goal_r, goal_c = [
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
    ][my_goal_idx]

    my_chips = Game_ct.convert_code(game.chip_sets[player_id], game.bin_max)
    opp_id = 1 - player_id
    opp_chips = Game_ct.convert_code(game.chip_sets[opp_id], game.bin_max)
    total_pool = [m + o for m, o in zip(my_chips, opp_chips)]

    board_str = format_board(game.board, goal_r, goal_c)

    if not history:
        history_str = "None (Start of negotiation)"
    else:
        history_str = "\n".join([f"- {entry}" for entry in history])

    message_block = _format_chat_history_block(chat_log, incoming_message)

    if offer_received is None:
        status_line = "It is your turn to start the negotiation."
        allowed_actions = '"propose" or "withdraw"'
    else:
        offer_vec = Game_ct.convert_code(offer_received, game.bin_max)
        offer_str = format_chips(offer_vec)
        status_line = (
            f"The current offer on the table (what YOU will hold) is: {offer_str}. "
            f"\nYou can either propose a counter-offer, accept the offer, or withdraw from negotiations completely."
        )
        allowed_actions = '"propose", "accept", or "withdraw"'

    prompt = f"""You are Player {player_id} in Colored Trails.

BOARD (row,col with 0,0 at top-left, [Start]=your position, *Goal*=your goal):

{board_str}

STARTING INFORMATION:
- Goal: ({goal_r}, {goal_c})
- Current position: [Start] (2, 2)
- Your chips:       {format_chips(my_chips)}
- Opponent chips:   {format_chips(opp_chips)}
- Total chip pool:  {format_chips(total_pool)}

GAME RULES:
- You can only move to adjacent tiles (up/down/left/right, not diagonally).
- You start at [Start] (2, 2). Leaving the starting tile is free (no chip needed).
- Moving onto any other tile costs 1 chip matching that tile's color.
- You DO need a chip to step onto the goal tile.
- Scoring: +100 per step toward goal, +500 for reaching goal, +50 per unused chip.

PROPOSAL RULES:
- You must propose a redistribution of the EXISTING chips (Your Hand + Opponent's Hand). No new chips can be created.
- The numbers you propose are the TOTAL chips you will hold — NOT chips added on top of what you already have.
- The "Total chip pool" is finite. For each color, you cannot request more than (Your Current Chips + Opponent's Current Chips).
- The opponent automatically receives whatever you do not take.
- If the opponent ACCEPTS your proposal, the negotiation ENDS IMMEDIATELY.
- If the opponent REJECTS, it becomes their turn to propose.

NEGOTIATION HISTORY:
{history_str}

{message_block}YOUR TASK:
{status_line}

Think step by step before deciding, and then output a JSON object with your move:
- "action": {allowed_actions}
- "offer": [w, b, m, g, y] (List of 5 integers. The count of White, Black, Magenta, Gray, Yellow chips YOU will hold.
- "message": "A short free-text message (max {MAX_MESSAGE_LENGTH} chars) that will be sent to your opponent along with this offer. You may leave it empty."
- "reasoning": "SHORT Strategy explanation"{DECEPTION_PROMPT if deception else ""}
"""
    return prompt

