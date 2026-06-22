"""
history.py

Hierarchical data classes for recording a complete Colored Trails game.

Domain entities (`PlayerState`, `NegotiationMessage`) are plain dataclasses —
`dataclasses.asdict()` handles `to_dict` for free, and `cls(**d)` is sufficient
for `from_dict` since their fields are all primitives. `GameHistory` is the
only class that needs an explicit `to_dict` / `from_dict` because it mixes
dataclasses with `DotMap` and has a `list[PlayerState]` field that must be
reconstructed (since `asdict` flattens it to `list[dict]`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from dotmap import DotMap

from game.colored_trails import Game_ct


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Pre-negotiation state for a single player."""

    type: str
    initial_chips: list[int]
    goal: int
    initial_utility: float


@dataclass
class NegotiationMessage:
    """A single exchange in the negotiation."""

    round: int
    player_id: int
    role: str
    action: str
    offer: Optional[list[int]] = None
    message: Optional[str] = None
    reasoning: Optional[str] = None
    thoughts: Optional[str] = None
    raw_response: Optional[str] = None
    prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level game record
# ---------------------------------------------------------------------------

@dataclass
class GameHistory:
    """Complete record of a single Colored Trails game."""

    board: DotMap
    players: list[PlayerState]
    initiator: int
    messages: list[NegotiationMessage] = field(default_factory=list)
    outcome: Optional[DotMap] = None
    meta: DotMap = field(default_factory=DotMap)

    # ── convenience accessors ───────────────────────────────────────────

    @property
    def player_0(self) -> PlayerState:
        return self.players[0]

    @property
    def player_1(self) -> PlayerState:
        return self.players[1]

    # ── construction from a loaded Game_ct ──────────────────────────────

    @classmethod
    def from_game(
        cls,
        game: Game_ct,
        p0_type: str,
        p1_type: str,
        *,
        initiator: int,
        board_index: Optional[int] = None,
        config: Optional[str] = None,
        negotiation_cost: int = 1,
    ) -> "GameHistory":
        """Build the pre-negotiation portion of a GameHistory from a loaded game.

        The returned object has empty `messages` and `outcome=None`; these are
        populated by `run_single_game()` during play.
        """
        total_chips = list(game.bin_max)
        players = [
            PlayerState(
                type=p0_type,
                initial_chips=Game_ct.convert_code(game.chip_sets[0], game.bin_max),
                goal=game.locations[0],
                initial_utility=game.utility_function[game.locations[0]][game.chip_sets[0]],
            ),
            PlayerState(
                type=p1_type,
                initial_chips=Game_ct.convert_code(game.chip_sets[1], game.bin_max),
                goal=game.locations[1],
                initial_utility=game.utility_function[game.locations[1]][game.chip_sets[1]],
            ),
        ]
        return cls(
            board=DotMap({"grid": game.board, "total_chips": total_chips}),
            players=players,
            initiator=initiator,
            meta=DotMap({
                "board_index": board_index,
                "config": config,
                "negotiation_cost": negotiation_cost,
            }),
        )

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "board": self.board.toDict(),
            "players": [asdict(p) for p in self.players],
            "initiator": self.initiator,
            "messages": [asdict(m) for m in self.messages],
            "outcome": self.outcome.toDict() if self.outcome is not None else None,
            "meta": self.meta.toDict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameHistory":
        return cls(
            board=DotMap(d["board"]),
            players=[PlayerState(**p) for p in d["players"]],
            initiator=d["initiator"],
            messages=[NegotiationMessage(**m) for m in d.get("messages", [])],
            outcome=DotMap(d["outcome"]) if d.get("outcome") else None,
            meta=DotMap(d.get("meta", {})),
        )

    def save(self, filepath: str | Path) -> None:
        """Serialize to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "GameHistory":
        """Deserialize from a JSON file."""
        with open(filepath) as f:
            return cls.from_dict(json.load(f))
