import threading
from rich.columns import Columns
from rich.console import Group
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from pynput import keyboard
from game.colored_trails import Game_ct
from game.history import NegotiationMessage
from agents.utils import MAX_MESSAGE_LENGTH, CHAT_HISTORY_MAX_MESSAGES

COLOURS = ["white", "black", "magenta", "grey", "yellow"]
MD_COLOURS = ["white", "grey7", "magenta", "grey50", "yellow"]


class Human:
    def __init__(self, player_id, debug=False) -> None:
        self.player_id = player_id
        self.offer: list[int | None] = [None] * 5
        self.received_offer = [None] * 5
        self.offer_index = 0
        self.offer_done = False
        self.incoming_message: str | None = None
        self.outgoing_message: str = ""
        self.chat_log: list[dict] = []
        self.phase: str = "offer"
        self.message_buffer: str = ""
        self.thoughts_buffer: str = ""
        self.ctrl_c_pending: bool = False
        self._dirty: threading.Event = threading.Event()
        self.debug = debug

    def init(self, game):
        self.game = game
        self.incoming_message = None
        self.outgoing_message = ""
        self.chat_log = []
        self.phase = "offer"
        self.message_buffer = ""
        self.thoughts_buffer = ""
        self.ctrl_c_pending = False
        self._dirty = threading.Event()
        self._dirty.set()

    def key_listener(self, key):
        self._handle_key(key)
        self._dirty.set()

    def _handle_key(self, key):
        if key == keyboard.Key.esc:
            if self.phase == "thoughts":
                self.thoughts_buffer = ""
                self.phase = "message"
            elif self.phase == "message":
                self.message_buffer = ""
                self.phase = "offer"
            else:
                self.offer = self.received_offer.copy()
                self.offer_index = 0
                self.phase = "offer"
            return

        if key == keyboard.Key.backspace:
            if self.phase == "offer":
                self.offer_index = max(0, self.offer_index - 1)
            elif self.phase == "thoughts":
                self.thoughts_buffer = self.thoughts_buffer[:-1]
            else:
                self.message_buffer = self.message_buffer[:-1]
            return

        if key == keyboard.Key.enter:
            if self.phase == "offer":
                self.offer = [i if i is not None else 0 for i in self.offer]
                self.phase = "message"
            elif self.phase == "message":
                self.outgoing_message = self.message_buffer[:MAX_MESSAGE_LENGTH]
                self.phase = "thoughts"
            else:
                self.phase = "done"
                self.offer_done = True
            return

        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl_c_pending = True
            self.offer_done = True
            return

        name = getattr(key, "name", None) or ""
        if name.startswith("ctrl") or name.startswith("ctrl_"):
            self.ctrl_c_pending = True
            self.offer_done = True
            return

        if self.phase == "message":
            if key == keyboard.Key.space:
                if len(self.message_buffer) < MAX_MESSAGE_LENGTH:
                    self.message_buffer += " "
                return
            for special, ch in ((keyboard.Key.tab, "\t"),):
                if key == special:
                    if len(self.message_buffer) < MAX_MESSAGE_LENGTH:
                        self.message_buffer += ch
                    return
        elif self.phase == "thoughts":
            if key == keyboard.Key.space:
                self.thoughts_buffer += " "
                return
            for special, ch in ((keyboard.Key.tab, "\t"),):
                if key == special:
                    self.thoughts_buffer += ch
                    return

        try:
            ch = key.char
        except AttributeError:
            return

        if ch is None:
            return

        if self.phase == "offer":
            if ch.isdigit():
                value = min(int(ch), self.game.bin_max[self.offer_index])
                self.offer[self.offer_index] = value
                self.offer_index = min(len(COLOURS) - 1, self.offer_index + 1)
        elif self.phase == "thoughts":
            self.thoughts_buffer += ch
        else:
            if len(self.message_buffer) < MAX_MESSAGE_LENGTH:
                self.message_buffer += ch

    def get_goal(self):
        return [
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
        ][self.game.locations[self.player_id]]

    def colour(self, string: str, colour: int) -> str:
        return f"[on {MD_COLOURS[colour]}]{string}[/on {MD_COLOURS[colour]}]"

    def _cell_rows(self, i, j, colour, goal):
        blank = self.colour("\u3000", colour)
        if (i, j) == goal:
            mid = self.colour("\u26f3", colour)
        elif (i, j) == (2, 2):
            mid = self.colour("\u274c", colour)
        else:
            mid = blank
        return [blank * 3, blank + mid + blank, blank * 3]

    def board_str(self):
        grid = self.game.board
        goal = self.get_goal()
        cell_rows = [[] for _ in range(15)]
        for i in range(5):
            for j in range(5):
                rows = self._cell_rows(i, j, grid[i][j], goal)
                for si in range(3):
                    cell_rows[i * 3 + si].append(rows[si])
        return "\n".join("".join(r) for r in cell_rows)

    def _offer_table_panel(self):
        my_chips = Game_ct.convert_code(
            self.game.chip_sets[self.player_id], self.game.bin_max
        )

        table = Table(title="")
        table.add_column("", min_width=14)
        for i, v in enumerate(COLOURS):
            table.add_column(self.colour(v, i), min_width=8, justify="center")
        table.add_section()

        table.add_row("All chips", *[str(i) for i in self.game.bin_max])
        table.add_row("Your chips", *[str(i) for i in my_chips])
        if not self.initial_offer:
            table.add_row("Received offer", *[str(i) for i in self.received_offer])
        table.add_row(
            "Your offer",
            *[
                f"[red]{v}[/red]" if self.offer_index == i else str(v)
                for i, v in enumerate(self.offer)
            ],
        )
        return table

    def _incoming_message_panel(self):
        last_opponent = None
        for entry in reversed(self.chat_log):
            if entry.get("speaker") == "opponent":
                text = (entry.get("text") or "").strip()
                if text:
                    last_opponent = text
                    break
        if last_opponent:
            body = Text(last_opponent, style="italic", overflow="fold")
        else:
            body = Text("(no message from opponent)", style="dim italic")
        return Panel(
            body,
            title="Opponent's last message",
            expand=False,
            width=73,
        )

    def _offer_phase_renderable(self):
        left_panel = Panel(
            Text().from_markup(self.board_str()), title="Board", expand=False
        )
        right_group = Group(
            self._offer_table_panel(),
            self._incoming_message_panel(),
        )
        return Columns(
            [left_panel, Align.center(right_group, vertical="top")],
            expand=True,
        )

    def _message_phase_renderable(self):
        body = (
            Text(self.message_buffer + "▌", style="bold")
            if self.message_buffer
            else Text("▌", style="bold")
        )
        caption = Text(
            f"Type your message to the opponent "
            f"({len(self.message_buffer)}/{MAX_MESSAGE_LENGTH}). "
            f"Enter to send. Esc to go back.",
            style="dim",
        )
        return Panel(
            Group(body, Text(""), caption),
            title="Your outgoing message",
            expand=True,
        )

    def _thoughts_phase_renderable(self):
        body = (
            Text(self.thoughts_buffer + "▌", style="bold")
            if self.thoughts_buffer
            else Text("▌", style="bold")
        )
        caption = Text(
            "Type your internal reasoning/beliefs. Enter to finish. Esc to go back.",
            style="dim",
        )
        return Panel(
            Group(body, Text(""), caption),
            title="Your internal thoughts",
            expand=True,
        )

    def update_offer_display(self):
        if self.phase == "message":
            return self._message_phase_renderable()
        if self.phase == "thoughts":
            return self._thoughts_phase_renderable()
        return self._offer_phase_renderable()

    def make_offer(
        self,
        received_offer,
        round_idx=0,
        role="initiator",
        incoming_message=None,
        chat_log=None,
    ):
        self.initial_offer = False
        if received_offer is None:
            self.initial_offer = True
            received_offer = self.game.chip_sets[self.player_id]
        chip_offer = Game_ct.convert_code(received_offer, self.game.bin_max)
        self.received_offer = chip_offer.copy()
        self.offer = chip_offer
        self.offer_index = 0
        if chat_log is not None:
            self.chat_log = list(chat_log)
        else:
            if incoming_message is not None and str(incoming_message).strip():
                self.chat_log.append(
                    {
                        "round": round_idx,
                        "speaker": "opponent",
                        "text": str(incoming_message).strip(),
                    }
                )
        self.incoming_message = (
            self.chat_log[-1]["text"]
            if self.chat_log and self.chat_log[-1].get("text")
            else None
        )
        self.outgoing_message = ""
        self.message_buffer = ""
        self.thoughts_buffer = ""
        self.phase = "offer"
        self.offer_done = False
        self.ctrl_c_pending = False
        self._dirty = threading.Event()
        self._dirty.set()

        with Live(
            self.update_offer_display(),
            refresh_per_second=10,
            transient=True,
            screen=False,
        ) as live:
            listener = keyboard.Listener(on_press=self.key_listener, suppress=True)
            listener.start()
            try:
                while not self.offer_done:
                    self._dirty.wait()
                    self._dirty.clear()
                    if self.offer_done:
                        break
                    live.update(self.update_offer_display())
            finally:
                listener.stop()

        if self.ctrl_c_pending:
            raise KeyboardInterrupt

        offer_code = Game_ct.convert_chips(self.offer, self.game.bin_max)
        chat_message = self.outgoing_message.strip() or None
        if chat_message:
            self.chat_log.append(
                {
                    "round": round_idx,
                    "speaker": "you",
                    "text": chat_message,
                }
            )
            if len(self.chat_log) > CHAT_HISTORY_MAX_MESSAGES:
                self.chat_log = self.chat_log[-CHAT_HISTORY_MAX_MESSAGES:]
        message = NegotiationMessage(
            round=round_idx,
            player_id=self.player_id,
            role=role,
            action="propose",
            offer=list(self.offer),
            message=chat_message,
            reasoning=None,
            thoughts=self.thoughts_buffer.strip() or None,
            raw_response=None,
            prompt=None,
        )
        return offer_code, message
