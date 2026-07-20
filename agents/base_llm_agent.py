from game.colored_trails import Game_ct
from game.history import NegotiationMessage
from agents.utils import (
    format_chips,
    parse_response,
    construct_prompt,
    extract_thoughts,
    MAX_MESSAGE_LENGTH,
    CHAT_HISTORY_MAX_MESSAGES,
)


class BaseLLMAgent:
    def __init__(self, player_id, debug=False):
        self.player_id = player_id
        self.game = None
        self.history = []
        self.order = "LLM"
        self.confidence = 1.0
        self.last_opponent_message = None
        self.chat_log: list[dict] = []
        self.debug = debug
        self.deception = False

    def init(self, game):
        self.game = game
        self.history = []
        self.last_opponent_message = None
        self.chat_log = []
        self.deception = False

    def make_offer(
        self,
        offer_received=None,
        round_idx=0,
        role="initiator",
        incoming_message=None,
        chat_log=None,
    ):
        if self.game is None:
            raise ValueError("Agent not initialized with a game instance.")

        if incoming_message is not None and str(incoming_message).strip():
            self.last_opponent_message = str(incoming_message).strip()

        if chat_log is not None:
            self.chat_log = list(chat_log)
        else:
            if self.last_opponent_message is not None:
                self.chat_log.append(
                    {
                        "round": round_idx,
                        "speaker": "opponent",
                        "text": self.last_opponent_message,
                    }
                )

        if offer_received is not None:
            opp_offer_str = format_chips(
                Game_ct.convert_code(offer_received, self.game.bin_max)
            )
            prefix = (
                "Rejected your offer and Counter-Proposed"
                if any(h.startswith("You:") for h in self.history)
                else "Proposes"
            )
            self.history.append(f"Opponent: {prefix} {opp_offer_str}")

        prompt = construct_prompt(
            self.player_id,
            self.game,
            self.history,
            offer_received,
            incoming_message=self.last_opponent_message,
            chat_log=self.chat_log,
            deception=self.deception,
        )

        # This calls the method defined in the child classes (API or HF)
        response_text = self._generate_llm_response(prompt)

        if self.debug:
            print(f"  [Debug] Prompt: {prompt}")
            print(f"  [Debug] Raw Response: {response_text}")

        response_data = parse_response(response_text)

        if response_data is None:
            raise ValueError(f"P{self.player_id}: Failed to parse LLM response.")
        action = response_data.get("action", "withdraw").lower()
        reasoning = response_data.get("reasoning")
        chat_message = response_data.get("message")
        if chat_message is not None:
            chat_message = str(chat_message).strip()
            if len(chat_message) > MAX_MESSAGE_LENGTH:
                chat_message = chat_message[:MAX_MESSAGE_LENGTH]
            if chat_message == "":
                chat_message = None

        if self.debug:
            print(
                f"  [Debug] Parsed Response: Action: {action}, Reason: {reasoning if reasoning else 'N/A'}"
            )
            if chat_message:
                print(f"  [Debug] Message: {chat_message}")

        if action == "accept":
            if offer_received is None:
                raise ValueError(
                    f"P{self.player_id}: Tried to accept on the opening turn."
                )
            offer_code = offer_received
            offer_list = Game_ct.convert_code(offer_code, self.game.bin_max)
        elif action == "propose":
            my_offer_list = response_data.get("offer", [])
            if len(my_offer_list) == 5 and all(
                0 <= my_offer_list[i] <= self.game.bin_max[i] for i in range(5)
            ):
                self.history.append(f"You: Propose {format_chips(my_offer_list)}")
                offer_code = Game_ct.convert_chips(my_offer_list, self.game.bin_max)
                offer_list = my_offer_list
            else:
                raise ValueError(
                    f"P{self.player_id}: Invalid proposal format/bounds: {my_offer_list}."
                )
        elif action == "withdraw":
            offer_code = self.game.chip_sets[self.player_id]
            offer_list = Game_ct.convert_code(offer_code, self.game.bin_max)
        else:
            raise ValueError(f"P{self.player_id}: Unknown action '{action}'.")

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
            action=action,
            offer=offer_list,
            message=chat_message,
            reasoning=reasoning,
            thoughts=extract_thoughts(response_text),
            raw_response=response_text,
            prompt=prompt,
        )
        return offer_code, message

    def _generate_llm_response(self, prompt):
        """Must be implemented by child classes."""
        raise NotImplementedError("This method should be overridden by subclass")
