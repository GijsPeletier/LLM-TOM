import numpy as np
from agents.tom_agent import Agent_ct
from game.colored_trails import Game_ct


class ShadowAgent(Agent_ct):
    """
    A passive observer agent used for model fitting and RFX-BMS evaluation.
    Inherits all Theory of Mind (ToM) logic from Agent_ct but adds probabilistic
    evaluation methods.
    """

    def __init__(self, order, player_id, learning_speed=0.8, debug=False):
        # Initialize the parent Agent_ct class
        # We default debug to False so shadow agents don't explode the terminal
        super().__init__(order, player_id, learning_speed=learning_speed, debug=debug)

    def get_value(self, offer_to_me):
        """
        The Master Evaluation Function (Shadow Override).

        This forces the Shadow Agent to run the full, recursive ToM simulation
        for ALL offers, even negative ones. This removes the asymmetric heuristic
        shortcut from the base agent, ensuring a mathematically pure Softmax gradient.
        """
        current_utility = self.game_to_play.utility_function[self.loc][self.game_to_play.chip_sets[self.player_id]]
        offer_utility = self.game_to_play.utility_function[self.loc][offer_to_me]
        utility_gain = offer_utility - current_utility

        # Note: we have deleted the `if utility_gain <= 0` shortcut here
        # The shadow agent will now simulate the opponent's response to terrible offers too.
        if self.order == 0:
            # Zero-order agents evaluate purely on frequency statistics.
            return self.opponent_model.get_expected_value(offer_to_me, utility_gain)

        # Higher-order agents use recursion to compute the expected value.
        current_value = 0
        if self.confidence_locked or self.confidence > 0:
            # Store beliefs before simulating the opponent's response.
            self.opponent_model.save_beliefs()

            # Send the hypothetical (flipped) offer to the opponent model.
            self.opponent_model.receive_offer(self.game_to_play.flip_array[offer_to_me])

            # Calculate the uncertain value by iterating over all assumed goal locations.
            for l in range(len(self.location_beliefs)):
                if self.location_beliefs[l] > 0:
                    self.opponent_model.set_location(l)
                    current_value += self.location_beliefs[l] * self.get_location_value(offer_to_me)

            # Restore beliefs after simulation
            self.opponent_model.restore_beliefs()

            if self.confidence_locked or self.confidence >= 1:
                return current_value

        # Blend calculated value with self-model value based on confidence.
        return self.confidence * current_value + (1 - self.confidence) * self.self_model.get_value(offer_to_me)

    def get_evs(self, chosen_offer, incoming_offer=None):
        """
        Calculates and returns the expected values (EVs) for all possible offers.
        """
        num_possible_offers = len(self.game_to_play.utility_function[0])
        offer_evs = []

        accept_gain = None
        if incoming_offer is not None:
            current_util = self.game_to_play.utility_function[self.loc][self.game_to_play.chip_sets[self.player_id]]
            accept_util = self.game_to_play.utility_function[self.loc][incoming_offer]
            accept_gain = accept_util - current_util

        for offer_code in range(num_possible_offers):
            if incoming_offer is not None and offer_code == incoming_offer:
                offer_evs.append(accept_gain)
            else:
                offer_evs.append(self.get_value(offer_code))

        own_chips = self.game_to_play.chip_sets[self.player_id]
        offer_evs[own_chips] = max(0.0, offer_evs[own_chips])

        # Return just the chosen index and raw EVs
        return chosen_offer, np.array(offer_evs).tolist()


class RandomShadowAgent(Agent_ct):
    """
    A Null Model baseline that assumes the agent plays completely randomly.
    Assigns equal probability to all legally valid moves on the board.
    Used to calculate McFadden's Pseudo-R^2.
    """

    def __init__(self, player_id, debug=False):
        # We pass order=0 just to satisfy the parent class, but it won't actually do any ToM math.
        super().__init__(order=0, player_id=player_id, debug=debug)

    def get_evs(self, chosen_offer, incoming_offer=None):
        # Random agent doesn't need EVs, just flag the choice
        return chosen_offer, None