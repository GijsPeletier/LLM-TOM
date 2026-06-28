"""
tom_agent.py

This module implements Theory-of-Mind (ToM) agents for the Colored Trails negotiation game.

It contains two primary classes:
1. Tom0_model_ct: A 0th-order model that contains frequency statistics.
2. Agent_ct: A recursive ToM agent (Order k) that models the mental state of other agents
   by simulating their decision-making process using Order k-1 models.

Author:
    Peter van den Bempt
"""

import random

# Ensure these modules are in your PYTHONPATH
from game.colored_trails import Game_ct
from game.history import NegotiationMessage

# GLOBAL CONFIGURATION
GLOBAL_LEARNING_SPEED = 0.8     # default learning speed value
GLOBAL_PRECISION = 0.00001      # floating point tolerance for comparing utility values
NEGOTIATION_COST = 1            # utility penalty applied per round of negotiation

# PRIOR BELIEFS
# These matrices represent the pre-trained beliefs of the agents.
# They contain the knowledge that generous offers are more likely to be accepted.
# These values were obtained by letting two order-zero agents play each other for 200 games.

INITIATOR_PRIORS_OBSERVED = [
    [15, 22, 5, 5, 5],
    [31, 13, 18, 5, 5],
    [13, 25, 10, 18, 5],
    [19, 13, 20, 8, 5],
    [25, 18, 13, 9, 5],
]

INITIATOR_PRIORS_TOTAL = [
    [36, 62, 16, 22, 30],
    [69, 45, 50, 16, 22],
    [13, 57, 60, 49, 16],
    [19, 13, 31, 31, 6],
    [25, 18, 13, 9, 7],
]

RESPONDER_PRIORS_OBSERVED = [
    [20, 38, 5, 5, 5],
    [35, 12, 36, 5, 5],
    [16, 28, 7, 16, 5],
    [22, 16, 28, 6, 5],
    [30, 22, 16, 6, 5],
]

RESPONDER_PRIORS_TOTAL = [
    [35, 80, 13, 19, 25],
    [54, 25, 76, 13, 18],
    [16, 49, 15, 33, 13],
    [22, 16, 41, 12, 9],
    [30, 22, 16, 6, 5],
]


class Tom0_model_ct:
    """
    Zero-order Theory of Mind model (ToM-0) for Colored Trails.

    This class maintains a probabilistic model of:
    1. Offer Type Beliefs: How likely an offer is to be accepted based on chip surplus/deficit.
    2. Color Beliefs: Specific preferences for certain colored chips, learned during a game.
    """
    player_id = 0           # ID used to refer to the player
    loc = 0                 # pre-initialization goal location
    saved_beliefs = []      # empty list for later belief storing
    save_count = 0          # belief saving tracker
    belief_offer = []       # list of belief values for every possible offer
    game_to_play = None     # pre-initialization game state

    def __init__(self, player_id, learning_speed=GLOBAL_LEARNING_SPEED):
        """
        Initialize the ToM-0 model with prior statistical beliefs.

        Args:
            player_id (int): The ID (0 or 1) of the agent.
            learning_speed (float): Factor [0,1] determining how fast beliefs change.
        """
        self.player_id = player_id
        self.learning_speed = learning_speed
        self.count_beliefs = [[5 for i in range(5)] for j in range(5)]  # the number of times a type was accepted
        self.count_total = [[5 for i in range(5)] for j in range(5)]    # the number of times a type was observed

        # Apply the pre-calculated priors
        for r in range(5):
            for c in range(5):
                if self.player_id == 0:
                    self.count_beliefs[r][c] = INITIATOR_PRIORS_OBSERVED[r][c]
                    self.count_total[r][c] = INITIATOR_PRIORS_TOTAL[r][c]
                else:
                    self.count_beliefs[r][c] = RESPONDER_PRIORS_OBSERVED[r][c]
                    self.count_total[r][c] = RESPONDER_PRIORS_TOTAL[r][c]

    def init(self, game, player_id=None):
        """
        Reset the ToM-0 model for a new game instance. Type beliefs are preserved.

        Args:
            game (Game_ct): The current game board and state.
            player_id (int, optional): Update the player ID if necessary.
        """
        if (player_id is not None):
            self.player_id = player_id
        self.belief_offer = []
        self.game_to_play = game

        # Initialize the  color beliefs using the preserved type beliefs.
        # This converts the abstract (pos, neg) priors into specific beliefs about
        # every possible chip set combination in the current game.
        for i in range(len(game.utility_function[0])):
            self.belief_offer.append(self.get_acceptance_rate(i))
        self.saved_beliefs = []

    def get_state(self):
        """Export persistent beliefs (type beliefs) to a dictionary."""
        return {
            "count_beliefs": [row[:] for row in self.count_beliefs],
            "count_total": [row[:] for row in self.count_total]
        }

    def load_state(self, state):
        """Restore persistent beliefs from a dictionary."""
        if state:
            self.count_beliefs = state.get("count_beliefs", self.count_beliefs)
            self.count_total = state.get("count_total", self.count_total)

    def save_beliefs(self):
        """Snapshot current beliefs (used during recursive simulation)."""
        self.saved_beliefs.append(self.belief_offer.copy())

    def restore_beliefs(self):
        """Restore beliefs to the previous snapshot."""
        self.belief_offer = self.saved_beliefs.pop()

    def observe(self, offer, is_accepted, player_id):
        """
        Update beliefs based on an observed action (Offer made, Accepted, or Rejected).

        This is the core learning loop for ToM-0.

        Args:
            offer (int): The code of the chip set involved.
            is_accepted (bool): True if the offer resulted in agreement.
            player_id (int): The player who performed the action.
        """
        # Calculate the structure of the offer, pos = chips gained, neg = chips given.
        pos = 0
        neg = 0
        diff = Game_ct.get_chip_difference(self.game_to_play.chip_sets[self.player_id], offer,
                                           self.game_to_play.bin_max)
        for i in range(len(diff)):
            if (diff[i] > 0):
                pos += diff[i]
            else:
                neg -= diff[i]

        if (player_id != self.player_id):
            # Implicit learning: If the opponent makes an offer, they must consider it acceptable.
            # Therefore, we increase our belief that such offers are good.
            self.increase_offer_type_belief(pos, neg)
            self.increase_color_beliefs(offer)
        elif (is_accepted):
            # Explicit learning: our offer was accepted, so we increase the relevant beliefs.
            # No use in updating color beliefs since the game will end immediately after acceptance.
            self.increase_offer_type_belief(pos, neg)
        else:
            # Explicit learning: our offer was rejected.
            # Decrease belief that this type of offer (and this specific color combination) is acceptable.
            self.decrease_offer_type_belief(pos, neg)
            self.decrease_color_beliefs(offer)

    def increase_offer_type_belief(self, pos, neg):
        """
        Reinforce the belief that an offer with (pos, neg) characteristics is acceptable.
        Increments both numerator (successes) and denominator (total encounters).
        """
        self.count_beliefs[pos][neg] += 1
        self.count_total[pos][neg] += 1

    def decrease_offer_type_belief(self, pos, neg):
        """
        Weaken the belief that an offer with (pos, neg) characteristics is acceptable.
        Increments only the denominator (total encounters), effectively lowering the ratio.
        """
        self.count_total[pos][neg] += 1

    def increase_color_beliefs(self, new_my_chips):
        """
        Update color beliefs when an offer was accepted/proposed by the opponent.

        When a chip set is accepted or proposed, the agent assumes that offers which are
        worse for the opponent are less likely to be acceptable.
        For each color dimension, potential offers that allow the agent to
        keep more chips than the accepted offer are penalized, resulting in
        a relative increase in probability for offers that are comparable
        or better for the opponent.

        Args:
            new_my_chips: The chip set code that was accepted/proposed.
        """
        # Decode offer.
        new_bins = Game_ct.convert_code(new_my_chips, self.game_to_play.bin_max)

        # Decrease all offer beliefs proportionally (resulting in a relative increase).
        for i in range(len(self.belief_offer)):
            # Decode offer.
            current_offer = Game_ct.convert_code(i, self.game_to_play.bin_max)
            for j in range(len(current_offer)):
                # Current offer allows the agent to keep more chips than the accepted offer.
                if current_offer[j] > new_bins[j]:
                    # Decrease offer belief.
                    self.belief_offer[i] *= (1 - self.learning_speed)

    def decrease_color_beliefs(self, new_my_chips):
        """
        Update color beliefs after an offer is rejected.

        When a chip set is rejected, the agent assumes that offers which are
        equal or better for itself (and thus worse for the opponent) are
        also unlikely to be acceptable. For each color dimension, potential
        offers that allow the agent to keep an equal or greater number of
        chips than the rejected offer are penalized.

        Args:
            new_my_chips: The chip set code of the rejected offer.
        """
        # Decode offer
        new_bins = Game_ct.convert_code(new_my_chips, self.game_to_play.bin_max)

        # Decrease all offer beliefs proportionally.
        for i in range(len(self.belief_offer)):
            # Decode offer.
            current_offer = Game_ct.convert_code(i, self.game_to_play.bin_max)
            for j in range(len(current_offer)):
                # Current offer allows the agent to keep the same amount or more chips than the accepted offer.
                if current_offer[j] >= new_bins[j]:
                    # Decrease offer belief.
                    # Note that this will nullify beliefs if lambda equals 1.
                    self.belief_offer[i] *= (1 - self.learning_speed)

    def get_acceptance_rate(self, offer):
        """
        Estimate the probability P(Accept) for a given offer based on the type beliefs.

        Args:
            offer (int): Encoded chip set.

        Returns:
            float: Probability between 0.0 and 1.0.
        """
        # Calculate the structure of the offer, pos = chips gained, neg = chips given.
        pos = 0
        neg = 0
        diff = Game_ct.get_chip_difference(self.game_to_play.chip_sets[self.player_id], offer,
                                           self.game_to_play.bin_max)
        for i in range(len(diff)):
            if (diff[i] > 0):
                pos += diff[i]
            else:
                neg -= diff[i]

        # We return the ratio of the values from the type belief matrices for the given type.
        return self.count_beliefs[pos][neg] / self.count_total[pos][neg]

    def get_expected_value(self, offer, utility_gain):
        """
        Calculate Expected Value (EV) directly using the learned color beliefs (zero-order reasoning).
        EV = P * G + (1 - P) * -C
        EV = P(G + C) - C
        P is the color belief value, G is the utility gain and C is the negotiation cost.
        Additional information can be found in the README file.

        Args:
            offer (int): Encoded chip set.
            utility_gain (float): The raw utility improvement if accepted.

        Returns:
            float: The risk-adjusted value of making this offer.
        """
        return self.belief_offer[offer] * (utility_gain + NEGOTIATION_COST) - NEGOTIATION_COST

    def print_beliefs_debug(self):
        """Dump belief matrices to console for debugging."""
        print(f"\n=== Zero-order Beliefs about Player {self.player_id}===")
        print("\nOffer Type Beliefs Matrix (pos/neg):")
        print("     ", end="")
        for neg in range(5):
            print(f"neg={neg:1d}  ", end="")
        print()
        for pos in range(5):
            print(f"pos={pos}: ", end="")
            for neg in range(5):
                print(f"{self.count_beliefs[pos][neg] / self.count_total[pos][neg]:.4f} ", end="")
            print()
        print(f"\nColor Beliefs:")
        for i in range(len(self.belief_offer)):
            chips = Game_ct.convert_code(i, self.game_to_play.bin_max)
            print(f"  Offer {i:3d} {chips}: {self.belief_offer[i]:.4f}")


class Agent_ct:
    """
    Recursive Theory of Mind (ToM-k) agent for Colored Trails.

    This agent simulates the reasoning process of its opponent to make decisions.
    It essentially asks itself: "What would my opponent do if I offered this?".

    If the agent's order is zero, it only holds a zero-order opponent model.
    Each higher-order agent holds an (N-1)-order opponent model, and an (N-1)-order model of itself.

    Higher-order agents also maintains beliefs about the opponent's goal location,
    and update these beliefs via Bayesian inference based on the offers received.
    """
    order = 0                       # reasoning order
    player_id = 0                   # ID used to refer to the player
    confidence_locked = False       # optional argument to fix confidence values (used for balancing different orders)
    confidence = 1.0                # initial confidence value
    location_beliefs = []           # list of belief values for each potential opponent goal location
    saved_beliefs = []              # empty list for later belief storing
    loc = 0                         # pre-initialization goal location
    last_accuracy = 0               # used to store the correctness of location beliefs based on the opponent's offers
    game_to_play = None             # pre-initialization game state

    def __init__(self, order, player_id, learning_speed=GLOBAL_LEARNING_SPEED, debug=False):
        """
        Recursively initialize the ToM hierarchy.

        Args:
            order (int): The ToM order.
            player_id (int): ID of this agent.
            learning_speed (float): Adaptation rate.
        """
        self.order = order
        self.player_id = player_id
        self.learning_speed = learning_speed
        self.debug = debug

        # RECURSIVE CONSTRUCTION:
        # An Order-N agent imagines an (N-1)-order opponent and an (N-1)-order self.
        if (order > 0):
            # Model of how the opponent thinks (1 level down).
            self.opponent_model = Agent_ct(order - 1, 1 - player_id, learning_speed)
            self.opponent_model.confidence_locked = True
            # Model of how I think (used for projecting my own future moves in simulation).
            self.self_model = Agent_ct(order - 1, player_id, learning_speed)
        else:
            # Base case: The imagined 'opponent' is a simply represented using frequency statistics.
            # In this case no simulation is performed; the acceptance rate is directly computed from the belief matrix.
            self.opponent_model = Tom0_model_ct(1 - player_id, learning_speed)
            self.self_model = None
        pass

    def init(self, game, player_id=None):
        """Initialize the agent hierarchy for a specific game board."""
        if (player_id is not None):
            self.player_id = player_id
        self.game_to_play = game                        # link the agent to the game state
        self.loc = game.locations[self.player_id]       # set the agent's goal location
        self.saved_beliefs = []

        if (self.order > 0):
            # Initialize recursive lower-order models.
            self.opponent_model.init(game, 1 - self.player_id)
            self.self_model.init(game, self.player_id)
            # Set uniform priors for location beliefs (1/12 chance for each square).
            self.location_beliefs = [1.0 / len(game.utility_function)] * len(game.utility_function)
        else:
            # Only one opponent model has to be initialized for the zero-order agent.
            self.opponent_model.init(game, 1 - self.player_id)

    def save_beliefs(self):
        """Save state of all sub-models (used before hypothetical simulations)."""
        if (self.order > 0):
            self.saved_beliefs.append(self.location_beliefs.copy())
            self.self_model.save_beliefs()
        self.opponent_model.save_beliefs()

    def restore_beliefs(self):
        """Restore state of all sub-models (after simulation is done)."""
        if (self.order > 0):
            self.location_beliefs = self.saved_beliefs.pop()
            self.self_model.restore_beliefs()
        self.opponent_model.restore_beliefs()

    def get_state(self):
        """Recursively export the agent's persistent state."""
        state = {"confidence": self.confidence}
        if self.order > 0:
            state["self_model"] = self.self_model.get_state()
            state["opponent_model"] = self.opponent_model.get_state()
        else:
            # Order 0 opponent_model is a Tom0_model_ct
            state["opponent_model"] = self.opponent_model.get_state()
        return state

    def load_state(self, state):
        """Recursively restore the agent's persistent state."""
        if not state:
            return
        self.confidence = state.get("confidence", self.confidence)
        if self.order > 0:
            self.self_model.load_state(state.get("self_model"))
            self.opponent_model.load_state(state.get("opponent_model"))
        else:
            self.opponent_model.load_state(state.get("opponent_model"))

    def get_location_beliefs(self, location):
        """
        Get the probability that the opponent is at a specific board location.
        Combines current confidence-weighted belief with lower-order model beliefs.
        """
        if (self.confidence_locked):
            return self.location_beliefs[location]
        if (self.order > 0):
            # Confidence determines how much weight is given to this agent's current belief
            # compared to the belief from the lower-order (self) model.
            return self.confidence * self.location_beliefs[location] + (
                    1 - self.confidence) * self.self_model.get_location_beliefs(location)
        return 1 / 12

    def inform_location(self):
        """Called when the true location is revealed (end of game/debug)."""
        self.loc = self.game_to_play.locations[self.player_id]
        if (self.order > 0):
            for i in range(len(self.location_beliefs)):
                self.location_beliefs[i] = 0
            self.location_beliefs[self.game_to_play.locations[1 - self.player_id]] = 1
            self.self_model.inform_location()
            self.opponent_model.inform_location()

    def get_location_value(self, offer_to_me):
        """
        Evaluate the value of an offer by simulating the opponent's response.

        This assumes the opponent is rational. We simulate the opponent receiving
        the proposal and see if they accept or counter-offer.

        Args:
            offer_to_me: The chip set being offered to me.

        Returns:
            The expected utility change.
        """
        # Ask the internal opponent model: "If I offer you X (flip of offer_to_me), what would you do?".
        response = self.opponent_model.select_offer(self.game_to_play.flip_array[offer_to_me])

        current_value = 0
        # Opponent would accept my offer.
        if (response == self.game_to_play.flip_array[offer_to_me]):
            # Calculate my hypothetical gain from this deal (hypothetical utility - initial utility).
            current_value += self.game_to_play.utility_function[self.loc][offer_to_me] - \
                             self.game_to_play.utility_function[self.loc][
                                 self.game_to_play.chip_sets[self.player_id]] - NEGOTIATION_COST
        # Opponent would make a counter-offer.
        elif (response != self.game_to_play.chip_sets[1 - self.player_id]):
            # Evaluate the utility of the counter-offer including additional negotiation cost.
            # If the counter-offer is worse than withdrawing immediately, treat the original offer as having no value.
            response = self.game_to_play.flip_array[response]
            current_value += max(-NEGOTIATION_COST, self.game_to_play.utility_function[self.loc][response] -
                                 self.game_to_play.utility_function[self.loc][
                                     self.game_to_play.chip_sets[self.player_id]] - (2 * NEGOTIATION_COST))
        return current_value

    def set_location(self, location):
        """
        Temporarily force the agent (or sub-models) to assume a specific board location.
        This is used for the agents' internal simulation of their opponent's response.
        """
        self.loc = location
        if (self.order > 0):
            # Also change the recursive inner model's locations
            self.self_model.set_location(location)
        else:
            self.opponent_model.loc = location

    def get_value(self, offer_to_me):
        """
        The Master Evaluation Function.

        Calculates the expected value of an offer. If the utility of an order > 0, this involves
        simulation: temporarily adopting the opponent's perspective,
        iterating over all possible locations they might be in, weighted by
        our belief that they are in that location.
        """
        # Calculate raw utility gain of the offer itself.
        current_utility = self.game_to_play.utility_function[self.loc][self.game_to_play.chip_sets[self.player_id]]
        offer_utility = self.game_to_play.utility_function[self.loc][offer_to_me]
        utility_gain = offer_utility - current_utility

        if (utility_gain <= 0):
            # The expected value of an offer with negative utility is directly set to minus the negotiation cost.
            return -NEGOTIATION_COST

        if (self.order == 0):
            # Zero-order agent's or models directly compute the expected value using frequency statistics.
            return self.opponent_model.get_expected_value(offer_to_me, utility_gain)

        # Higher-order agents use recursion to compute the expected value of an offer.
        current_value = 0
        if (self.confidence_locked or self.confidence > 0):
            # Store beliefs before simulating the opponent's response.
            self.opponent_model.save_beliefs()
            # Send the hypothetical (flipped) offer to the opponent model.
            self.opponent_model.receive_offer(self.game_to_play.flip_array[offer_to_me])
            # Calculate the uncertain value by iterating over all goal locations the opponent might have.
            for l in range(len(self.location_beliefs)):
                if (self.location_beliefs[l] > 0):
                    # Assume opponent is at location `l` and sum to the averaged expected value
                    self.opponent_model.set_location(l)
                    current_value += self.location_beliefs[l] * self.get_location_value(offer_to_me)

            # Restore beliefs after simulation
            self.opponent_model.restore_beliefs()

            # If the agent has full confidence in the highest order available to them, return the expected value.
            if (self.confidence_locked or self.confidence >= 1):
                return current_value

        # Blend calculated value with self-model value based on confidence.
        return self.confidence * current_value + (1 - self.confidence) * self.self_model.get_value(offer_to_me)

    def observe(self, offer, is_accepted, player_id):
        """
        Update beliefs for this agent and recursively for all sub-models.
        Updates 'confidence' based on how well the opponent's move matched expectations.
        """
        self.opponent_model.observe(offer, is_accepted, player_id)
        if (self.order > 0):
            self.self_model.observe(offer, is_accepted, player_id)
            # Update confidence if the move was made by the opponent.
            if (player_id != self.player_id and not self.confidence_locked):
                # Did the opponent make a move that generates high value for them? 
                # If so, our model is accurate and we increase confidence.
                # Likewise, confidence will decrease if the opponent's move generates low value for them.
                self.confidence = (1 - self.learning_speed) * self.confidence + self.learning_speed * (
                        max(0, self.opponent_model.get_value(offer)) + 1.0) / (
                                          max(0, self.opponent_model.get_best_value()) + 1.0)

    def select_offer(self, offer_to_me):
        """
        Tiebreaker: randomly select one of the 'best' offers (those within precision tolerance of max value).
        """
        all_offers = self.get_valid_offers(offer_to_me)
        selected = all_offers[0]

        return selected

    def make_offer(self, offer_to_me=None, round_idx=0, role="initiator", incoming_message=None, chat_log=None):
        """
        Main method to generate a move.

        Args:
            offer_to_me: The offer currently on the table (Dummy offer if initiating).
            round_idx: Current round number (for NegotiationMessage bookkeeping).
            role: "initiator" or "responder" (for NegotiationMessage bookkeeping).
            incoming_message: Free-text message from the opponent (ignored by ToM agents).
            chat_log: Full chat history (ignored by ToM agents).

        Returns:
            tuple[int, NegotiationMessage]: (encoded offer code, structured message)
        """
        if (offer_to_me == None):
            offer_to_me = self.game_to_play.chip_sets[self.player_id]

        # Only treat it as receiving an external offer if it's not my own default chip set.
        # This prevents belief updates before the game has started.
        if offer_to_me != self.game_to_play.chip_sets[self.player_id]:
            self.receive_offer(offer_to_me)

        choice = self.select_offer(offer_to_me)
        # Sends the offer to the (recursive) opponent- and self models.
        self.send_offer(choice)

        my_chips = self.game_to_play.chip_sets[self.player_id]
        if choice == my_chips:
            action = "withdraw"
        elif offer_to_me != my_chips and choice == offer_to_me:
            action = "accept"
        else:
            action = "propose"

        offer_list = Game_ct.convert_code(choice, self.game_to_play.bin_max)
        current_utility = self.game_to_play.utility_function[self.loc][my_chips]
        gain = self.game_to_play.utility_function[self.loc][choice] - current_utility

        thoughts = (
            f"ToM-{self.order} agent. "
            f"Action: {action} (gain={gain}). "
            f"Confidence: {self.confidence:.3f}."
        )

        message = NegotiationMessage(
            round=round_idx,
            player_id=self.player_id,
            role=role,
            action=action,
            offer=offer_list,
            message=None,
            reasoning=None,
            thoughts=thoughts,
            raw_response=None,
            prompt=None,
        )
        return choice, message

    def get_best_value(self):
        """Helper: find the scalar maximum expected value achievable."""
        best_value = 0
        # For all potential offers, call the master evaluation function (get_value()).
        for i in range(len(self.game_to_play.utility_function[0])):
            best_value = max(best_value, self.get_value(i))
        return best_value

    def get_valid_offers(self, offer_to_me):
        """
        Exhaustively evaluate all possible chip combinations to find optimal moves.

        Returns:
            list: A list of offer codes that all share the (approximate) highest expected value.
        """
        if self.debug:
            print(
                f"  [Debug] Evaluating offers (Current chips: {Game_ct.convert_code(self.game_to_play.chip_sets[self.player_id], self.game_to_play.bin_max)})")

        all_offers = [self.game_to_play.chip_sets[self.player_id]]
        best_value = 0
        offer_values = []

        # Evaluate every possible chip redistribution (offer).
        for i in range(len(self.game_to_play.utility_function[0])):
            value = self.get_value(i)

            # Compute offer information for debugging purposes.
            if self.debug:
                chips = Game_ct.convert_code(i, self.game_to_play.bin_max)
                current_utility = self.game_to_play.utility_function[self.loc][
                    self.game_to_play.chip_sets[self.player_id]]
                offer_utility = self.game_to_play.utility_function[self.loc][i]
                utility_gain = offer_utility - current_utility

                pos = 0
                neg = 0
                diff = Game_ct.get_chip_difference(self.game_to_play.chip_sets[self.player_id], i,
                                                   self.game_to_play.bin_max)
                for j in range(len(diff)):
                    if (diff[j] > 0):
                        pos += diff[j]
                    else:
                        neg -= diff[j]

                if self.order == 0:
                    color_belief = self.opponent_model.belief_offer[i]
                    type_belief = self.opponent_model.count_beliefs[pos][neg] / self.opponent_model.count_total[pos][
                        neg]
                    offer_values.append((value, i, chips, utility_gain, color_belief, type_belief, pos, neg))
                else:
                    offer_values.append((value, i, chips, utility_gain))

            # Keep track of the best offers (accounting for floating point precision).
            if (value > best_value - GLOBAL_PRECISION):
                if (value > best_value + GLOBAL_PRECISION):
                    all_offers = []
                    best_value = value
                all_offers.append(i)

        # Print only top 5 offers for readability.
        if self.debug:
            offer_values.sort(reverse=True, key=lambda x: x[0])
            for idx, offer_data in enumerate(offer_values[:5]):
                if self.order == 0:
                    val, i, chips, gain, color_b, type_b, pos, neg = offer_data
                    print(
                        f"    Offer {i:3d} {chips}: ExpVal={val:7.3f} | Gain={gain:4d} | ColorBelief={color_b:.4f} | TypeBelief[{pos},{neg}]={type_b:.4f}")
                else:
                    val, i, chips, gain = offer_data
                    print(f"    Offer {i:3d} {chips}: ExpVal={val:7.3f} | Gain={gain:4d}")

        # Compare best offers against the offer currently on the table.
        if (offer_to_me >= 0):
            current_offer_utility = self.game_to_play.utility_function[self.loc][offer_to_me]
            current_offer_gain = current_offer_utility - self.game_to_play.utility_function[self.loc][
                self.game_to_play.chip_sets[self.player_id]]

            if (current_offer_gain > best_value - GLOBAL_PRECISION):
                # Optionally print first, then overwrite best_value
                if self.debug:
                    print(f"  -> Accepting offer on table (gain={current_offer_gain:.3f} >= best_ev={best_value:.3f})")

                all_offers = [offer_to_me]
                best_value = current_offer_gain

        # Check for withdrawal (if best option is negative utility).
        if (best_value < GLOBAL_PRECISION):
            all_offers = [self.game_to_play.chip_sets[self.player_id]]
            if self.debug:
                print(f"  -> Withdrawing from negotiation (best_value={best_value:.3f} < precision)")

        return all_offers

    def update_location_beliefs(self, offer_received):
        """
        Perform (iterative) Bayesian inference to update where we think the opponent is.
        We iterate through every possible board location. If the offer
        makes sense for a player at that location (rationality assumption),
        belief in that location is strengthened.

        Formula: Posterior(L) = (Likelihood(O|L) * Prior(L)) / Marginal(O).

        Implementation mapping:
        1. Prior P(L): The existing `self.location_beliefs[l]`.
        2. Likelihood P(O|L): Estimated via a "rationality heuristic."
           We calculate how optimal the offer is for a specific location.
           Likelihood = (Offer Value / Best Possible Value).
        3. Marginal P(O): The `accuracy` variable, which acts as the normalizing sum.

        Args:
            offer_received (int): The offer code received from opponent.
        """
        offer_to_other = self.game_to_play.flip_array[offer_received]
        accuracy = 0
        for l in range(len(self.location_beliefs)):
            # Hypothesis: opponent has goal location 'l'.
            self.opponent_model.set_location(l)

            # Check rationality: would a player with location 'l' actually benefit from this offer?
            if (self.game_to_play.utility_function[l][offer_to_other] <= self.game_to_play.utility_function[l][
                self.game_to_play.chip_sets[1 - self.player_id]]):
                # If the offer yields <= 0 gain for the opponent with this location, it's irrational.
                # Therefore, the opponent does likely not have this location.
                self.location_beliefs[l] = 0
            else:
                # Update belief proportional to how good the offer is (Value + 1 / Max + 1 normalization).
                self.location_beliefs[l] *= max(0, (self.opponent_model.get_value(offer_to_other) + 1) / (
                        self.opponent_model.get_best_value() + 1))
                accuracy += self.location_beliefs[l]

        # Normalize beliefs to sum to 1.
        if (accuracy > 0):
            inv_sum_beliefs = 1.0 / accuracy
            for l in range(len(self.location_beliefs)):
                self.location_beliefs[l] *= inv_sum_beliefs
        else:
            # Fallback: if no location makes sense (irrational move observed), reset to uniform prior.
            self.location_beliefs = [1.0 / len(self.game_to_play.utility_function)] * len(
                self.game_to_play.utility_function)

        # Apply the confidence update.
        if (not self.confidence_locked):
            self.confidence = (1 - self.learning_speed) * self.confidence + self.learning_speed * accuracy

    def receive_offer(self, offer_to_me):
        """
        Handle an incoming offer. Triggers belief updates in all sub-models.
        """
        if (offer_to_me >= 0):
            if (self.order > 0):
                # Update opponent's location based on what they just offered.
                self.update_location_beliefs(offer_to_me)
                # The agent's lower-order model of itself also has to be informed of the incoming offer.
                self.self_model.receive_offer(offer_to_me)
                # Update opponent model (flip the offer so it looks like an offer they made).
                self.opponent_model.send_offer(self.game_to_play.flip_array[offer_to_me])
            else:
                # ToM-0 logic: receiving an offer is treated as a signal that the opponent 'likes' this offer.
                # We simulate this by telling the model it observed an acceptance.
                self.opponent_model.observe(offer_to_me, True, 1 - self.player_id)

    def send_offer(self, offer_to_me):
        """
        Handle sending an offer. Triggers state updates.
        """
        if (self.order > 0):
            # Inform lower-order self- and opponent models.
            self.self_model.send_offer(offer_to_me)
            self.opponent_model.receive_offer(self.game_to_play.flip_array[offer_to_me])
        else:
            # By default, we treat any offer that was sent as being rejected.
            # This triggers negative type belief updates. We could invert the operation if the assumption turns out
            # to be false, but the effect is insignificant because only a small fraction of the updates are spurious.
            self.opponent_model.observe(offer_to_me, False, 1 - self.player_id)
