"""
colored_trails.py

This module implements the core logic for the Colored Trails game environment.

It handles:
1.  Board generation (random colors, start/goal locations).
2.  Chip distribution and management.
3.  State encoding: Converting sets of chips into unique integer IDs for efficient lookup.
4.  Utility calculation: Determining the value of a position based on held chips and
    distance to the goal using a dynamic programming approach.

Author:
    Peter van den Bempt
"""

import random


class Game_ct:
    number_of_colors = 5  # number of allowed, distinct colors on the board
    number_of_chips_per_player = 4  # initial number of chips per player
    locations = []  # stores the players' goal locations
    chips = []  # stores the players' chip sets (e.g., [[1,0,3,0,0], ...])
    chip_sets = []  # encoded representation (IDs) of the players' chip sets
    flip_array = []  # lookup table to invert a chip set ID (maps 'My ID' -> 'Opponent's ID')
    bin_max = []  # the total count of chips per color in the game (Player 0 + Player 1)
    # acts as the mixed-radix base for encoding chip sets into unique integer IDs

    def print_board(self):
        """
        Prints the current board configuration and player status to the console.
        """
        for line in self.board:
            print(line)
        print(
            f"0: {self.locations[0]} {self.chip_sets[0]} - {Game_ct.convert_code(self.chip_sets[0], self.bin_max)}"
        )
        print(
            f"1: {self.locations[1]} {self.chip_sets[1]} - {Game_ct.convert_code(self.chip_sets[1], self.bin_max)}"
        )

    def get_chip_difference(offer1, offer2, bin_max):
        """
        Calculates the difference between two chip sets (offer1 - offer2).

        Args:
            offer1 (int): ID of the first chip set.
            offer2 (int): ID of the second chip set.
            bin_max (list): The radix basis for decoding the IDs.

        Returns:
            list: An array where index i is the difference in count for color i.
        """
        # Decode offers
        bins1 = Game_ct.convert_code(offer1, bin_max)
        bins2 = Game_ct.convert_code(offer2, bin_max)

        # Compute differences
        for i in range(len(bins1)):
            bins1[i] -= bins2[i]
        return bins1

    def invert_code(offer, bin_max):
        """
        Inverts an encoded chip set ID relative to the total chips in the game.
        Used to determine what the 'other' player has if 'offer' is one player's set.

        Args:
            offer (int): The chip set ID.
            bin_max (list): Total chips available per color.

        Returns:
            int: The encoded ID of the complementary chip set.
        """
        out_array = Game_ct.convert_code(offer, bin_max)
        for i in range(len(bin_max)):
            out_array[i] = bin_max[i] - out_array[i]
        return Game_ct.convert_chips(out_array, bin_max)

    def invert_chips(offer, bin_max):
        """
        Manually added method to invert chips

        Args:
            offer (list): The chip set ID.
            bin_max (list): Total chips available per color.

        Returns:
            int: The encoded ID of the complementary chip set.
        """
        out_array = offer.copy()
        for i in range(len(bin_max)):
            out_array[i] = bin_max[i] - out_array[i]
        return out_array

    def convert_code(offer, bin_max):
        """
        Decodes an integer ID into a readable list of chip counts.
        Uses a mixed-radix representation where the base for color i is (bin_max[i] + 1).

        Args:
            offer (int): The unique ID of the chip set.
            bin_max (list): The bases for the encoding.

        Returns:
            list: Decoded array of chip counts per color.
        """
        out_array = []
        for i in range(len(bin_max)):
            out_array.append(offer % (bin_max[i] + 1))
            offer = int(offer / (bin_max[i] + 1))
        return out_array

    def convert_chips(chips_array, bin_max):
        """
        Encodes a list of chip counts into a unique integer ID.

        Args:
            chips_array (list): Array of chip counts per color.
            bin_max (list): The bases for the encoding.

        Returns:
            int: The unique ID.
        """
        code = chips_array[len(chips_array) - 1]
        for i in range(len(chips_array) - 2, -1, -1):
            code = code * (bin_max[i] + 1) + chips_array[i]
        return code

    def load_setting(self, board, chips1, chips2, location1, location2):
        """
        Manually loads a specific game setting.

        Args:
            board (list): 5x5 grid of colors.
            chips1 (list): Chip counts for player 0.
            chips2 (list): Chip counts for player 1.
            location1 (int): Goal location index for player 0.
            location2 (int): Goal location index for player 1.
        """
        self.board = board
        self.chips = [chips1, chips2]
        self.calculate_setting()
        self.locations = [location1, location2]

    def get_random_board_color(self):
        """
        Selects a random color from the pool of colors currently on the board.
        Ensures chips distributed to players match colors actually present on the board.
        """
        i = random.randrange(len(self.unused_board_colors))
        n = int(self.unused_board_colors[i])
        # Remove the selected color instance from the bag of possible chips.
        self.unused_board_colors = (
            self.unused_board_colors[:i] + self.unused_board_colors[i + 1 :]
        )
        return n

    def init(self):
        """
        Internal method to generate a single random game configuration.
        Sets up the board, distributes chips, and calculates utilities.
        """
        self.unused_board_colors = ""
        self.board = [[0 for i in range(5)] for j in range(5)]
        self.chips = [[0 for i in range(self.number_of_colors)] for j in range(2)]

        # Generate board colors.
        for i in range(5):
            for j in range(5):
                if i == 2 and j == 2:
                    self.board[i][j] = (
                        0  # the center is the start location; its color is irrelevant
                    )
                else:
                    # Choose a random color for this tile.
                    self.board[i][j] = random.randrange(self.number_of_colors)
                    # This color is now eligible for chip drawing later on.
                    # We essentially keep track of a bag of chips that can be drawn.
                    self.unused_board_colors += str(self.board[i][j])

        # Distribute chips based on board colors.
        for i in range(self.number_of_chips_per_player):
            self.chips[0][self.get_random_board_color()] += 1
            self.chips[1][self.get_random_board_color()] += 1

        # Precompute utilities.
        self.calculate_setting()

    def calculate_setting(self):
        """
        Precomputes the utility functions and possible chip permutations for the current game.
        This sets up the `bin_max` bases, `flip_array`, and calculates the score for
        every possible chip combination at every possible goal location.

        Returns:
            bool: True if a valid goal configuration was found, False otherwise.
        """
        # Determine the maximum chips of each color (sum of both players).
        offer_count = 1
        self.bin_max = [0] * self.number_of_colors
        for i in range(self.number_of_colors):
            self.bin_max[i] = self.chips[0][i] + self.chips[1][i]
            offer_count *= self.bin_max[i] + 1

        # Create a lookup table to invert offers (My Chips <-> Opponent Chips).
        self.flip_array = [0] * offer_count
        for i in range(offer_count):
            self.flip_array[i] = Game_ct.invert_code(i, self.bin_max)

        # Store current chip holdings as encoded integer IDs.
        self.chip_sets = [
            Game_ct.convert_chips(self.chips[0], self.bin_max),
            Game_ct.convert_chips(self.chips[1], self.bin_max),
        ]

        # Calculate utility for all potential goal locations.
        self.utility_function = []
        for i in range(5):
            for j in range(5):
                # Only consider goals that are at least distance 3 from the center (2,2).
                if abs(i - 2) + abs(j - 2) > 2:
                    self.utility_function.append(
                        self.get_utility_function(i, j, offer_count)
                    )

        # Create an ID representing possessing all chips in the game (for Rule 1).
        all_chips_code = Game_ct.convert_chips(self.bin_max, self.bin_max)

        # Assign random goals to players.
        self.locations = [0, 0]
        attempts = 0

        while self.locations[0] == self.locations[1]:
            # Guard against infinite loops if no valid distinct goals exist at all
            if attempts > 50:
                return False

            for player in range(2):
                for i in range(20):  # try 20 times to find a valid goal
                    self.locations[player] = random.randrange(
                        len(self.utility_function)
                    )

                    # Look up utilities for the newly picked location
                    utility_with_start = self.utility_function[self.locations[player]][
                        self.chip_sets[player]
                    ]
                    utility_with_all = self.utility_function[self.locations[player]][
                        all_chips_code
                    ]

                    # First pruning rule: ensure the goal is reachable with ALL chips (utility >= 800).
                    # Second pruning rule: ensure the goal is strictly unreachable with STARTING chips (utility < 800).
                    if utility_with_all >= 800 and utility_with_start < 800:
                        break  # Both conditions met, we can accept this location and stop trying
                else:
                    # The else block executes ONLY if the loop finishes without hitting 'break'
                    return False

            attempts += 1

        return True

    def get_utility_function(self, x, y, offer_count):
        """
        Calculates the maximum score achievable for a specific goal (x, y)
        given every possible subset of chips.

        Uses an iterative dynamic programming approach (value iteration) to propagate
        scores from the start position.

        Args:
            x (int): Goal x-coordinate.
            y (int): Goal y-coordinate.
            offer_count (int): Total number of unique chip set combinations.

        Returns:
            list: A list of size `offer_count` where index `k` is the score achievable
                  with chip set `k`.
        """
        # score_matrix[i][j][k] stores the best score at board pos (i,j) with chip set k.
        score_matrix = [
            [[0 for k in range(offer_count)] for j in range(5)] for i in range(5)
        ]

        # Calculate the distance from the center to the goal location (either 3 or 4 steps).
        start_dist = abs(2 - x) + abs(2 - y)

        # Initialize base scores.
        for k in range(offer_count):
            # Store the number of remaining chips that the agent has.
            n = sum(Game_ct.convert_code(k, self.bin_max))

            for i in range(5):
                for j in range(5):
                    # Distance from current tile (i,j) to goal (x,y).
                    dist_to_goal = abs(i - x) + abs(j - y)

                    # 100 points per step closer to the goal, and 50 points for every unused chip.
                    steps_made = start_dist - dist_to_goal
                    score_matrix[i][j][k] = 50 * n + 100 * steps_made

                    # Bonus for reaching the goal.
                    if i == x and j == y:
                        score_matrix[i][j][k] += 500

        # --- VALUE ITERATION LOOP ---
        # This loop propagates high scores from the goal (or high-value positions)
        # outwards to the rest of the board.
        # It continues until the scores stabilize (converge), meaning no further
        # moves can improve a player's utility.

        do_continue = True
        while do_continue:
            do_continue = False  # assume we are done unless we find an improvement

            # Iterate over every tile on the board.
            for i in range(5):
                for j in range(5):
                    # Try to push the score from cell (i,j) to its neighbors.
                    # If any neighbor's score is updated, process_location returns True.
                    if self.process_location(score_matrix, i, j):
                        do_continue = True

        # Return the scores for the starting position (2,2).
        return score_matrix[2][2]

    def process_location(self, score_matrix, x, y):
        """
        Updates the score of a location (x,y) by checking if it can reach neighbors
        with higher scores by spending a chip.

        Args:
            score_matrix (list): The 3D DP table.
            x (int): Current x-coordinate.
            y (int): Current y-coordinate.

        Returns:
            bool: True if any score was updated (convergence check).
        """
        has_changed = False
        for k in range(len(score_matrix[0][0])):
            # Convert ID 'k' to actual chip counts.
            bins = Game_ct.convert_code(k, self.bin_max)

            # Check if we have a chip matching the color of the current tile (x,y).
            # If so, it means we could have moved here from a neighbor by spending that chip.
            # We look backwards: if I have `bins` chips at (x,y), I had `bins + 1 color`
            # at the previous square.

            current_color = self.board[x][y]
            if bins[current_color] < self.bin_max[current_color]:
                # Simulate having one more chip of the current tile's color.
                bins[current_color] += 1
                old_state = Game_ct.convert_chips(bins, self.bin_max)

                # Compare current score vs coming from neighbors (up, left, down, right).
                # This logic effectively pushes values from (x,y) to neighbors.

                # Check up neighbor (x-1, y).
                if x > 0 and score_matrix[x - 1][y][old_state] < score_matrix[x][y][k]:
                    score_matrix[x - 1][y][old_state] = score_matrix[x][y][k]
                    has_changed = True

                # Check left neighbor (x, y-1).
                if y > 0 and score_matrix[x][y - 1][old_state] < score_matrix[x][y][k]:
                    score_matrix[x][y - 1][old_state] = score_matrix[x][y][k]
                    has_changed = True

                # Check down neighbor (x+1, y).
                if x < 4 and score_matrix[x + 1][y][old_state] < score_matrix[x][y][k]:
                    score_matrix[x + 1][y][old_state] = score_matrix[x][y][k]
                    has_changed = True

                # Check right neighbor (x, y+1).
                if y < 4 and score_matrix[x][y + 1][old_state] < score_matrix[x][y][k]:
                    score_matrix[x][y + 1][old_state] = score_matrix[x][y][k]
                    has_changed = True

        return has_changed
