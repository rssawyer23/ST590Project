import environment as e
import numpy as np


class BlackjackEnvironment(e.Environment):
    def __init__(self):
        e.Environment.__init__(self)
        # State representation is a vector with 3 variables
        # Hand Value - sum of card values in hand
        # Usable Ace - Boolean if a current card value of 11 can be replaced by a 1
        # Dealer Value - value of the card the dealer is showing

    def generate_all_states(self):
        player_card_values = range(2,32)
        usable_ace = [True, False]
        dealer_value = range(2,12)
        terminal = [False, True]
        possible_state_tuples = []
        for pcv in player_card_values:
            for ua in usable_ace:
                for dv in dealer_value:
                    for term in terminal:
                        possible_state_tuples.append((pcv, ua, dv, term))
        return possible_state_tuples

    def possible_actions(self, state):
        return [0, 1]

    def initial_state(self):
        state = [0, False, 0, False]
        card_one = self.draw_card()
        card_two = self.draw_card()
        if card_one == 1 and card_two == 1:  # Pair of aces drawn, using one ace as 1, one ace as a usable 11
            state[0] = 12
            state[1] = True
        elif card_one == 1 or card_two == 1:  # One ace drawn, is a usable 11 (already at 1 so adding 10 to make 11)
            state[0] = card_one + card_two + 10
            state[1] = True
        else:  # No aces drawn
            state[0] = card_one + card_two
            state[1] = False
        dealer_card = self.draw_card()
        if dealer_card == 1:
            state[2] = 11
        else:
            state[2] = dealer_card
        return state

    def draw_card(self):
        card = np.random.randint(1, 14, 1)
        if card > 10:
            card = 10
        return int(card)

    def stay_vs_dealer(self, state):  # Agent performed "stay" or reached 21, dealer will play to 17+
        dealer_value = state[2]
        usable_ace = dealer_value == 11
        while dealer_value < 17 and not usable_ace and state[0] > dealer_value:
            dealer_card = self.draw_card()
            if dealer_card == 1 and not usable_ace:
                dealer_card = 11
            dealer_value += dealer_card
            if dealer_value > 21 and usable_ace:
                dealer_value -= 10
                usable_ace = False
        if dealer_value > 21:  # Dealer Bust
            return 1
        elif dealer_value >= state[0]:  # Ties also go to dealer
            return -1
        else:  # Agent value higher than dealer value
            return 1

    def determine_next_state(self, state, action):
        next_state = state.copy()
        if action == 0:
            next_state[3] = True
            return next_state
        else:
            card_value = self.draw_card()
            if card_value == 1 and not next_state[1]:  # Do not already have usable ace
                card_value = 11
                next_state[1] = True
            elif card_value == 1:  # Already have a usable ace, so to prevent bust this will not be a usable ace
                card_value = 1
            next_state[0] += card_value
            if next_state[0] > 21 and next_state[1]:  # Bust, so converting usable ace from eleven to one (subtract 10)
                next_state[1] = False
                next_state[0] -= 10
            if next_state[0] > 21:
                next_state[3] = True
        return next_state

    def get_reward(self, state, action, next_state): # will return reward and boolean determining if continue allowed
        if next_state[0] > 21:
            return -1
        elif next_state[3]:  # Blackjack Agent stays or is forced to stay by reaching 21
            return self.stay_vs_dealer(next_state)  # Evaluate dealer hand, give reward, end round
        else:
            return 0  # Agent has not busted and did not stay on last hand

