from Objects import Deck
import numpy as np
import random
import math
import config


steps_done = 0
EPS_START = .82
EPS_END = 0.01
EPS_DECAY = 50000


class GameTrain():
    '''
    Simulation environment used for training the agent
    '''

    def __init__(self):

        # Define the object: Deck and players
        self.deck = Deck()
        self.player_1 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Q_learning', 'points': 0}
        self.player_2 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Random', 'points': 0}

        # Initial state
        self.state = self.initial_state(self.player_1['cards'], 
                                        self.player_2['cards'], 
                                        self.deck.briscola)
        
        # State from the players POV
        self.player_1_state = self.get_state_for_player(1)
        self.player_2_state = self.get_state_for_player(2)
        
        # Select who starts the game
        self.first_to_play = np.random.randint(2)
        self.card_on_table = None


    def reset(self):
        '''
        Reset the game after a game
        '''
        self.deck = Deck()
        self.player_1 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Q_learning', 'points': 0}
        self.player_2 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Random', 'points': 0}

        self.state = self.initial_state(self.player_1['cards'], 
                                        self.player_2['cards'], 
                                        self.deck.briscola)
        
        self.player_1_state = self.get_state_for_player(1)
        self.player_2_state = self.get_state_for_player(2)
        
        self.first_to_play = np.random.randint(2)
        self.card_on_table = None


# -------------------- STATE FUNCTIONS --------------------

    def initial_state(self, cards_1, cards_2, briscola):
        '''
        Function the initialize the state of the game, that is represented with
        a list of 162 elements:

        - Form 1 to 40, to identify the cards that were already played.
        - From 41 to 80, to identify the cards that are in the hand of the players (1 for player 1 and 2 for player 2).
        - From 81 to 120, to identify the card that was selected as Briscola.
        - From 121 to 160, to identify the cards on table, i.e. the cards played during the hand.
        - 161, the points of the player 1.
        - 162, the points of the player 2.
        '''

        # Initialize the state vector
        state = np.zeros(162)

        # 1 for player 1 cards
        for card in cards_1:
            state[int(card.id) + 40] = 1

        # 2 for player 2 cards
        for card in cards_2:
            state[card.id + 40] = 2

        # 1 for the Briscola
        state[briscola.id + 80] = 1
        
        return state
    
    def get_state_for_player(self, player):
        '''
        Given 1 or 2 to identify the player gives back a state vector from the 
        POV of that player, so a list of 162 elements:

        - Form 1 to 40, to identify the cards that were already played.
        - From 41 to 80, to identify the cards that are in the hand of the player
        - From 81 to 120, to identify the card that was selected as Briscola.
        - From 121 to 160, to identify the cards on table, i.e. the cards played during the hand.
        - 161, the points of the player.
        - 162, the points of the opponent.
        '''

        if player == 1:
            player_state = np.copy(self.state)

            # Identify the cards that are in the opponents hand to mask them
            idx = np.where(self.state[40:80] == 2)[0]
            player_state[40 + idx] = 0
            player_state[160] = self.player_1['points']
            player_state[161] = self.player_2['points']
            return player_state
        else:
            player_state = np.copy(self.state)

            # Identify the cards that are in the opponents hand to mask them
            idx_1 = np.where(self.state[40:80] == 1)[0]
            # And the cards in the player hand to change the value to 1 for 2
            idx_2 = np.where(self.state[40:80] == 2)[0]
            player_state[40 + idx_1] = 0
            player_state[40 + idx_2] = 1

            # Put the points in the right order
            player_state[160] = self.player_2['points']
            player_state[161] = self.player_1['points']
            return player_state


    # -------------------- STATE UPDATE FUNCTIONS --------------------

    def update_state_after_play(self, card, player, on_table = False):
        '''
        Function that update the state after a card is played, 
        in case it's the first card to be played set it as on table.
        '''
        self.state[card.id + 40] = player
        self.state[card.id + 120] = 1

        # There is nothing on the table
        if on_table == True:
            self.card_on_table = card

        # There is already one card on the table
        else:
            on_table = self.card_on_table
            self.card_on_table = [on_table, card]


    def update_state_after_hand(self):
        '''
        Function that update the state of the cards after a full
        hand of the game. 
        
        - The card on table will be set to None
        - And the played cards moved from the hand of the players to the pile of the cards already played 
        '''

        # Identify the cards that are on the table
        idx = np.where(self.state[120:160] == 1)[0]

        self.state[idx] = 1
        self.state[40 + idx] = 0
        self.state[120 + idx] = 0

        # Reset the card on table
        self.card_on_table = None

    def update_state_after_draw(self, card_1, card_2):
        '''
        Function that update the state of the cards after the cards
        are drawn. 
        '''
        self.state[card_1.id + 40] = 1 
        self.state[card_2.id + 40] = 2 


    # ------------------- Q LEARN FUNCTIONS -------------------

    def first_hand(self):
        '''
        Function to use in case the first to play is not the Q-Agent
        '''

        # Get the state for the other player
        state = self.get_state_for_player(2)
        # Select the card
        card = self.get_action_train(state, self.player_2)
        # And update the states
        self.update_state_after_play(card, 2, True)


    def step(self, card):
        '''
        Function that, given the card played by the Q-Agent, finds:

        - The state the action leads to
        - The reward obtained with the action
        - Whether the game is done
        '''

        # Score of the Q-Agent has before playing the cards
        init_score_1 = self.player_1['points']

        winner = None
        if self.first_to_play == 1:
            # Update the state given the action
            self.update_state_after_play(card, 1, True)
        else:
            # Update the state given the action
            self.update_state_after_play(card, 1)
            # And find who won the hand
            winner = self.find_hand_winner(card, self.card_on_table[0], 2)
            self.first_to_play = winner

        # Go on with the game untill it's the agent turn again
        done = self.finish_step(winner)

        # If the game ended
        if done:
            # Reward 1 for a win
            if (self.player_1['points'] - self.player_2['points'])>0:
                reward = 1
            # Small reward for ties
            elif (self.player_1['points'] - self.player_2['points']) == 0:
                reward = 0.2
            # No reward for loss
            else:
                reward = 0
            return (self.get_state_for_player(1), reward, done)
        
        # else the reward will be equal to the points gained during the hand
        # divided by the maximum number of points that can be gained in a hand (22)
        hand_point_1 = self.player_1['points'] - init_score_1

        return (self.get_state_for_player(1), hand_point_1/22, done)
    

    def finish_step(self, winner):
        '''
        Function that finishes the step done before and set up for 
        the new iteration.
        '''
        # The hand was not finished
        if not winner:
            # Let player 2 play it's card
            state = self.get_state_for_player(2)
            card = self.get_action_train(state, self.player_2)
            self.update_state_after_play(card, 2)
            # Fine the new winner
            winner = self.find_hand_winner(self.card_on_table[0], card, 1)
            self.first_to_play = winner

        # Now we have the end of the hand so we update the state
        self.update_state_after_hand()

        # If the game is over return
        if not len(self.player_1['cards']):
            return True

        # If there are still cards in the deck set up for the new iteration
        if len(self.deck.deck):
            card_1 = self.deck.draw_card()
            card_2 = self.deck.draw_card()

            if(self.first_to_play == 1):
                self.player_1['cards'].append(card_1)
                self.player_2['cards'].append(card_2)
                self.update_state_after_draw(card_1, card_2)
            else:
                self.player_1['cards'].append(card_2)
                self.player_2['cards'].append(card_1)
                self.update_state_after_draw(card_2, card_1)

        # If the first to play in the new hand is player 2
        if self.first_to_play == 2:
            state = self.get_state_for_player(2)
            card = self.get_action_train(state, self.player_2)
            self.update_state_after_play(card, 2, True)

        return False
    


    def find_hand_winner(self, card_1, card_2, first):
        '''
        Function that given the cards played, respecly from player 1 and 2,
        find the winner and gives the points accordingly
        '''

        points = card_2.value + card_1.value
        
        # Both players played a card with the same seed
        if card_1.seed == card_2.seed:
            if card_1.value == card_2.value:
                return 1 if card_1.numb > card_2.numb else 2
            elif card_1.value > card_2.value:
                self.player_1['points'] += points
                return 1
            else:
                self.player_2['points'] += points
                return 2
        
        # One player played a briscola and the other didn't
        elif card_1.is_Briscola:
            self.player_1['points'] += points
            return 1
        elif card_2.is_Briscola:
            self.player_2['points'] += points
            return 2
        
        # If the seeds are different and there are no briscole then 
        # who played first wins.
        else:
            if first == 1:
                self.player_1['points'] += points
                return 1
            else:
                self.player_2['points'] += points
                return 2


    # -------------------- GET ACTION FUNCTION --------------------

    def get_action_train(self, state, player, brain = None):
        '''
        Function that finds which policy to be followed by the agents
        '''

        lambda_ = 0.7 
        # If the player isn't the Q-Agent
        if player['policy'] == 'Random':
            # It will play randomly with probability 30%
            if random.random() > lambda_:
                return self.random_action(player)
            # And using the a greedy action otherwise
            else:
                return self.greedy_action(player, self.card_on_table)
        else: 
            return self.Q_action(state, player, brain, self.card_on_table)
            
    

    def Q_action(self, state, player, brain, card_on_table):
        '''
        Choose the card:
        1. Randomly (exploration) if the random number is less than eps.
        2. Using the nn predicton (exploitation) otherwise.
        '''

        global steps_done

        # Find the new threshold that changes with the number of steps
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        config.eps = eps_threshold
        steps_done += 1


        # Under the threshold we just use the random policy (exploration)
        if random.random() < eps_threshold:
            card = self.random_action(player)
            if card_on_table == None:
                self.card_on_table = card
            return card
        
        # Over the threshold we use the nn prediction (explitation)
        else:
            card_id = brain.predict_next_action(state)
            for card in player['cards']:
                if card.id == card_id:
                    player['cards'].remove(card)
                    if not card_on_table:
                        self.card_on_table = card
                    return card


# -------------------- RANDOM PLAYER --------------------

    def random_action(self, player):
        '''
        Select a random action out of the possible one.
        '''

        card_index = np.random.randint(len(player['cards']))
        card_to_play = player['cards'].pop(card_index)

        return card_to_play
    

# -------------------- GREEDY PLAYER --------------------

    def max_min_values(self, cards, maxx = True):
        '''
        Function that will return the card with either the maximum 
        of the minimum value out of the player current playable cards.
        '''
        
        # Want to retrieve the max
        if maxx == True:
            max_card = None
            max_value = float('-inf')

            for card in cards:
                if card.value > max_value:
                    max_value = card.value
                    max_card = card

            return max_card
        
        # Want to retrieve the min
        else:
            min_card = None
            min_value = float('inf')

            for card in cards:
                if card.value < min_value:
                    min_value = card.value
                    min_card = card
                elif card.value == min_value and not card.is_Briscola:
                    min_value = card.value
                    min_card = card
            return min_card
        

    
    def greedy_action(self, player, card_on_table = None):
        '''
        Choose the card to play with a greedy method, the agent will 
        take the hand if it can else just throw the one with the lowest 
        value.
        '''

        # List that will contain all the cards suitable for the greedy method
        possible_action = []

        if card_on_table == None:
            # No informations about the other player so play the card less valuable
            c = self.max_min_values(player['cards'], maxx = False)
            player['cards'].remove(c)
            return c
        else:
            # The card on the table is a briscola
            if card_on_table.is_Briscola:
                # check if you have a briscola greater than the 
                # one on the table
                for card in player['cards']:
                    if (card.is_Briscola and 
                        card.value > card_on_table.value):

                        possible_action.append(card)
            else:
                for card in player['cards']:
                    # Check if you have cards that can win the hand
                    if ((card.seed == card_on_table.seed and 
                        card.value > card_on_table.value) or 
                        card.is_Briscola):

                        possible_action.append(card)
        
        # There aren't card that can win the hand
        if len(possible_action) == 0:
            # Select the card with less points
            c = self.max_min_values(player['cards'], maxx = False)
            player['cards'].remove(c)
            return c

        else:
            # Select the card with more points out of the ones that can win the hand
            c = self.max_min_values(possible_action, maxx = True)
            player['cards'].remove(c)
            return c
