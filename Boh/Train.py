from Objects import Deck
import numpy as np
import random
import math

steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000



class Game_Train():

    def __init__(self):

        self.deck = Deck()
        
        self.player_1 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Q_learning', 'points': 0}
        self.player_2 = {'cards': self.deck.draw_first_hand(), 
                        'policy': 'Greedy', 'points': 0}

        self.state = self.initial_state(self.player_1['cards'], 
                                        self.player_2['cards'], 
                                        self.deck.briscola)
        
        self.player_1_state = self.get_state_for_player(1)
        self.player_2_state = self.get_state_for_player(2)
        
        self.first_to_play = np.random.randint(3)
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
        
        self.first_to_play = np.random.randint(3)
        self.card_on_table = None


# -------------------- STATE FUNCTIONS --------------------

    def initial_state(self, cards_1, cards_2, briscola):
        '''
        Define the initial state of the game, right after the first 
        cards are given to the players.

        In general the states will be:
        - 0, card still in the deck.
        - 1, card in the hand of the player 1.
        - 2, card in the hand of the player 1 and is a briscola.
        - 3, card in the hand of the player 2.
        - 4, card in the hand of the player 2 and is a briscola.
        - 5, the last card of the deck (known to the players).
        - 6, card played by player 1 and player 1 played first.
        - 7, card played by player 1 and player 1 played second.
        - 8, card played by player 2 and player 2 played first.
        - 9, card played by player 2 and player 2 played second.
        - 10, card already played.
        '''
        state = np.zeros(40)
        
        for card in cards_1:
            state[card.id] = 1
            if card.is_Briscola: state[card.id] = 2

        for card in cards_2:
            state[card.id] = 3
            if card.is_Briscola: state[card.id] = 4

        state[briscola.id] = 5

        return state

    def get_state_for_player(self, player):
        '''
        Function that will give back the player's informations.
        So no info about the cards in the hand of the opposing player.

        For each player the state will be:
        - 0, if the card is still in the deck in the opponent's hand.
        - 1, if the card is in the hand of the player
        - 2, if the card is in the hand of the player and is a briscola
        - 3, the last card of the deck (known to the players).
        - 4, card played by the player and the player played first.
        - 5, card played by the player and the player played second.
        - 6, card played by the other player, and they played first.
        - 7, card played by the other player, and they played second.
        - 8, card already played.
        '''
        player_state = []

        if player == 1:
            # The state the player can be aware of
            known_states = [i for i in range(11) if i not in [3,4]]
            for state in self.state:
                # If the player can't know this state, set it to 0
                if state not in known_states:
                    player_state.append(0)
                elif state > 2: 
                    # Take 2 out since the states regarding player 2 
                    # hand are skipped 
                    player_state.append(state - 2)
                else:
                    player_state.append(state)

        else: 
            # The state the player can be aware of
            known_states = [i for i in range(11) if i not in [1,2]]
            for state in self.state:
                # If the player can't know this state, set it to 0
                if state not in known_states:
                    player_state.append(0)
                elif state in [6, 7]:
                    player_state.append(state)
                elif state in [8, 9]:
                    player_state.append(state - 4)
                else:
                    # Take 2 out since the states regarding player 1 
                    # hand are skipped 
                    player_state.append(state - 2)

        return player_state
    
    # -------------------- STATE UPDATE FUNCTIONS --------------------

    def update_state_after_play(self, card, new_state, on_table = False):
            '''
            Function that update the state after a card is played, 
            in case it's the first card to be played set it as on table.
            '''
            self.state[card.id] = new_state
            if on_table == True:
                self.card_on_table = card

            else:
                on_table = self.card_on_table
                self.card_on_table = [on_table, card]


    def update_state_after_hand(self):
            '''
            Function that update the state of the cards after a full
            hand of the game. 
            
            The card on table will be set to None 
            '''

            # Set the state of the cards played during the hand to 10 
            for card in self.card_on_table:
                self.state[card.id] = 10
            
            # Reset the card on table
            self.card_on_table = None

    def update_state_after_draw(self, card_1, card_2):
            '''
            Function that update the state of the cards after the cards
            are drawn. 
            '''

            self.state[card_1.id] = 2 if card_1.is_Briscola else 1
            self.state[card_2.id] = 4 if card_2.is_Briscola else 3


    # ------------------- Q LEARN FUNCTIONS -------------------

    def step(self, card):
        '''
        Function that finds the new 
        '''
        init_score_1 = self.player_1['points']
        init_score_2 = self.player_2['points']
        
        winner = None
        if self.first_to_play == 1:
            self.update_state_after_play(card, 6, True)
        else:
            self.update_state_after_play(card, 7)
            winner = self.find_hand_winner(card, self.card_on_table, 2)
            self.first_to_play = winner

        done = self.finish_step(winner)

        if done:
            return (self.get_state_for_player(1), 
                (self.player_1['points'] - self.player_2['points'])*2, done)
        
        hand_point_1 = init_score_1 - self.player_1['points']
        hand_point_2 = init_score_2 - self.player_2['points']

        return (self.get_state_for_player(1), hand_point_1 - hand_point_2, done)

    

    def finish_step(self, winner):
        '''
        Function that finishes the step done before and set up for 
        the new iteration.
        '''
        # The hand was not finished
        if not winner:
            # Let player 2 play it's card
            state = self.get_state_for_player(2)
            card = self.get_action_train(state, self.player_2, self.card_on_table)
            self.update_state_after_play(card, 9)

            # Fine the new winner
            winner = self.find_hand_winner(self.card_on_table[0], card, 1)
            self.first_to_play == winner

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
            card = self.get_action_train(state, self.player_2, self.card_on_table)
            self.update_state_after_play(card, 8, True)

        return False
    

    # -------------------- GET ACTION FUNCTION --------------------

    def get_action_train(self, state, player, brain = None, card_on_table = None):
        if player['policy'] == 'Random':
            return self.random_action(player)
        elif player['policy'] == 'Greedy':
            return self.greedy_action(player, card_on_table)
        else: 
            return self.Q_action(state, player, brain)
            
    

    def Q_action(self, state, player, brain):
        '''
        Choose the card:
        1. Randomly (exploration) if the random number is less than eps.
        2. Using the nn predicton (exploitation) otherwise.
        '''

        global steps_done

        # Find the new threshold that changes with the number of steps
        #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        #math.exp(-1. * steps_done / EPS_DECAY)
        eps_threshold = 0
        steps_done += 1


        # Under the threshold we just use the random policy 
        if random.random() < eps_threshold:
            return self.random_action(player)
        
        # Over the threshold we use the nn prediction
        else:
            card_id = brain.predict_next_action(state)
            for card in player['cards']:
                if card.id == card_id:
                    player['cards'].remove(card)
                    return card



    def random_action(self, player):
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
        possible_action = []


        if card_on_table == None:
            return self.max_min_values(player['cards'], maxx = False)
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
                    if ((card.seed == card_on_table.seed and 
                        card.value > card_on_table.value) or 
                        card.is_Briscola):

                        possible_action.append(card)

        # I cannot take
        if len(possible_action) == 0:
            # I want to put the card with less points
            return self.max_min_values(player['cards'], maxx = False)

        else:
            # I want to put the card with more points
            return self.max_min_values(possible_action, maxx = True)
    

    # -------------------- GAME FUNCTIONS USED TO TRAIN --------------------


    def find_hand_winner(self, card_1, card_2, first):
        '''
        Function that given the cards player and the order in which 
        they were played find the winner and gives the points accordingly
        '''
        points = card_2.value + card_1.value

        # Both players played a card with the same seed
        if card_1.seed == card_2.seed:
            if card_1.value >= card_2.value and card_1.numb > card_2.numb:
                self.player_1['points'] += points
                return 1
            else:
                self.player_2['points'] += points
                return 2
        
        # One player played a briscola and the other didn't
        if card_1.is_Briscola:
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