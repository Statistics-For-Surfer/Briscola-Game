import numpy as np
from Brain_DDQN import Brain
import random
import math

steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# --------------------------- PLAYER CLASS ---------------------------

class Player:

    def __init__(self, hand, policy, train = False):
        self.cards = hand
        self.policy = policy
        self.points = 0
        self.state = None

        if policy == 'Q_learning':
            self.brain = Brain(40, 40, train)


    def get_action(self, state, card_on_table = None):
        '''
        Function that will call the function corresponding to the 
        policy that is been used. 
        '''
        if self.policy == 'Greedy':
            return self.greedy_action(card_on_table)
        elif self.policy == 'Random':
            return self.random_action()
        else:
            return self.Q_action(state)

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
        

    
    def greedy_action(self, card_on_table = None):
        '''
        Choose the card to play with a greedy method, the agent will 
        take the hand if it can else just throw the one with the lowest 
        value.
        '''
        possible_action = []


        if card_on_table == None:
            return self.max_min_values(self.cards, maxx = False)
        else:
            # The card on the table is a briscola
            if card_on_table.is_Briscola:
                # check if you have a briscola greater than the 
                # one on the table
                for card in self.cards:
                    if (card.is_Briscola and 
                        card.value > card_on_table.value):

                        possible_action.append(card)
            else:
                for card in self.cards:
                    if ((card.seed == card_on_table.seed and 
                        card.value > card_on_table.value) or 
                        card.is_Briscola):

                        possible_action.append(card)

        # I cannot take
        if len(possible_action) == 0:
            # I want to put the card with less points
            return self.max_min_values(self.cards, maxx = False)

        else:
            # I want to put the card with more points
            return self.max_min_values(possible_action, maxx = True)
    

    # -------------------- RANDOM PLAYER --------------------

    def random_action(self):
        '''
        Choose the card to play randomly.
        '''

        card_index = np.random.randint(len(self.cards))
        card_to_play = self.cards.pop(card_index)

        return card_to_play



    # -------------------- DQN PLAYER --------------------


    def Q_action(self, state):
        '''
        Choose the card:
        1. Randomly (exploration) if the random number is less than eps.
        2. Using the nn predicton (exploitation) otherwise.
        '''

        global steps_done

        # Find the new threshold that changes with the number of steps
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1


        # Under the threshold we just use the random policy 
        if random.random() < eps_threshold:
                return self.random_action()
        
        # Over the threshold we use the nn prediction
        else:
            card_id = self.brain.predict_next_action(state)
            for card in self.cards:
                if card.id == card_id:
                        return card


