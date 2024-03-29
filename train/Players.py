from Brain_DDQN import Brain
import numpy as np
import torch

steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# --------------------------- PLAYER CLASS ---------------------------

class Player:

    def __init__(self, hand, policy):
        self.cards = hand
        self.policy = policy
        self.points = 0
        self.state = None

        if policy == 'Q_learning':
            # It the policy is 'Q_learning' we have to use the trained network
            self.brain = Brain(162, 40)


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

        # List that will contain all the cards suitable for the greedy method
        possible_action = []


        if card_on_table == None:
            # No informations about the other player so play the card less valuable
            c = self.max_min_values(self.cards, maxx = False)
            self.cards.remove(c)
            return c
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
                    # Check if you have cards that can win the hand
                    if ((card.seed == card_on_table.seed and 
                        card.value > card_on_table.value) or 
                        card.is_Briscola):

                        possible_action.append(card)

        # There aren't card that can win the hand
        if len(possible_action) == 0:
            # Select the card with less points
            c = self.max_min_values(self.cards, maxx = False)
            self.cards.remove(c)
            return c

        else:
            # Select the card with more points out of the ones that can win the hand
            c = self.max_min_values(possible_action, maxx = True)
            self.cards.remove(c)
            return c
    

    # -------------------- RANDOM PLAYER --------------------

    def random_action(self):
        '''
        Choose the card to play randomly.
        '''
        card_index = np.random.randint(len(self.cards) + 1)
        card_to_play = self.cards.pop(card_index - 1)

        return card_to_play



    # -------------------- DQN PLAYER --------------------


    def Q_action(self, state):
        '''
        Choose the card Using the nn predicton.
        '''

        card_id = self.brain.predict_next_action(torch.tensor([state], device='cpu',  
                                                dtype=torch.float64))
        for card in self.cards:
            if card.id == card_id:
                self.cards.remove(card)
                return card


