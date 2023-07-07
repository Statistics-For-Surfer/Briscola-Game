from _scoring_functions import *

class PlayerBoard():

    def __init__(self, initial_hand, briscola):

        self.hand = initial_hand[:]
        self.score = 0
        self.briscola = briscola
        self.card_on_table = None
        self.first = 1

    def add_card(self, card):
        '''aggiunta della carta'''
        self.hand.append(card)


    def play_card(self, card):
        '''Pass in an integer between 0 and 3 for the suit and an integer between
        0 and 9 for the card.'''
        self.hand.remove(card)
    