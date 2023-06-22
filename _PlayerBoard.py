from _scoring_functions import *

class PlayerBoard():

    def __init__(self, initial_hand, briscola):

        self.hand = initial_hand[:]
        self.score = 0
        self.briscola = briscola
        self.card_on_table = None
        

    def add_card(self, card):

        self.hand.append(card)


    def play_card(self, card):
        '''Pass in an integer between 0 and 3 for the suit and an integer between
        0 and 9 for the card.'''
        self.hand.remove(card)


    def card_exists(self, card):

        return (card in self.hand)


    def find_lowest_card(self, suit):
        '''Return the smallest possible valid card to be added.'''

        played_suit = self.query_played_cards(suit)

        # The played cards are in order, and the suit is already initialized and the list 
        # is not empty.
        lowest_card = played_suit[-1]

        return lowest_card


    