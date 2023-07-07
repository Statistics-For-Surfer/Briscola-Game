import sys
import numpy as np

class Deck:
    """A class of a deck of playingcards"""
    def __init__(self):

        deckofcards = np.arange(40)
        np.random.shuffle(deckofcards)

        deck_list = deckofcards.tolist()

        self.deckofcards = deck_list
    
        self.last_card = self.deckofcards[-1]
        
        self.briscola =  self.get_briscola()



    def get_briscola(self):
        if self.last_card < 10:
            return list(np.arange(0,10))
        if self.last_card < 20:
            return list(np.arange(10,20))
        if self.last_card < 30:
            return list(np.arange(20,30))
        return list(np.arange(30,40))


    def draw_card(self):

        return self.deckofcards.pop(0)


    def deal_hand(self):

        hand_size = 3
        hand = []

        for i in range(hand_size):
            hand.append(self.deckofcards.pop(0))

        return hand
