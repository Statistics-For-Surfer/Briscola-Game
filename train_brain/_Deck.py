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


        self.deck_dic = self.get_deck_dic()
        
        self.seed_briscola = self.deck_dic[self.last_card][0]

        

    def get_deck_dic():
        '''
        da una carta gli da il seme e il valore
        '''

        deck = []
        seeds = ['Bastoni', 'Denara' , 'Coppe' , 'Spade']
        cards = [1,2,3,4,5,6,7,8,9,10]
        values = [11,0,10,0,0,0,0,2,3,4]

        # Create the list of tuples that represent each card of the deck
        for seed in seeds:
            for j , card in enumerate(cards):
                deck.append((seed, card, values[j]))

        map_value = {i: deck[i] for i in range(len(deck))}

        return map_value



    def get_briscola(self):
        '''
        rappresentiamo la briscola come la lista delle carte
        che sono briscola
        '''
        if self.last_card < 10:
            return list(np.arange(0,10))
        if self.last_card < 20:
            return list(np.arange(10,20))
        if self.last_card < 30:
            return list(np.arange(20,30))
        return list(np.arange(30,40))


    def draw_card(self):
        '''
        Peschiamo una carta
        '''

        return self.deckofcards.pop(0)


    def deal_hand(self):
        '''
        dare la prima mano
        '''

        hand_size = 3
        hand = []

        for i in range(hand_size):
            hand.append(self.deckofcards.pop(0))

        return hand
