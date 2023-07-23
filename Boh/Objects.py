
import random
import numpy as np


# --------------------------- CARD CLASS ---------------------------

class Card:
    '''
    Class used to define each card of the deck. 
    The cards will have 5 different attributes:

    - An identifier, that goes from 0 to 39.
    - The number of the card, from 1 to 10.
    - The seed of the card, 'Bastoni', 'Coppe', 'Denari' or 'Spade'.
    - The values of the card.
    - Whether the card is a bricola.
    '''

    def __init__(self, args):

        self.id = None
        self.numb = None
        self.seed = None
        self.value = None
        self.is_Briscola = False

        self.set_values(*args)

    def set_values(self, card_id, numb, seed, value, briscola = False):
        '''
        To set the attrubutes of the card
        '''

        self.id = card_id
        self.numb = numb
        self.seed = seed
        self.value = value
        self.is_Briscola = briscola

    def to_tuple(self):
        '''
        Return the information of the card as a tuple of seed,
        number of the card and its value
        '''
        return (self.seed, self.numb, self.value)


# --------------------------- DECK CLASS ---------------------------


class Deck:
    '''
    Class used to define the deck, that is going to be 
    composed of 40 cards.
    '''

    def __init__(self):# rand = True):
        self.deck = self.create_deck()
        random.seed(random.randint(1, 10))
        random.shuffle(self.deck)
        random.seed()
        self.briscola = self.get_briscola()
        self.all_cards = self.set_briscola()

    def create_deck(self):
        '''
        Function that creates a list of card objects
        '''

        # Possible attributes of the cards
        seeds = ['Bastoni', 'Coppe', 'Denari', 'Spade']
        numbers = [i for i in range(1,11)]
        values = [11, 0, 10, 0, 0, 0, 0, 2, 3, 4]
        card_id = 0 

        # Create the list of the possible combinations
        deck = []
        for seed in seeds:
            for idx in range(10):
                attrs = [card_id, numbers[idx],seed,values[idx]]
                deck.append(Card(attrs))
                card_id += 1

        # Return the deck
        return deck
    
    def get_briscola(self):
        '''
        Find the card that should be the briscola for the game and 
        put it at the end of the deck
        '''

        briscola = self.deck.pop(6)
        self.deck.append(briscola)

        return briscola


    def set_briscola(self):
        '''
        Function that marks the cards of the deck as briscola.
        The function will also give back all the cards in the deck 
        with their completed attributes
        '''
        all_cards = []
        for card in self.deck:
            if card.seed == self.briscola.seed:
                card.is_Briscola = True
            all_cards.append(card)
        return all_cards


    def draw_card(self):
        '''
        Function that return the first element of the deck
        '''

        try:
            return self.deck.pop(0)
        except:
            print('The deck was empty, no card was returned')


    def draw_first_hand(self):
        '''
        Function that will give the first hand to the players
        '''
        try:
            hand = []
            for _ in range(3):
                hand.append(self.deck.pop(0))
            return hand
        except:
            print('The deck was empty, no card was returned')

    def get_card(self, card_id):
        '''
        Function that given the id of a card will give back the 
        correspondent card object
        '''
        for card in self.all_cards:
            if card_id == card.id: return card
