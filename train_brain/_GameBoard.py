from _PlayerBoard import PlayerBoard
from _DiscardBoard import DiscardBoard
from _Deck import Deck
from _GamePlay import make_move

class GameBoard():

    def __init__(self):

        # Set variables for the game
        self.cards_remaining = 34
        # Create a deck and shuffle it 3 times
        self.deck = Deck()
        self.briscola = self.deck.briscola

        self.p1_board = PlayerBoard(self.deck.deal_hand(),self.briscola)
        self.p2_board = PlayerBoard(self.deck.deal_hand(),self.briscola)

        self.card_on_table =  None
        

    def report_score(self, player):
        if (player == 1):
            player_board = self.p1_board
        else:
            player_board = self.p2_board

        return player_board.score
