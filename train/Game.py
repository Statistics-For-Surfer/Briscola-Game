from Objects import Deck
from Players import Player
import numpy as np

# --------------------------- GAME CLASS ---------------------------

class Game:

    def __init__(self, policy):

        self.deck = Deck()
        self.player_1 = Player(self.deck.draw_first_hand(), policy[0])
        self.player_2 = Player(self.deck.draw_first_hand(), policy[1])

        self.state = self.initial_state(self.player_1.cards, 
                                        self.player_2.cards, 
                                        self.deck.briscola)
        
        self.player_1.state = self.get_state_for_player(1)
        self.player_2.state = self.get_state_for_player(2)
        
        self.first_to_play = np.random.randint(2) + 1
        self.card_on_table = None

    def reset(self):
        '''
        Reset the game after a game
        '''
        self.deck = Deck()
        self.player_1 = Player(self.deck.draw_first_hand(), 
                            self.player_1.policy)
        self.player_2 = Player(self.deck.draw_first_hand(),
                            self.player_2.policy)

        self.state = self.initial_state(self.player_1.cards, 
                                        self.player_2.cards, 
                                        self.deck.briscola)
        
        self.player_1.state = self.get_state_for_player(1)
        self.player_2.state = self.get_state_for_player(2)
        
        self.first_to_play = np.random.randint(3)
        self.card_on_table = None


# -------------------- STATE FUNCTIONS --------------------

    def initial_state(self, cards_1, cards_2, briscola):

        state = np.zeros(162)

        for card in cards_1:
            state[int(card.id) + 40] = 1

        
        for card in cards_2:
            state[card.id + 40] = 2

        state[briscola.id + 80] = 1
        
        return state
    
    def get_state_for_player(self, player):

        if player == 1:
            idx = np.where(self.state[40:80] == 2)[0]
            player_state = np.copy(self.state)
            player_state[40 + idx] = 0
            player_state[160] = self.player_1.points
            player_state[161] = self.player_2.points
            return player_state
        else:
            idx_1 = np.where(self.state[40:80] == 1)[0]
            idx_2 = np.where(self.state[40:80] == 2)[0]
            player_state = np.copy(self.state)
            player_state[40 + idx_1] = 0
            player_state[40 + idx_2] = 1
            player_state[160] = self.player_2.points
            player_state[161] = self.player_1.points
            return player_state


    # -------------------- STATE UPDATE FUNCTIONS --------------------

    def update_state_after_play(self, card, player, on_table = False):
            '''
            Function that update the state after a card is played, 
            in case it's the first card to be played set it as on table.
            '''
            self.state[card.id + 40] = player
            self.state[card.id + 120] = 1

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


    # ------------------- GAME SIMULATION FUNCTIONS -------------------

    def game_simulation(self):
        '''
        Function that simulate a game of briscola. 
        '''

        while True:
            self.hand()
            self.update_state_after_hand()

            if not len(self.player_1.cards):
                break

            if len(self.deck.deck):
                card_1 = self.deck.draw_card()
                card_2 = self.deck.draw_card()

                if self.first_to_play == 1:
                    self.player_1.cards.append(card_1)
                    self.player_2.cards.append(card_2)
                    self.update_state_after_draw(card_1, card_2)
                else:
                    self.player_1.cards.append(card_2)
                    self.player_2.cards.append(card_1)
                    self.update_state_after_draw(card_2, card_1)
                    

        final_score = self.player_1.points - self.player_2.points

    
        # Reset for the new game
        self.reset()

        return final_score



    def hand(self, Q_learning = None):
        '''
        Simulate one hand of the game.
        '''
        if self.first_to_play == 1:
            state_1 = self.get_state_for_player(1)
            card_1 = self.player_1.get_action(state_1, self.card_on_table)
            self.update_state_after_play(card_1, 1, True)
            state_2 = self.get_state_for_player(2)
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            self.update_state_after_play(card_2, 2)

        else:
            state_2 = self.get_state_for_player(2)
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            self.update_state_after_play(card_2, 2, True)
            state_1 = self.get_state_for_player(1)
            card_1 = self.player_1.get_action(state_1, self.card_on_table)
            self.update_state_after_play(card_1, 1)

        self.first_to_play = self.find_hand_winner(card_1, card_2,
                                            self.first_to_play)



    def find_hand_winner(self, card_1, card_2, first):
        '''
        Function that given the cards player and the order in which 
        they were played find the winner and gives the points accordingly
        '''
        points = card_2.value + card_1.value

        # Both players played a card with the same seed
        if card_1.seed == card_2.seed:
            if card_1.value == card_2.value:
                return 1 if card_1.numb > card_2.numb else 2
            
            elif card_1.value > card_2.value:
                self.player_1.points += points
                return 1
            else:
                self.player_2.points += points
                return 2
        
        # One player played a briscola and the other didn't
        if card_1.is_Briscola:
            self.player_1.points += points
            return 1
        elif card_2.is_Briscola:
            self.player_2.points += points
            return 2
        
        # If the seeds are different and there are no briscole then 
        # who played first wins.
        else:
            if first == 1:
                self.player_1.points += points
                return 1
            else:
                self.player_2.points += points
                return 2
