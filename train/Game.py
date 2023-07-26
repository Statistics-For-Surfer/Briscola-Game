from Objects import Deck
from Players import Player
import numpy as np

# --------------------------- GAME CLASS ---------------------------


class Game:
    '''
    Simulation environment used for testing after the training
    '''

    def __init__(self, policy):
        # Define the object: Deck and players
        self.deck = Deck()
        self.player_1 = Player(self.deck.draw_first_hand(), policy[0])
        self.player_2 = Player(self.deck.draw_first_hand(), policy[1])

        # Initial state
        self.state = self.initial_state(self.player_1.cards, 
                                        self.player_2.cards, 
                                        self.deck.briscola)
        
        # State from the players POV
        self.player_1.state = self.get_state_for_player(1)
        self.player_2.state = self.get_state_for_player(2)
        
        # Select who starts the game
        self.first_to_play = np.random.randint(2) + 1
        self.card_on_table = None

    def reset(self):
        '''
        Reset the game to initial state
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
        '''
        Function the initialize the state of the game, that is represented with
        a list of 162 elements:

        - Form 1 to 40, to identify the cards that were already played.
        - From 41 to 80, to identify the cards that are in the hand of the players (1 for player 1 and 2 for player 2).
        - From 81 to 120, to identify the card that was selected as Briscola.
        - From 121 to 160, to identify the cards on table, i.e. the cards played during the hand.
        - 161, the points of the player 1.
        - 162, the points of the player 2.
        '''

        # Initialize the state vector
        state = np.zeros(162)

        # 1 for player 1 cards
        for card in cards_1:
            state[int(card.id) + 40] = 1
        
        # 2 for player 2 cards
        for card in cards_2:
            state[card.id + 40] = 2

        # 1 for the Briscola
        state[briscola.id + 80] = 1
        
        return state
    
    def get_state_for_player(self, player):
        '''
        Given 1 or 2 to identify the player gives back a state vector from the 
        POV of that player, so a list of 162 elements:

        - Form 1 to 40, to identify the cards that were already played.
        - From 41 to 80, to identify the cards that are in the hand of the player
        - From 81 to 120, to identify the card that was selected as Briscola.
        - From 121 to 160, to identify the cards on table, i.e. the cards played during the hand.
        - 161, the points of the player.
        - 162, the points of the opponent.
        '''

        if player == 1:
            player_state = np.copy(self.state)
            # Identify the cards that are in the opponents hand to mask them
            idx = np.where(self.state[40:80] == 2)[0]
            player_state[40 + idx] = 0
            player_state[160] = self.player_1.points
            player_state[161] = self.player_2.points
            return player_state
        else:
            player_state = np.copy(self.state)
            # Identify the cards that are in the opponents hand to mask them
            idx_1 = np.where(self.state[40:80] == 1)[0]
            # And the cards in the player hand to change the value to 1 for 2
            idx_2 = np.where(self.state[40:80] == 2)[0]
            player_state[40 + idx_1] = 0
            player_state[40 + idx_2] = 1

            # Put the points in the right order
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

            # There is nothing on the table
            if on_table == True:
                self.card_on_table = card
            
            # There is already one card on the table
            else:
                on_table = self.card_on_table
                self.card_on_table = [on_table, card]

    def update_state_after_hand(self):
        '''
        Function that update the state of the cards after a full
        hand of the game. 
        
        The card on table will be set to None 
        '''
        # Identify the cards that are on the table
        idx = np.where(self.state[120:160] == 1)[0]

        # Set them as played
        self.state[idx] = 1
        # Remove them from players hand/table
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
            # Play a hand
            self.hand()
            # Update the states
            self.update_state_after_hand()

            # If the players don't have any cards finish the game
            if not len(self.player_1.cards):
                break

            # If there are still cards in the deck draw
            if len(self.deck.deck):
                # Cards to be drawn 
                card_1 = self.deck.draw_card()
                card_2 = self.deck.draw_card()

                # Assign the cards to the right player given the outcome of the previous hand
                if self.first_to_play == 1:
                    self.player_1.cards.append(card_1)
                    self.player_2.cards.append(card_2)
                    self.update_state_after_draw(card_1, card_2)
                else:
                    self.player_1.cards.append(card_2)
                    self.player_2.cards.append(card_1)
                    self.update_state_after_draw(card_2, card_1)
                    
        # Find the final score           
        final_score = self.player_1.points - self.player_2.points

        # Reset for the new game
        self.reset()

        return final_score



    def hand(self, Q_learning = None):
        '''
        Simulate one hand of the game. 
        '''
        # Player 1 plays first
        if self.first_to_play == 1:
            # Get the player POV
            state_1 = self.get_state_for_player(1)
            # Select the card to be played
            card_1 = self.player_1.get_action(state_1, self.card_on_table)
            # Update the game state
            self.update_state_after_play(card_1, 1, True)
            # Get the other player POV
            state_2 = self.get_state_for_player(2)
            # Select the card to be played
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            # Update the game state
            self.update_state_after_play(card_2, 2)

        # Player 2 plays first, same as before but in different order
        else:
            state_2 = self.get_state_for_player(2)
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            self.update_state_after_play(card_2, 2, True)
            state_1 = self.get_state_for_player(1)
            card_1 = self.player_1.get_action(state_1, self.card_on_table)
            self.update_state_after_play(card_1, 1)

        # Find how won the hand and will have to play first in the following
        self.first_to_play = self.find_hand_winner(card_1, card_2,
                                            self.first_to_play)



    def find_hand_winner(self, card_1, card_2, first):
        '''
        Function that given the cards played, respecly from player 1 and 2,
        find the winner and gives the points accordingly
        '''

        points = card_2.value + card_1.value

        # Both players played a card with the same seed
        if card_1.seed == card_2.seed:
            if card_1.value == card_2.value:
                # Same seed + same value means no points to be assigned
                return 1 if card_1.numb > card_2.numb else 2
            
            elif card_1.value > card_2.value:
                self.player_1.points += points
                return 1
            else:
                self.player_2.points += points
                return 2
        
        # If one player played a briscola and the other didn't
        if card_1.is_Briscola:
            self.player_1.points += points
            return 1
        elif card_2.is_Briscola:
            self.player_2.points += points
            return 2
        
        # If the seeds are different and there are no briscole then who played first wins.
        else:
            if first == 1:
                self.player_1.points += points
                return 1
            else:
                self.player_2.points += points
                return 2
