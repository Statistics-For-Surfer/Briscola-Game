from Objects import Deck
from Players import Player
import numpy as np

# --------------------------- GAME CLASS ---------------------------

class Game:

    def __init__(self):

        self.deck = Deck()
        self.player_1 = Player(self.deck.draw_first_hand(), 'Random')
        self.player_2 = Player(self.deck.draw_first_hand(), 'Greedy')

        self.state = self.initial_state(self.player_1.cards, 
                                        self.player_2.cards, 
                                        self.deck.briscola)
        
        self.player_1.state = self.get_state_for_player(1)
        self.player_2.state = self.get_state_for_player(2)
        
        self.first_to_play = np.random.randint(3)
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
        '''
        Define the initial state of the game, right after the first 
        cards are given to the players.

        In general the states will be:
        - 0, card still in the deck.
        - 1, card in the hand of the player 1.
        - 2, card in the hand of the player 1 and is a briscola.
        - 3, card in the hand of the player 2.
        - 4, card in the hand of the player 2 and is a briscola.
        - 5, the last card of the deck (known to the players).
        - 6, card played by player 1 and player 1 played first.
        - 7, card played by player 1 and player 1 played second.
        - 8, card played by player 2 and player 2 played first.
        - 9, card played by player 2 and player 2 played second.
        - 10, card already played.
        '''
        state = np.zeros(40)
        
        for card in cards_1:
            state[card.id] = 1
            if card.is_Briscola: state[card.id] = 2

        for card in cards_2:
            state[card.id] = 3
            if card.is_Briscola: state[card.id] = 4

        state[briscola.id] = 5

        return state

    def get_state_for_player(self, player):
        '''
        Function that will give back the player's informations.
        So no info about the cards in the hand of the opposing player.

        For each player the state will be:
        - 0, if the card is still in the deck in the opponent's hand.
        - 1, if the card is in the hand of the player
        - 2, if the card is in the hand of the player and is a briscola
        - 3, the last card of the deck (known to the players).
        - 4, card played by the player and the player played first.
        - 5, card played by the player and the player played second.
        - 6, card played by the other player, and they played first.
        - 7, card played by the other player, and they played second.
        - 8, card already played.
        '''
        player_state = []

        if player == 1:
            # The state the player can be aware of
            known_states = [i for i in range(11) if i not in [3,4]]
            for state in self.state:
                # If the player can't know this state, set it to 0
                if state not in known_states:
                    player_state.append(0)
                elif state > 2: 
                    # Take 2 out since the states regarding player 2 
                    # hand are skipped 
                    player_state.append(state - 2)
                else:
                    player_state.append(state)

        else: 
            # The state the player can be aware of
            known_states = [i for i in range(11) if i not in [1,2]]
            for state in self.state:
                # If the player can't know this state, set it to 0
                if state not in known_states:
                    player_state.append(0)
                elif state in [6, 7]:
                    player_state.append(state)
                elif state in [8, 9]:
                    player_state.append(state - 4)
                else:
                    # Take 2 out since the states regarding player 1 
                    # hand are skipped 
                    player_state.append(state - 2)

        return player_state
    
    # -------------------- STATE UPDATE FUNCTIONS --------------------

    def update_state_after_play(self, card, new_state, on_table = False):
            '''
            Function that update the state after a card is played, 
            in case it's the first card to be played set it as on table.
            '''
            self.state[card.id] = new_state
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

            # Set the state of the cards played during the hand to 10 
            for card in self.card_on_table:
                self.state[card.id] = 10
            
            # Reset the card on table
            self.card_on_table = None

    def update_state_after_draw(self, card_1, card_2):
            '''
            Function that update the state of the cards after the cards
            are drawn. 
            '''

            self.state[card_1.id] = 2 if card_1.is_Briscola else 1
            self.state[card_2.id] = 4 if card_2.is_Briscola else 3


    # ------------------- GAME SIMULATION FUNCTIONS -------------------

    def game_simulation(self):
        '''
        Function that simulate a game of briscola. 
        '''

        done = False

        while(not done):
            self.hand()
            self.update_state_after_hand()

            if not len(self.player_1.cards):
                break

            if len(self.deck.deck):
                card_1 = self.deck.draw_card()
                card_2 = self.deck.draw_card()

                if(self.first_to_play == 1):
                    self.player_1.cards.append(card_1)
                    self.player_2.cards.append(card_2)
                    self.update_state_after_draw(card_1, card_2)
                else:
                    self.player_1.cards.append(card_2)
                    self.player_2.cards.append(card_1)
                    self.update_state_after_draw(card_2, card_1)
                    

        final_score = self.player_1.points - self.player_2.points

    
        '''
        if self.player_1.policy == 'dnq_policy':
            #TODO give reward

        if self.player_1.policy == 'dnq_policy':
            #TODO give reward
        '''
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
            self.update_state_after_play(card_1, 6, True)
            state_2 = self.get_state_for_player(2)
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            self.update_state_after_play(card_2, 9)

        else:
            state_2 = self.get_state_for_player(2)
            card_2 = self.player_2.get_action(state_2, self.card_on_table)
            self.update_state_after_play(card_2, 8, True)
            state_1 = self.get_state_for_player(1)
            card_1 = self.player_1.get_action(state_1,
                                            self.card_on_table)
            self.update_state_after_play(card_1, 7)

        # If there is a q_learning agent and we are training
        if Q_learning != None:
            # Store the states (s, a, r, s'): 
            # s: state before playing
            # a: card played
            # r: reward
            # s': state after playing
            states = {}
            if 1 in Q_learning:
                if self.first_to_play == 1:
                    states[1] = (state_1, card_1.id, 0, state_2)
                else:
                    states[1] = (state_1, card_1.id, 0, 
                                self.get_state_for_player(1))
            elif 2 in Q_learning:
                if self.first_to_play == 2:
                    states[2] = (state_2, card_2.id, 0, state_1)
                else:
                    states[2] = (state_2, card_2.id, 0, 
                                self.get_state_for_player(2))

                    


        self.first_to_play = self.find_hand_winner(card_1, card_2, 
                                            self.player_1, self.player_2,
                                            self.first_to_play)



    def find_hand_winner(self, card_1, card_2, player_1, player_2, first):
        '''
        Function that given the cards player and the order in which 
        they were played find the winner and gives the points accordingly
        '''
        points = card_2.value + card_1.value

        # Both players played a card with the same seed
        if card_1.seed == card_2.seed:
            if card_1.value >= card_2.value and card_1.numb > card_2.numb:
                player_1.points += points
                return 1
            else:
                player_2.points += points
                return 2
        
        # One player played a briscola and the other didn't
        if card_1.is_Briscola:
            player_1.points += points
            return 1
        elif card_2.is_Briscola:
            player_2.points += points
            return 2
        
        # If the seeds are different and there are no briscole then 
        # who played first wins.
        else:
            if first == 1:
                player_1.points += points
                return 1
            else:
                player_2.points += points
                return 2

