import os
import numpy as np
import pandas as pd
import random



class Briscola_game():
    '''
    Classe per eseguire simulazioni di gioco per il gioco di carte "Briscola".
    '''


    def __init__(self):
        # deck, tuple of seed, card , and value

        # Get the new deck 
        self.deck = self.create_deck()
        # Shuffle the cards
        random.shuffle(self.deck)

        # Select the "winner" ie in the first hand is going to be the player that starts to play 
        self.winner = np.random.randint(2)
        
        # Let both players collect their first cards
        self.player_1 = self.draw_cards()
        self.player_2 = self.draw_cards()

        # Select the seed for the briscola and put the drawn cards as the last in the deck
        self.last_card = self.deck.pop()
        self.briscola =  self.last_card[0]
        self.deck = [self.last_card] + self.deck

        # Initialize the other fields
        self.scores = {0:0 , 1:0}
        self.played_cards = []
        self.card_on_table = None

        # Select the policy to be used
        #self.action_type = 'random_policy'
        self.action_type = 'greedy_policy'


        # State, Action, Reward, Next State arrays
        #self.sarsp = []
        #self.sarsp_arr = np.array([], dtype=’int’).reshape(0,4)
        #self.action_type = params.action_type # ’input’, ’random_policy’, ’fixed_policy’
        #self.verbose = (params.action_type == ’input’)
        #self.num_games = params.num_games
        #self.fixed_policy_filepath = params.fixed_policy_filepath
        #self.policy = self.load_policy()
        #self.state_mapping = params.state_mapping


        # Probably do not need to change these
        #self.lose_state = 0
        #self.win_state = 1
        #self.terminal_state = 2
        # Also do not need to change these
        self.lose_reward = -10
        self.win_reward = 10
        return


    def reset(self):
        '''
        Ripristina il mazzo dopo una partita
        '''
        # Reinitialize the class attributes like it was done in the constructor

        # The deck
        self.deck = self.create_deck()
        random.shuffle(self.deck)

        # First to play in the new game
        self.winner = np.random.randint(2)

        # Cards
        self.player_1 = self.draw_cards()
        self.player_2 = self.draw_cards()

        # Briscola
        self.last_card = self.deck.pop()
        self.briscola =  self.last_card[0]
        self.deck = [self.last_card] + self.deck

        # Other fields
        self.scores = {0:0 , 1:0}
        self.played_cards = []
        self.card_on_table = None
        self.action_type = 'random_policy'



        # State, Action, Reward, Next State arrays
        #self.sarsp = []
        #self.sarsp_arr = np.array([], dtype=’int’).reshape(0,4)
        #self.action_type = params.action_type # ’input’, ’random_policy’, ’fixed_policy’
        #self.verbose = (params.action_type == ’input’)
        #self.num_games = params.num_games
        #self.fixed_policy_filepath = params.fixed_policy_filepath
        #self.policy = self.load_policy()
        #self.state_mapping = params.state_mapping


        # Probably do not need to change these
        #self.lose_state = 0
        #self.win_state = 1
        #self.terminal_state = 2
        # Also do not need to change these
        self.lose_reward = -10
        self.win_reward = 10
        return



    def draw_cards(self):
        '''
        lista di 3 carte prese dal deck
        '''
        cards = [self.deck.pop() for i in range(3)]
        return cards

    #def hand_to_state(self, player):
    #    if self.state_mapping == 1:
    #        return self.sum_hand(player) - 1
    #    elif self.state_mapping == 2:
    #        return (self.sum_hand(player) - 1) + (18 * (dealer[0] - 1))


    def load_policy(self):
        # Policy not needed if a user is playing or a random policy is being used
        if self.action_type in ['random_policy', 'input']:
            return None


    def print_iter(self):
        if not self.verbose:
            return
        print(f'Player_1 hand: {self.player}\t\t sum: {self.sum_hand(self.player)}')
        print(f'Player_2 hand: {self.dealer}\t\t sum: {self.sum_hand(self.dealer)}')
        return



    def greedy_action(self, player ,state):
        player_seed  = [x[0] for x in player]
        player_card  = [x[1] for x in player]
        player_value = [x[2] for x in player]
        possible_action = []


        if self.card_on_table == None:
            return self.max_min_values(player, maxx = False)
        else:
            # The card on the table is a briscola
            if self.card_on_table[0] == self.briscola:
                # check if you have a briscola greater than the one on the table
                for i in range(len(player)):
                    if player_seed[i] == self.card_on_table[0] and player_value[i] > self.card_on_table[2]:
                        possible_action.append(player[i])
            else:
                for i in range(len(player)):
                    if (player_seed[i] == self.card_on_table[0] and  player_value > self.card_on_table[2]) or player_seed == self.briscola:
                        possible_action.append(player)
        # I cannot take
        if len(possible_action) == 0:
            # I want to put the card with less points
            return self.max_min_values(possible_action, maxx = False)

        else:
            # I want to put the card with more points
            return self.max_min_values(possible_action, maxx = True)


    def max_min_values(self,player, maxx = True):
        if maxx == True:
            max_tuple = None
            max_value = float('-inf')

            for tuple in player:
                if tuple[2] > max_value:
                    max_value = tuple[2]
                    max_tuple = tuple

            return max_tuple
        else:
            min_tuple = None
            min_value = float('-inf')

            for tuple in player:
                if tuple[2] < min_value:
                    min_value = tuple[2]
                    min_tuple = tuple
            return min_tuple

    def get_action(self,player,state):
        '''
        Scelta della carta da giocare in base alla policy
        '''
        if self.action_type == 'input':
            action = int(input('Hit (1) or Pass (0): '))
        elif self.action_type == 'random_policy':
            # Select and play a random card
            action = np.random.randint(len(player))
            played_card = player.pop(action)
        elif self.action_type == 'greedy_policy':
            action = self.greedy_action(player,state)
            print(action)
            #played_card = player.pop(action)
        return played_card
    

    def draw_action(self, player):
        '''
        Per pescare una carta
        '''
        if len(self.deck) != 0:
            player.append(self.deck.pop())
        return
    
    def create_deck(self):
        '''
        Creare mazzo iniziale
        '''
        deck = []
        seeds = ['Bastoni', 'Denara' , 'Coppe' , 'Spade']
        cards = [1,2,3,4,5,6,7,8,9,10]
        values = [11,0,10,0,0,0,0,2,3,4]

        # Create the list of tuples that represent each card of the deck
        for seed in seeds:
            for j , card in enumerate(cards):
                deck.append((seed, card, values[j]))
        return deck


    def hand_to_state(self , player, mapping = False):
            '''
            Creare dizionario che rappresenta lo stato.
            
            Prende in input il giocatore che sta per giocare e un booleano che indica se il giocatore gioca per secondo.
            '''
            state_dic = {}
            # Cards in the player hand
            state_dic['cards'] = player
            # The current score
            state_dic['score_1'] = self.scores[0]
            state_dic['score_2'] = self.scores[1]
            # All the cards that were already played
            state_dic['played_card'] = self.played_cards
            # If the other player already played their card than we have a card on the table
            state_dic['card_on_the_table'] =  self.card_on_table if mapping else None
            # The seed of the briscola
            state_dic['last_card'] = self.briscola
            return state_dic

    def hand(self, winner: int):
        '''
        Simulazione di una mano di briscola dato in input il giocatore che deve giocare per primo. 

        Aggiorna il punteggio del giocatore che vince e restituisce come output 0 o 1 in base al giocatore the ha vinto la mano. 
        '''
        # Player 1 has to play first
        if winner == 0:
            # Update the state
            self.state = self.hand_to_state(self.player_1)
            # Card played by player 1
            card_1 = self.get_action(self.player_1, self.state)
            # Change and update the state
            self.card_on_table = card_1
            self.state = self.hand_to_state(self.player_2, True)
            # Card played by player 2
            card_2 = self.get_action(self.player_2, self.state)
        # Player 2 is the one that has to play
        else:
            # Update the state
            self.state = self.hand_to_state(self.player_2)
            # Card played by player 2
            card_1 = self.get_action(self.player_2, self.state)
            # Change and update the state
            self.card_on_table = card_1
            self.state = self.hand_to_state(self.player_1, True)
            # Card played by player 1
            card_2 = self.get_action(self.player_1, self.state)

        # Add the played cards to the list of the cards already played
        self.played_cards.extend((card_1 ,card_2))

        # Select the winner of the hand
        '''caso briscole'''
        if card_1[0] == self.briscola and card_2[0] != self.briscola:
            self.scores[winner]+= card_1[2] + card_2[2]
            self.winner = winner
        elif card_1[0] != self.briscola and card_2[0] == self.briscola:
            self.scores[1-winner] += card_1[2] + card_2[2]
            self.winner = 1-winner
        
        else:
            '''caso non briscole''' 
            if card_1[2] >= card_2[2]:
                self.scores[winner] += card_1[2] + card_2[2]
                self.winner = 1-winner if card_1[1] < card_2[1] and card_1[0] == card_2[0] else winner
            elif card_1[2] < card_2[2] and card_1[0] == card_2[0]:
                self.scores[1-winner] += card_1[2] + card_2[2]
                self.winner = 1-winner
            else:
                self.scores[winner] += card_1[2] + card_2[2]
                self.winner = winner            
        return self.winner


    def play_game(self):
        '''
        Simulazione di una partita di briscola.
        Dopo aver simulato un'intera partita restituisce 0, 1 o -1.
        - 0 per la vittoria del player 1.
        - 1 per la vittoria del player 2.
        - -1 nel caso di pareggio.

        Alla fine il mazzo viene resettato.
        '''
        # Only for ’input’ mode
        #if self.verbose:
        #    print('New Game!\n')
        # Iterate through game
        final_winner = None
        done = False
        while(not done):
            # Simulate a hand
            winner = self.hand(self.winner)
            # Draw if there are still cards in the deck 
            if len(self.deck) != 0:
                if winner == 0:
                    self.draw_action(self.player_1)
                    self.draw_action(self.player_2)
                else:
                    self.draw_action(self.player_2)
                    self.draw_action(self.player_1)
            # If the cards are finished stop the game
            if len(self.player_1) == 0:
                done = True
        # Check who won
        if self.scores[0] > self.scores[1]:
            final_winner = 0
        elif self.scores[0] < self.scores[1]: 
            final_winner = 1
        else:
            final_winner = -1

        # Reset the deck
        self.reset()
        
        # return the winner
        return final_winner
        
        


