import os
import numpy as np
import pandas as pd
import random



class Briscola_game():
    def __init__(self):
        # deck, tuple of seed, card , and value
        self.deck = self.create_deck()
        random.shuffle(self.deck)
        self.winner = 0
        self.player_1 = self.draw_cards()
        self.player_2 = self.draw_cards()
        self.last_card = self.deck.pop()
        self.briscola =  self.last_card[0]
        self.deck = [self.last_card]+ self.deck
        self.scores = {0:0 , 1:0}
        self.played_cards = []
        self.card_on_table = None



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
        self.player = self.draw_hand()
        #self.dealer = [self.draw_card()]
        self.sarsp = []
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*4
        random.shuffle(self.deck)
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

    

    def get_action(self,player,state):
        if self.action_type == 'input':
            action = int(input('Hit (1) or Pass (0): '))
        elif self.action_type == 'random_policy':
            action = np.random.randint(len(player)- 1)
            played_card = player.pop(action)
        elif self.action_type == 'fixed_policy':
            action = self.p
        return played_card
    

    def draw_action(self, player):
        player.append(self.deck.pop())
        return
    
    def create_deck(self):
        deck = []
        seeds = ['Bastoni', 'Denara' , 'Coppe' , 'Spade']
        cards = [1,2,3,4,5,6,7,8,9,10]
        values = [11,0,10,0,0,0,0,2,3,4]
        for seed in seeds:
            for j , card in enumerate(cards):
                deck.append((seed, card, values[j]))
        return deck
    



    def hand_to_state(self, player, mapping):
            '''boh ci si pensa'''
            state_dic = {}
            state_dic['cards'] = player
            state_dic['score_1'] = self.score_1
            state_dic['score_2'] = self.score_2
            state_dic['played_card'] = self.played_cards
            state_dic['card_on_the_table'] =  self.card_on_table if mapping == 2 else None
            state_dic['last_card'] = self.briscola
            return state_dic

    def hand(self, winner):
        if winner == 0:
            self.state = self.hand_to_state(self,self.player_1,1)
            card_1 = self.get_action(self, self.player_1, self.state)
            self.card_on_table = card_1
            self.state = self.hand_to_state(self , self.player_2,2)
            card_2 = self.get_action(self , self.player_2, self.state)
        else:
            self.state = self.hand_to_state(self , self.player_2,1)
            card_1 = self.get_action(self, self.player_2, self.state)
            self.card_on_table = card_1
            self.state = self.hand_to_state(self , self.player_1,2)
            card_2 = self.get_action(self, self.player_1, self.state)
            
        self.played_cards.append(card_1 ,card_2)

        '''caso briscole'''
        if card_1[0] == self.briscola[0] and card_2[0] != self.briscola[0]:
            self.scores[winner]+= card_1[2] + card_2[2]
            self.winner = winner
        elif card_1[0] != self.briscola[0] and card_2[0] == self.briscola[0]:
            self.scores[abs(1-winner)] += card_1[2] + card_2[2]
            self.winner = abs(1-winner)
        else:
            '''
            caso non briscole
            '''        
            
            if card_1[2] >= card_2[2]:
                self.scores[winner] += card_1[2] + card_2[2]
                self.winner = card_1[1] < card_2[1]
            elif card_1[2] < card_2[2] and card_1[0] == card_2[0]:
                self.scores[abs(1-winner)] += card_1[2] + card_2[2]
                self.winner = abs(1-winner)
            else:
                self.scores[winner] += card_1[2] + card_2[2]
                self.winner = winner            
        return winner


    def play_game(self):
        # Only for ’input’ mode
        if self.verbose:
            print('New Game!\n')
        # Iterate through game
        done = False
        while(not done):
            winner = self.hand(self.winner)
            if len(self.deck != 0):
                if winner == 0:
                    self.draw_action(self.player_1)
                    self.draw_action(self.player_2)
                else:
                    self.draw_action(self.player_2)
                    self.draw_action(self.player_1)
            if len(self.player_1 == 0):
                done = True
        if self.score_1 > self.score_2:
            final_winner = 0
        else: 
            final_winner = 1
        return final_winner
        
        


