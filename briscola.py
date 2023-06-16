import os
import numpy as np
import pandas as pd
import random



class Briscola_game():
    def __init__(self, params):
    # deck, tuple of seed, card , and value
    self.deck = [('Bastoni', 1, 11) , ('Bastoni', 2, 0),  ('Bastoni', 3, 10)] #TODO 
    random.shuffle(self.deck)
    self.winner = 0
    self.player_1 = self.draw_cards()
    self.player_2 = self.draw_cards()
    self.briscola = random.choice(self.deck)
    self.last_card =self.deck.pop(self.briscola)
    self.deck = self.deck.append(last_card)

    self.score_1 = 0
    self.score_2 = 0



    # State, Action, Reward, Next State arrays
    #self.sarsp = []
    #self.sarsp_arr = np.array([], dtype=’int’).reshape(0,4)
    self.action_type = params.action_type # ’input’, ’random_policy’, ’fixed_policy’
    self.verbose = (params.action_type == ’input’)
    self.num_games = params.num_games
    self.fixed_policy_filepath = params.fixed_policy_filepath
    self.policy = self.load_policy()
    self.state_mapping = params.state_mapping


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
    if self.action_type in ['random_policy', ’input’]:
                            return None


def print_iter(self):
    if not self.verbose:
        return
    print(f'Player_1 hand: {self.player}\t\t sum: {self.sum_hand(self.player)}')
    print(f'Player_2 hand: {self.dealer}\t\t sum: {self.sum_hand(self.dealer)}')
    return

def get_action(self, state):
    if self.action_type == ’input’:
        action = int(input(’Hit (1) or Pass (0): ’))
    elif self.action_type == 'random_policy':
        action = np.random.randint(2)
    elif self.action_type == 'fixed_policy':
        action = self.p
    return action 


def hand_to_state(self, player, dealer):
    if self.state_mapping == 1:  # Primo a giocare
        state_dic = {}
        state_dic['cards'] = self.player
        state_dic['score_1'] = self.score_1
        state_dic['score_2'] = self.score_2
        # [TODO] carte che sono uscite
        return state_dic
    elif self.state_mapping == 2: # secondo a giocare
        '''boh ci si pensa'''
        state_dic = {}
        state_dic['cards'] = self.player
        state_dic['score_1'] = self.score_1
        state_dic['score_2'] = self.score_2
        # [TODO] carte che sono uscite
        state_dic['card_on_the_table'] =  'boh'

        return state_dic

def mano(self, action, winner):
    if winner == 1:
        card_1 = self.player_1.pop(action_1)
        action_2 = get_action(self, state)
        card_2 = self.player_2.pop(action_2)
    else:
        card_2 = self.player_1.pop(action_1)
        action_1 = get_action(self, state)
        card_1 = self.player_2.pop(action_1)
    

    if card_1[0] == self.briscola[0] and card_2[0] != self.briscola[0]:
        self.score_1 += card_1[2] + card_2[2]
        self.winner = 0
    elif card_1[0] != self.briscola[0] and card_2[0] == self.briscola[0]:
        self.score_2 += card_1[2] + card_2[2]
        self.winner = 1
    else:
        if card_1[2] > card_2[2]:
            self.score_1 += card_1[2] + card_2[2]
            self.winner = 0
        if card_1[2] < card_2[2]:
            self.score_2 += card_1[2] + card_2[2]
            self.winner = 1
        else:
            self.winner = card_1[1] < card_2[1]
    
    return 




def play_game(self):
    # Only for ’input’ mode
    if self.verbose:
        print('New Game!\n')
    # Iterate through game
    done = False
    while(not done):
    # Only for ’input’ mode
        self.print_iter()
        # Current state/action/reward
        state = self.hand_to_state(self.player, self.dealer)
        action = self.get_action(state)
        #reward = self.get_reward(state, action, self.player, self.dealer)
        mano(self, action)
        if len(self.deck == 0):
            done = True
    # chek the final winner

    if self.score_1 > self.score_2:
        final_winner = 0
    else: 
        final_winner = 1
    
    


