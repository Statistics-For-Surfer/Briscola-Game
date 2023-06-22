from _scoring_functions import ace_degenerate_scoring
import numpy as np

def get_idx(card, state):

    n_states = 8

    return card * n_states + state


def draw_new_card(player_board, discard_board, deck, draw, top_discards):

    if (draw == 0):
        new_card = draw_deck_card(player_board, deck)
        new_top_discard = None
    else:
        (new_card, new_top_discard) = draw_discarded_card(player_board, discard_board, draw, top_discards)

    return (new_card, new_top_discard)


def draw_deck_card(player_board, deck):

    new_card = deck.draw_card()

    player_board.add_card(new_card)

    return new_card


def draw_discarded_card(player_board, discard_board, draw_int, top_discards):

    # Suits are assigned integers in in alphabetical order: (deck), clubs, diamonds, hearts, spades
    if (draw_int == 1):
        discarded_cards = discard_board.clubs_discarded
    elif (draw_int == 2):
        discarded_cards = discard_board.diamonds_discarded
    elif (draw_int == 3):
        discarded_cards = discard_board.hearts_discarded
    elif (draw_int == 4):
        discarded_cards = discard_board.spades_discarded

    desired_card = discarded_cards[-1]

    if (len(discarded_cards) == 1):
        top_discards[draw_int-1] = -1
        new_top_discard = None
    else:
        new_top_discard = discarded_cards[-2]
        top_discards[draw_int-1] = new_top_discard

    successful_draw = discard_board.remove_from_board(desired_card)

    player_board.add_card(desired_card)

    return (desired_card, new_top_discard)


def discard_hand_card(player_board, discard_board, card):

    discard_board.add_to_board(card)
    player_board.remove_card(card)


def update_state_discard(state, discard_card, top_discard, player, p1_obs, p2_obs):

    suit = int(discard_card / 13)
    top_discarded_card = top_discard[suit]

    # Check if there is currently a top discarded card
    if (top_discarded_card != -1):

        # Check if player 1 has the current top discarded card and update accordingly
        if (state[get_idx(top_discarded_card, 7)] == 1):

            new_discarded_card_state = 6

        elif (state[get_idx(top_discarded_card, 9)] == 1): 

            new_discarded_card_state = 8

        else:

            print('Error')
            for state_opt in range(11):
                print(state[get_idx(top_discarded_card, state_opt)])

        new_idx = get_idx(top_discarded_card, new_discarded_card_state)

        zero_card(state, top_discarded_card)
        zero_card(p1_obs, top_discarded_card)
        zero_card(p2_obs, top_discarded_card)

        state[new_idx] = 1
        p1_obs[new_idx] = 1
        p2_obs[new_idx] = 1

    # Change the discarded card to the top of its respective suit
    if (player == 1):

        new_state = 7

    elif (player == 2):

        new_state = 9

    new_idx = get_idx(discard_card, new_state)

    zero_card(state, discard_card) 
    zero_card(p1_obs, discard_card) 
    zero_card(p2_obs, discard_card) 

    state[new_idx] = 1
    p1_obs[new_idx] = 1
    p2_obs[new_idx] = 1

    # Update the list of top discarded cards
    top_discard[suit] = discard_card


def update_state_deck_draw(state, new_card, player, p1_obs, p2_obs):

    old_state = 10

    if (player == 1):

        new_state = 0
        update_p1 = True

    elif (player == 2):

        new_state = 3
        update_p1 = False

    old_idx = get_idx(new_card, old_state)
    new_idx = get_idx(new_card, new_state)

    state[old_idx] = 0
    state[new_idx] = 1

    if update_p1:

        p1_obs[old_idx] = 0
        p1_obs[new_idx] = 1

    else:

        p2_obs[old_idx] = 0
        p2_obs[new_idx] = 1


def update_state_discard_draw_new_card(state, new_card, player, p1_obs, p2_obs):

    zero_card(state, new_card)
    zero_card(p1_obs, new_card)
    zero_card(p2_obs, new_card)

    if (player == 1):

        new_state = 0

    elif (player == 2):

        new_state = 3

    new_idx = get_idx(new_card, new_state)

    state[new_idx] = 1
    p1_obs[new_idx] = 1
    p2_obs[new_idx] = 1


def update_state_discard_draw_new_top_card(state, new_top_card, player, p1_obs, p2_obs):

    if (new_top_card is None):
        return

    # Check if player 1 has the current top discarded card and update accordingly
    if (state[get_idx(new_top_card, 6)] == 1):

        new_state = 7

    elif (state[get_idx(new_top_card, 8)] == 1): 

        new_state = 9

    new_idx = get_idx(new_top_card, new_state)

    zero_card(state, new_top_card)
    zero_card(p1_obs, new_top_card)
    zero_card(p2_obs, new_top_card)

    state[new_idx] = 1
    p1_obs[new_idx] = 1
    p2_obs[new_idx] = 1


def update_state_draw(state, new_card, new_top_card, player, p1_obs, p2_obs, draw_int):
    # Change card state from deck/discard to hand

    if (draw_int == 0):

        update_state_deck_draw(state, new_card, player, p1_obs, p2_obs)

    else:

        update_state_discard_draw_new_card(state, new_card, player, p1_obs, p2_obs)
        update_state_discard_draw_new_top_card(state, new_top_card, player, p1_obs, p2_obs)


def zero_card(arr, card):
    '''Remove any state information about a given card.'''

    for state_opt in range(11):
        arr[get_idx(card, state_opt)] = 0


def update_state_play(state, play_card, player, player_board, player_board2 , p1_obs, p2_obs):
    '''Update both the state and the flex_options list after a new card is played. At this point
    the hand has been updated but these variables have not, so flex options contains information 
    about flex options before current card played. 'Flex option' elements are:

    -1: suit initialized
    0: no current flex play
    '''
    # Update the state of the card played
    new_state = 4
    if player_board.card_on_table == None:
        new_state = 5 
    # Update the state of the card on table
    else:
        state[get_idx(player_board.card_on_table,5)] = 0
        state[get_idx(player_board.card_on_table,4)] = 1
    state[get_idx(play_card,new_state)] = 1
    
    #state[get_idx(play_card,)]


        


    # If there was not a previous flex option, we only need to change the state of the new card


    new_idx = get_idx(play_card, new_state)

    zero_card(state, play_card)
    zero_card(p1_obs, play_card)
    zero_card(p2_obs, play_card)

    state[new_idx] = 1
    p1_obs[new_idx] = 1
    p2_obs[new_idx] = 1


def check_trivial_turn(card, draw_int, top_discards):

    suit = int(card / 13)

    if ((draw_int-1) == suit):
        return True

    return False


def make_move(player_board, deck, action, state, flex_options, player, p1_obs, p2_obs):

    card = action[0]
    play = action[1]

    #print 'initial cards in p1 hand: '
    #for card_opt in range(52):
    #    if (state[get_idx(card_opt, 0)] == 1):
    #        print card_opt

    #print 'initial cards in p2 hand: '
    #for card_opt in range(52):
    #    if (state[get_idx(card_opt, 3)] == 1):
    #        print card_opt

    if play:
        player_board.play_card(card)
        update_state_play(state, card, flex_options, player, player_board, p1_obs, p2_obs)

    else:

        trivial = check_trivial_turn(card, draw_int, top_discard)

        if trivial:
            return

        discard_hand_card(player_board, discard_board, card)
        update_state_discard(state, card, top_discard, player, p1_obs, p2_obs)

    (new_card, new_top_discard) = draw_new_card(player_board, discard_board, deck, draw_int, top_discard)
    update_state_draw(state, new_card, new_top_discard, player, p1_obs, p2_obs, draw_int)
