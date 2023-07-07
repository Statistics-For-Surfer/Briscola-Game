def unpack_action(action):

    cards_in_deck = 40
    
    remainder = action % (2 * cards_in_deck)

    play = int(remainder / cards_in_deck)
    card_int = remainder % cards_in_deck

    return (card_int, play)


def pack_action(card_int, play):

    cards_in_deck = 40

    if play:
        row_val = card_int + cards_in_deck
    else:
        row_val = card_int

    offset = (2 * cards_in_deck)

    return row_val + offset
