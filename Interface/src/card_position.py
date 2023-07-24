

class CardPosition(object):
    def __init__(self, pos_function, game):
        self.name = pos_function
        self.x, self.y = self.get_position(game)

    def get_position(self, game):
        if self.name == 'deck':
            return game.WIDTH - game.card_space_w, (game.HEIGHT - game.card_h)/2
        elif self.name == 'trump':
            return game.WIDTH - game.card_h/2 - game.card_space_w, game.HEIGHT/2 - game.card_h/4
        elif self.name == 'player_card_1':
            return game.first_card_pos + (game.card_space_w * 0), game.HEIGHT - game.card_h - game.card_space_h
        elif self.name == 'player_card_2':
            return game.first_card_pos + (game.card_space_w * 1), game.HEIGHT - game.card_h - game.card_space_h
        elif self.name == 'player_card_3':
            return game.first_card_pos + (game.card_space_w * 2), game.HEIGHT - game.card_h - game.card_space_h
        elif self.name == 'bot_card_1':
            return game.first_card_pos + (game.card_space_w * 0), game.card_space_h
        elif self.name == 'bot_card_2':
            return game.first_card_pos + (game.card_space_w * 1), game.card_space_h
        elif self.name == 'bot_card_3':
            return game.first_card_pos + (game.card_space_w * 2), game.card_space_h
        elif self.name == 'bot_played_card':
            return game.WIDTH/2 - game.card_w/2 + game.card_space_w/2, game.HEIGHT/2 - game.card_h/2
        elif self.name == 'player_played_card':
            return game.WIDTH/2 - game.card_w/2 - game.card_space_w/2, game.HEIGHT/2 - game.card_h/2
