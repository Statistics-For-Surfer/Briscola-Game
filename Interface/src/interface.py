import random
import pygame



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



class BriscolaApp(object):
    def __init__(self):

        pygame.display.set_caption('Pausetta Briscola?')
        pygame.init()

        self.WIDTH = 1200
        self.HEIGHT = self.WIDTH / 3 * 2
        self.screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT])
        self.card_w = self.WIDTH * .097
        # self.card_h = self.card_w * 9.4 / 5
        self.card_h = self.card_w * 285 / 177
        self.card_space_w = self.card_w + 20
        self.card_space_h = self.HEIGHT / 33
        self.first_card_pos = self.WIDTH/2 - (self.card_w *.5) - self.card_space_w
        self.card_smoothing = 10
        self.card_backside = self.load_card_img("Interface/images/card_backside.png")
        self.logo = pygame.image.load("Interface/images/big_logo.png")
        self.small_logo = pygame.image.load("Interface/images/small_logo.png")
        self.table_color = '#207438'
        self.font = pygame.font.SysFont('candara', int(self.WIDTH * .029))
        self.button_font = pygame.font.SysFont('georgia', int(self.WIDTH * .031))
        self.level_font = pygame.font.SysFont('georgia', int(self.WIDTH * .019))
        self.label_color = 'black'
        self.running = True
        self.active = False
        self.reactive = False
        self.level = 0


    def clean_values(self):
        '''Clean saved values from previous game'''

        self.len_virtual_deck = 33 # Start with 33
        self.player_turn = bool(random.randint(0, 1))
        self.last_two_hand = False
        self.bot_points, self.player_points, self.points = 0, 0, 0
        self.rect_img_dict, self.img_card_dict = {}, {}
        self.img_card_match()
        self.virtual_deck = list(self.img_card_dict.items())
        random.shuffle(self.virtual_deck)
        self.over = False
        self.last_hand = False


    def start_game(self):
        '''Create initial page'''

        self.screen.fill(self.table_color)
        w, h = 300, 100
        x, y = (self.WIDTH - w) /2, (self.HEIGHT - h) /2

        self.start_button = pygame.draw.rect(self.screen, 'white', [x, y, w, h], 0, self.card_smoothing)
        pygame.draw.rect(self.screen, 'darkgray', [x, y, w, h], 7, self.card_smoothing)
        pygame.draw.rect(self.screen, 'black', [x, y, w, h], 5, self.card_smoothing)

        text = self.button_font.render('START', True, 'black')
        self.screen.blit(text, text.get_rect(center = (x+w/2, y+h/2)))
        self.screen.blit(self.logo, self.logo.get_rect(center = (self.WIDTH/2, self.HEIGHT/4)))
        self.levels_interface()


    def levels_interface(self):
        w, h = 300, 100
        level_text = self.button_font.render('Select game level', True, 'black')
        self.screen.blit(level_text, level_text.get_rect(center = (self.WIDTH/2, self.HEIGHT/6*4.2)))

        self.easy_level_button = pygame.draw.rect(self.screen, self.table_color, [self.WIDTH/2 - self.card_w*1.42, self.HEIGHT/6*4.5, w/4, h/2], 0)
        easy_level = self.level_font.render('Easy', True, 'black')
        self.screen.blit(easy_level, easy_level.get_rect(center = (self.WIDTH/2 - self.card_w*1.1, self.HEIGHT/6*5)))

        self.medium_level_button = pygame.draw.rect(self.screen, self.table_color, [self.WIDTH/2 - self.card_w*0.32, self.HEIGHT/6*4.5, w/4, h/2], 0)
        medium_level = self.level_font.render('Intermediate', True, 'black')
        self.screen.blit(medium_level, medium_level.get_rect(center = (self.WIDTH/2, self.HEIGHT/6*5)))

        self.hard_level_button = pygame.draw.rect(self.screen, self.table_color, [self.WIDTH/2 + self.card_w*0.78, self.HEIGHT/6*4.5, w/4, h/2], 0)
        hard_level = self.level_font.render('Hard', True, 'black')
        self.screen.blit(hard_level, hard_level.get_rect(center = (self.WIDTH/2 + self.card_w*1.1, self.HEIGHT/6*5)))
        self.select_level()


    def select_level(self):

        empty_color = 'gray'
        pygame.draw.circle(self.screen, empty_color, (self.WIDTH/2 - self.card_w*1.1, self.HEIGHT/6*4.7), 7)
        pygame.draw.circle(self.screen, empty_color, (self.WIDTH/2, self.HEIGHT/6*4.7), 7)
        pygame.draw.circle(self.screen, empty_color, (self.WIDTH/2 + self.card_w*1.1, self.HEIGHT/6*4.7), 7)

        if self.level == 1:
            pygame.draw.circle(self.screen, 'black', (self.WIDTH/2 - self.card_w*1.1, self.HEIGHT/6*4.7), 5)
            
        elif self.level == 3:
            pygame.draw.circle(self.screen, 'black', (self.WIDTH/2 + self.card_w*1.1, self.HEIGHT/6*4.7), 5)
             
        else:
            pygame.draw.circle(self.screen, 'black', (self.WIDTH/2, self.HEIGHT/6*4.7), 5)


    def build_game(self):
        '''Build the screen of the game'''

        self.screen.fill(self.table_color)
        self.deck = CardPosition('deck', self)
        self.trump = CardPosition('trump', self)
        self.player_card_1 = CardPosition('player_card_1', self)
        self.player_card_2 = CardPosition('player_card_2', self)
        self.player_card_3 = CardPosition('player_card_3', self)
        self.bot_card_1 = CardPosition('bot_card_1', self)
        self.bot_card_2 = CardPosition('bot_card_2', self)
        self.bot_card_3 = CardPosition('bot_card_3', self) 
        self.player_played_card = CardPosition('player_played_card', self)
        self.bot_played_card = CardPosition('bot_played_card', self)
        self.deck_pos = pygame.draw.rect(self.screen, self.table_color, [self.deck.x, self.deck.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.trump_pos = pygame.draw.rect(self.screen, self.table_color, [self.trump.x, self.trump.y, self.card_h, self.card_w], 0, self.card_smoothing)
        self.player_card_1_pos = pygame.draw.rect(self.screen, self.table_color, [self.player_card_1.x, self.player_card_1.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.player_card_2_pos = pygame.draw.rect(self.screen, self.table_color, [self.player_card_2.x, self.player_card_2.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.player_card_3_pos = pygame.draw.rect(self.screen, self.table_color, [self.player_card_3.x, self.player_card_3.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.bot_card_1_pos = pygame.draw.rect(self.screen, self.table_color, [self.bot_card_1.x, self.bot_card_1.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.bot_card_2_pos = pygame.draw.rect(self.screen, self.table_color, [self.bot_card_2.x, self.bot_card_2.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.bot_card_3_pos = pygame.draw.rect(self.screen, self.table_color, [self.bot_card_3.x, self.bot_card_3.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.bot_played_card_pos = pygame.draw.rect(self.screen, self.table_color, [self.bot_played_card.x, self.bot_played_card.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.player_played_card_pos = pygame.draw.rect(self.screen, self.table_color, [self.player_played_card.x, self.player_played_card.y, self.card_w, self.card_h], 0, self.card_smoothing)
        self.bot_hand = [self.bot_card_1_pos, self.bot_card_2_pos, self.bot_card_3_pos]

        # ADD CARD IMAGES
        
        # Trump
        path, card_id = self.virtual_deck.pop(0)
        img = pygame.transform.rotate(self.load_card_img(path), 90)
        self.rect_img_dict[str(self.trump_pos)] = (img, card_id)
        self.screen.blit(img, self.trump_pos)

        # Deck
        self.screen.blit(self.card_backside, self.deck_pos)
        
        # Player cards
        for pos in [self.player_card_1_pos, self.player_card_2_pos, self.player_card_3_pos]:
            path, card_id = self.virtual_deck.pop(0)
            img = self.load_card_img(path)
            self.screen.blit(img, pos)
            self.rect_img_dict[str(pos)] = (img, card_id)

        # Bot cards
        for pos in [self.bot_card_1_pos, self.bot_card_2_pos, self.bot_card_3_pos]:
            self.screen.blit(self.card_backside, pos)
            path, card_id = self.virtual_deck.pop(0)
            img = self.load_card_img(path)
            self.rect_img_dict[str(pos)] = (img, card_id)

        # Logo
        self.screen.blit(self.small_logo, self.small_logo.get_rect(center = (self.WIDTH/7, self.HEIGHT/2)))

        self.update_cards_left()
        self.update_turn()
        self.update_score()


    def select_card(self, pos):
        '''Put the card selected by the player in the middle'''

        # Take card img and info.
        img, card_id = self.rect_img_dict[str(pos)]

        # Remove position from mapping dictionary.
        del self.rect_img_dict[str(pos)]

        # Add the card in the middle.
        self.screen.blit(img, self.player_played_card_pos)

        # Save the mapping for player middle card.
        self.rect_img_dict[str(self.player_played_card_pos)] = (img, card_id)

        # Set as free the old card's position.
        self.player_free_position = pos

        # Cover the old card position.
        pygame.draw.rect(self.screen, self.table_color, pos)


    def select_bot_card(self):
        '''Card selected by the bot'''

        # Add pause to make the game more enjoyble.
        pygame.time.wait(1000)

        # [INSERT BOT BRAIN]
        # Check if the game is arrived to last two hand to update bot_hand.
        if self.len_virtual_deck == 0:
            if not self.last_two_hand:
                self.last_two_hand = True
            else:
                self.bot_hand = [x for x in self.bot_hand if x != self.bot_free_position]

        # Condition for game over.
        if len(self.bot_hand) == 1:
            self.over = True
 
        # Remove previous position.
        pos = None

        # EASY LEVEL: random choice.
        if self.level == 1:
            pos = random.choice(self.bot_hand)

        # INTERMEDIATE LEVEL: greedy choice.
        elif self.level == 2:
            print('I still not have a greedy brain!!')
            self.running = False

        # HARD LEVEL: reinforced choice.
        elif self.level == 3:
            print('I still not have a reinforced brain!!')
            self.running = False

        img, card_id = self.rect_img_dict[str(pos)]

        # Remove position from mapping dictionary.
        del self.rect_img_dict[str(pos)]

        # Add card image in the middle and map its new position.
        self.screen.blit(img, self.bot_played_card_pos)
        self.rect_img_dict[str(self.bot_played_card_pos)] = (img, card_id)

        # Set old position to free and cover it.
        self.bot_free_position = pos
        pygame.draw.rect(self.screen, self.table_color, pos)


    def img_card_match(self):
        '''Map images and cards'''

        seeds = ['B', 'D', 'C', 'S']
        cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        values = [11, 0, 10, 0, 0, 0, 0, 2, 3, 4]
        for seed in seeds:
            for j, card in enumerate(cards):
                self.img_card_dict[f'Interface/images/cards/{seed}_{card}.png'] = (seed, card, values[j])


    def load_card_img(self, path):
        '''Load and resize image of the card'''

        img = pygame.image.load(path)
        img = img.convert_alpha()
        img = pygame.transform.smoothscale(img, (self.card_w, self.card_h))
        img = pygame.transform.scale(img, (self.card_w, self.card_h))

        return img


    def check_hand_winner(self):
        '''Check who win the hand (True: player wins | False: bot wins)'''

        # Pause to make the game more enjoyble.
        pygame.time.wait(1500)

        # Get cards info to compare them.
        player_card = self.rect_img_dict[str(self.player_played_card_pos)][1]
        bot_card = self.rect_img_dict[str(self.bot_played_card_pos)][1]
        trump = self.rect_img_dict[str(self.trump_pos)][1][0]

        # Remove mapping of both card in the middle and cover them.
        del self.rect_img_dict[str(self.player_played_card_pos)]
        del self.rect_img_dict[str(self.bot_played_card_pos)]
        pygame.draw.rect(self.screen, self.table_color, self.player_played_card_pos)
        pygame.draw.rect(self.screen, self.table_color, self.bot_played_card_pos)

        # Save the total points of the hand.
        points = player_card[2] + bot_card[2]

        # Check who win the hand.
        if player_card[0] == bot_card[0]:
            if player_card[2] > bot_card[2]:
                self.player_turn, self.points = True, points
                self.player_points += self.points
            else:
                self.player_turn, self.points = False, points
                self.bot_points += self.points
        else:
            if bot_card[0] == trump and player_card[0] != trump:
                self.player_turn, self.points = False, points
                self.bot_points += self.points
            elif bot_card[0] != trump and player_card[0] == trump:
                self.player_turn, self.points = True, points
                self.player_points += self.points
            else:
                if self.player_turn:
                    self.player_turn, self.points = True, points
                    self.player_points += self.points
                else:
                    self.player_turn, self.points = False, points
                    self.bot_points += self.points

        # Condition to close the game.
        if sum([True for x in [self.player_card_1_pos, self.player_card_2_pos, self.player_card_3_pos] if str(x) in self.rect_img_dict.keys()]) == 0:
            self.last_hand = True

        # Add new cards to both player and bot if the deck contain still more than one card.
        if self.len_virtual_deck > 1:

            # Take img and info of player's new card, draw it and save its mapping.
            path, card_id = self.virtual_deck.pop(0)
            img = self.load_card_img(path)
            self.screen.blit(img, self.player_free_position)
            self.rect_img_dict[str(self.player_free_position)] = (img, card_id)

            # Add backsie of a card in bot free position.
            self.screen.blit(self.card_backside, self.bot_free_position)

            # Take img and info of bot's new card and save its mapping.
            path, card_id = self.virtual_deck.pop(0)
            img = self.load_card_img(path)
            self.rect_img_dict[str(self.bot_free_position)] = (img, card_id)

            # Remove the two cards from the deck.
            self.len_virtual_deck -= 2

        # If there is one reamining card in the deck and the trump.
        elif self.len_virtual_deck == 1:

            # Cover deck and trump.
            pygame.draw.rect(self.screen, self.table_color, self.deck_pos)
            pygame.draw.rect(self.screen, self.table_color, self.trump_pos)

            # Empty the deck.
            self.len_virtual_deck = 0

            # Check if player win the hand.
            if self.player_turn:

                # Get last card of the deck, draw it in the player hand and map it.
                path, card_id = self.virtual_deck.pop(0)
                img = self.load_card_img(path)
                self.screen.blit(img, self.player_free_position)
                self.rect_img_dict[str(self.player_free_position)] = (img, card_id)

                # Take the trump card, rotate it, draw it and map its new position in bot hand.

                self.screen.blit(self.card_backside, self.bot_free_position)
                img, card_id = self.rect_img_dict[str(self.trump_pos)]
                img = pygame.transform.rotate(img, 270)
                self.rect_img_dict[str(self.bot_free_position)] = (img, card_id)

            # Do the above code inverting player and bot.
            else:
                path, card_id = self.virtual_deck.pop(0)
                img = self.load_card_img(path)
                self.screen.blit(self.card_backside, self.bot_free_position)
                self.rect_img_dict[str(self.bot_free_position)] = (img, card_id)

                img, card_id = self.rect_img_dict[str(self.trump_pos)]
                img = pygame.transform.rotate(img, 270)
                self.screen.blit(img, self.player_free_position)
                self.rect_img_dict[str(self.player_free_position)] = (img, card_id)  

        # Labels updating.
        self.update_cards_left()
        self.update_turn()
        self.update_score()


    def update_cards_left(self):
        '''Update number of card remain on the deck'''

        if self.len_virtual_deck !=0:
            self.screen.blit(self.card_backside, self.deck_pos)
            x = self.deck.x + self.card_w/2
            y = self.deck.y + self.card_h/2
            pygame.draw.circle(self.screen, 'black', (x, y), 20)
            pygame.draw.circle(self.screen, 'white', (x, y), 18)
            font = pygame.font.Font(None, 20)
            text = font.render(str(self.len_virtual_deck), True, 'black')
            self.screen.blit(text, text.get_rect(center = (x-1, y)))


    def update_turn(self):
        '''Change turn label'''

        x_bot = self.bot_card_3.x + self.card_w 
        y_bot = self.bot_card_3.y + self.card_h/3
        pygame.draw.rect(self.screen, self.table_color, [x_bot, y_bot, self.card_h, self.card_w/2])
        x_player = self.player_card_3.x + self.card_w 
        y_player = self.player_card_3.y + self.card_h/3
        pygame.draw.rect(self.screen, self.table_color, [x_player, y_player, self.card_h, self.card_w/2])

        # Check if game is not over.
        if not self.over:

            if self.player_turn:
                text = 'Your turn'
                text = self.font.render(text, True, self.label_color)
                self.screen.blit(text, text.get_rect(center = (x_player + self.card_h/2, y_player + self.card_w/4)))
            else:
                text = 'Bot turn'
                text = self.font.render(text, True, self.label_color)
                self.screen.blit(text, text.get_rect(center = (x_bot+ self.card_h/2, y_bot + self.card_w/4)))


    def update_score(self):
        '''Change score label'''
        
        # Update bot's score.
        x_bot = self.bot_card_1.x - self.card_w * 3
        y_bot = self.bot_card_1.y + self.card_h/3
        pygame.draw.rect(self.screen, self.table_color, [x_bot, y_bot, self.card_h*1.5, self.card_w/2])
        text = f'Bot score: {self.bot_points}'
        text = self.font.render(text, True, self.label_color)
        self.screen.blit(text, text.get_rect(center = (x_bot + self.card_h/2, y_bot + self.card_w/4)))

        # Update player's score.
        x_player = self.player_card_1.x - self.card_w * 3
        y_player = self.player_card_1.y + self.card_h/3
        pygame.draw.rect(self.screen, self.table_color, [x_player, y_player, self.card_h*1.5, self.card_w/2])
        text = f'Your score: {self.player_points}'
        text = self.font.render(text, True, self.label_color)
        self.screen.blit(text, text.get_rect(center = (x_player + self.card_h/2, y_player + self.card_w/4)))


    def game_over(self):
        '''Final window after game over'''

        pygame.time.wait(1000)
        self.screen.fill(self.table_color)
        w, h = 300, 100
        x, y = (self.WIDTH - w) /2, (self.HEIGHT - h) /2
        self.restart_button = pygame.draw.rect(self.screen, 'white', [x, y, w, h], 0, self.card_smoothing)
        pygame.draw.rect(self.screen, 'darkgray', [x, y, w, h], 7, self.card_smoothing)
        pygame.draw.rect(self.screen, 'black', [x, y, w, h], 5, self.card_smoothing)
        button_text = self.button_font.render('RESTART', True, 'black')
        self.screen.blit(button_text, button_text.get_rect(center = (x+w/2, y+h/2)))

        points_text = self.button_font.render(f'{self.player_points} - {self.bot_points}', True, 'black')
        self.screen.blit(points_text, points_text.get_rect(center = (self.WIDTH/2, self.WIDTH/6)))

        if self.bot_points > self.player_points:
            message = 'Ops, you lose... Wanna try again?'
        elif self.bot_points < self.player_points:
            message = 'You win!! Let me retry.'
        elif self.bot_points == self.player_points:
            message = 'Play off?'
        message_text = self.button_font.render(message, True, 'black')
        self.screen.blit(message_text, message_text.get_rect(center = (self.WIDTH/2, self.WIDTH/4)))

        # Initialize again all values.
        self.player_turn = bool(random.randint(0, 1))
        self.len_virtual_deck = 1
        self.last_two_hand = False
        self.bot_points, self.player_points, self.points = 0, 0, 0
        self.rect_img_dict, self.img_card_dict = {}, {}
        self.level = 0
        self.levels_interface()


    def run(self):
        while self.running:
            if not self.active and not self.reactive:
                self.start_game()
            elif self.active and not self.reactive:
                self.clean_values()
                self.build_game()
                self.reactive = True
            
            for self.event in pygame.event.get():
                if self.event.type == pygame.QUIT:
                    self.running = False
                elif self.event.type == pygame.MOUSEBUTTONDOWN and not self.active and not self.reactive:
                    if self.start_button.collidepoint(self.event.pos):
                        if self.level == 0: self.level = 2
                        self.active = True
                        continue

                elif self.event.type == pygame.MOUSEBUTTONDOWN and not self.active and self.reactive:
                    if self.restart_button.collidepoint(self.event.pos):
                        if self.level == 0: self.level = 2
                        self.active = True
                        self.reactive = False
                        continue

                # Levels.
                if self.event.type == pygame.MOUSEBUTTONDOWN and not self.active:
                    if self.easy_level_button.collidepoint(self.event.pos):
                        self.level = 1
                        self.select_level()

                    elif self.medium_level_button.collidepoint(self.event.pos):
                        self.level = 2
                        self.select_level()

                    elif self.hard_level_button.collidepoint(self.event.pos):
                        self.level = 3
                        self.select_level()


                if self.active:

                    # Check if the game match is over and possibly start another one.
                    if self.over and self.last_hand:
                        self.game_over()
                        self.active, self.reactive = False, True
                        continue


                    # Bot still don't select card => player still don't select card.
                    if str(self.bot_played_card_pos) not in self.rect_img_dict.keys():

                        # Check if it's player turn.
                        if self.player_turn:

                            # Get player decision and then bot decision, putting both card in the middle.
                            if self.event.type == pygame.MOUSEBUTTONDOWN:
                                if self.player_card_1_pos.collidepoint(self.event.pos) and str(self.player_card_1_pos) in self.rect_img_dict.keys():
                                    self.select_card(self.player_card_1_pos)
                                    self.select_bot_card()
                                    
                                elif self.player_card_2_pos.collidepoint(self.event.pos) and str(self.player_card_2_pos) in self.rect_img_dict.keys():
                                    self.select_card(self.player_card_2_pos)
                                    self.select_bot_card()

                                elif self.player_card_3_pos.collidepoint(self.event.pos) and str(self.player_card_3_pos) in self.rect_img_dict.keys():
                                    self.select_card(self.player_card_3_pos)
                                    self.select_bot_card()

                        # If it's not player turn just get the bot's card an put it in the middle.
                        else:
                            self.select_bot_card()

                    # If bot has select its card check the decision of the player and put his card in the middle.
                    elif str(self.player_played_card_pos) not in self.rect_img_dict.keys():
                        if self.event.type == pygame.MOUSEBUTTONDOWN:
                            
                            if self.player_card_1_pos.collidepoint(self.event.pos) and str(self.player_card_1_pos) in self.rect_img_dict.keys():
                                self.select_card(self.player_card_1_pos)
                                
                            elif self.player_card_2_pos.collidepoint(self.event.pos) and str(self.player_card_2_pos) in self.rect_img_dict.keys():
                                self.select_card(self.player_card_2_pos)

                            elif self.player_card_3_pos.collidepoint(self.event.pos) and str(self.player_card_3_pos) in self.rect_img_dict.keys():
                                self.select_card(self.player_card_3_pos)

                    # If both have select the cards check who win the hand.
                    elif str(self.player_played_card_pos) in self.rect_img_dict.keys() and str(self.bot_played_card_pos) in self.rect_img_dict.keys():
                        self.check_hand_winner()


            pygame.display.flip()

        pygame.quit()