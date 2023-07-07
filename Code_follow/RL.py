import random
import numpy as np

import time

from envirement import Briscola_env
from SumTree import SumTree

from find_valid_actions import find_all_valid_actions
from transform_action import pack_action

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from keras import backend as K
import tensorflow as tf

config=K.tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=1)
session = K.tf.Session(config=config)
K.set_session(session)

ACTIVATION1 = 'relu'
ACTIVATION2 = 'relu'

LEARNING_RATE = 0.00025
BATCH_SIZE = 256

GAMMA = 0.99
LAMBDA = 0.0001      # speed of decay

MODEL_NAME = '1x512net-act1-%s-act2-%s-BS-%d-LR-%f-GAMMA-%f-LAMBDA-%f--%d' % (ACTIVATION1, ACTIVATION2, BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, int(time.time()))


def huber_loss(y_true, y_pred):

    HUBER_LOSS_DELTA = 1.0

    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


class Brain:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()

        # print self.model.summary()

        # self.model.load_weights('Rocketman-network.h5')
        # self.model_.load_weights('Rocketman-t_network.h5')


    def _createModel(self):

        model = Sequential()

        model.add(Dense(units=256, activation=ACTIVATION1, input_dim=self.stateCnt))
        model.add(Dense(512))
        model.add(Dense(units=self.actionCnt, activation=ACTIVATION2))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model


    def train(self, x, y, epoch=1, verbose=0):

        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)


    def evaluate(self, x, y):

        return self.model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0)


    def predict(self, s, target=False):

        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

        return self.model.predict(s)


    def predictOne(self, s, target=False):

        return self.predict(s.reshape(1, self.stateCnt), target).flatten()


    def predict_next_action(self, state, player_board, top_discards, target=False):

        valid_actions = find_all_valid_actions(player_board, top_discards)

        next_Qs = self.predictOne(state, target)
        next_Qs = next_Qs[valid_actions]

        idx = np.argmax(next_Qs)

        return valid_actions[idx]


    def updateTargetModel(self):

        self.model_.set_weights(self.model.get_weights())


class Memory:   # stored as ( s, a, r, s_ )

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

        self.e = 0.01
        self.a = 0.6


    def get_priority(self, error):

        return (error + self.e) ** self.a


    def add(self, error, sample):

        p = self.get_priority(error)
        self.tree.add(p, sample)


    def sample(self, n):

        batch = []
        segment = self.tree.total() / n

        for i in range(n):

            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch


    def update(self, idx, error):
        p = self.get_priority(error)
        self.tree.update(idx, p)


#----------
MEMORY_CAPACITY = 300000

MAX_EPSILON = 1
MIN_EPSILON = 0.01

UPDATE_TARGET_FREQUENCY = 2000
#----------

class Agent:

    def __init__(self, stateCnt, actionCnt):

        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.rewards_log = np.zeros(50000, dtype=np.int16)
        self.scores_log = np.zeros(50000, dtype=np.int16)
        self.loss_log = np.zeros(50000, dtype=np.float32)

        self.episode = 0


    def act(self, s, player_board, discards):

        hand = player_board.hand

        if random.random() < self.epsilon:

            rand_card = random.choice(hand)

            play = random.choice([0, 1])

            deck_draw = random.choice([0, 1])

            if (deck_draw == 1):
                
                draw_action = 0
            else:

                rand_suit = random.choice([1, 2, 3, 4])

                if (discards[rand_suit-1] == -1):

                    draw_action = 0

                else:

                    draw_action = rand_suit

            return pack_action(rand_card, play, draw_action)

        else:

            return self.brain.predict_next_action(s, player_board, discards)


    def observe(self, sample):  # in (s, a, r, s_) format

        (x, y, errors) = self.getTargets([(0, sample)])
        self.memory.add(errors[0], sample)        

        if (self.steps % UPDATE_TARGET_FREQUENCY == 0):
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)


    def replay(self):    

        batch = self.memory.sample(BATCH_SIZE)
        (x, y, errors) = self.getTargets(batch)

        for i in range(BATCH_SIZE):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

        return self.brain.evaluate(x, y)


    def getTargets(self, batch):

        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        p_target = self.brain.predict(states_, True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))
        errors = np.zeros(batchLen)
        
        for i in range(batchLen):

            obs = batch[i][1]

            s = obs[0]
            a = obs[1]
            r = obs[2]
            s_ = obs[3]

            target = p[i]
            oldVal = target[a]

            if s_ is None:
                target[a] = r
            else:
                target[a] = r + GAMMA * p_target[i][np.argmax(p_[i])]

            x[i] = s
            y[i] = target
            errors[i] = abs(oldVal - target[a])

        return (x, y, errors)


class RandomAgent:
    
    def __init__(self, load_samples):

        self.memory = Memory(MEMORY_CAPACITY)
        self.exp = 0

        if load_samples:
            self.memory.tree.load()
            self.exp = MEMORY_CAPACITY


    def act(self, s, player_board, discards):

        hand = player_board.hand

        rand_card = random.choice(hand)

        play = random.choice([0, 1])

        deck_draw = random.choice([0, 1])

        if (deck_draw == 1):
                
            draw_action = 0

        else:

            rand_suit = random.choice([1, 2, 3, 4])

            if (discards[rand_suit-1] == -1):

                draw_action = 0

            else:

                draw_action = rand_suit

        return pack_action(rand_card, play, draw_action)


    def observe(self, sample):  # in (s, a, r, s_) format

        error = abs(sample[2])

        self.memory.add(error, sample)
        self.exp += 1


    def replay(self):

        pass


    def save(self):

        self.memory.tree.save()


class Environment:

    def __init__(self):

        self.env = RocketmanEnv()


    def run(self, agent, logRewards=False):

        self.env.reset()
        done = False

        n_steps = 0

        cum_loss = 0
        R = 0 

        while not done:            

            (r, done, loss) = self.run_agent(agent)

            R += r

            cum_loss += loss
            n_steps += 1

        if logRewards:

            agent.rewards_log[agent.episode] = R
            agent.loss_log[agent.episode] = cum_loss / n_steps

            cum_score = self.env.gameboard.report_score(1) + self.env.gameboard.report_score(2)
            agent.scores_log[agent.episode] = cum_score

            if ((agent.episode % 50) == 0):
                print('Episode: ', agent.episode)

            agent.episode += 1


    def run_agent(self, agent):

        rewards = [0, 0]
        done_list = [False, False]
        losses = [0, 0]

        for i in range(2):

            player = i + 1

            if (player == 1):
                state = self.env.p1_obs
                player_board = self.env.gameboard.p1_board

            elif (player == 2):
                state = self.env.p2_obs
                player_board = self.env.gameboard.p2_board

                if (done_list[0]):
                    break

            discards = self.env.top_discard

            s = state.copy()

            a = agent.act(state, player_board, discards)

            (r, done) = self.env.step(a, player)

            if done: # terminal state
                s_ = None
            else:
                s_ = state.copy()

            agent.observe((s, a, r, s_))
            loss = agent.replay()            

            rewards[i] = r
            done_list[i] = done
            losses[i] = loss

        R = rewards[0] + rewards[1]
        Done = done_list[0] or done_list[1]
        loss = (losses[0] + losses[1]) / 2.

        return (R, Done, loss)


#-------------------- MAIN ----------------------------

if __name__ == "__main__":
    env = Environment()

    stateCnt  = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n

    agent = Agent(stateCnt, actionCnt)

    load_random_samples = True
    n_rand_games = 0

    randomAgent = RandomAgent(load_random_samples)

    try:

        while randomAgent.exp < MEMORY_CAPACITY:

            if ((n_rand_games % 25) == 0):

                print(randomAgent.exp, "/", MEMORY_CAPACITY, " random samples")

            env.run(randomAgent)

            n_rand_games += 1

        if not load_random_samples:

            randomAgent.save()

            print('Random samples saved.')

        agent.memory = randomAgent.memory

        randomAgent = None

        print('Beginning learning')
        while True:
            env.run(agent, logRewards=True)
    
    finally:

        agent.brain.model.save('logs/Rocketman-network--' + MODEL_NAME + '.h5')
        agent.brain.model_.save('logs/Rocketman-t_network--' + MODEL_NAME + '.h5')

        np.save('logs/Rocketman-rewards--' + MODEL_NAME, agent.rewards_log)
        np.save('logs/Rocketman-scores--' + MODEL_NAME, agent.scores_log)
        np.save('logs/Rocketman-loss--' + MODEL_NAME, agent.loss_log)