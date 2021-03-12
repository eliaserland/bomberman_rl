import pickle
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import settings as s
import events as e
from .callbacks import state_to_features
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
BATCH_SIZE = 3000
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.2
EXPLORATION_DECAY = 0.9995
LEARNING_RATE = 1  # test 0.5
GAMMA = 0.90
N_STEP_TD = 20


# Events
SURVIVED_STEP = "SURVIVED_STEP"
DIED_DIRECT_NEXT_TO_BOMB = "DIED_DIRECT_NEXT_TO_BOMB"
ALREADY_KNOW_FIELD = "ALREADY_KNOW_FIELD"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
BACK_AND_FORTH = "BACK_AND_FORTH"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=10) # long term memory of complete step
    self.X = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.targets = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.coordinate_history = deque([], 10)      # short term memory of agent position
    self.reward_history = deque([], N_STEP_TD)   # for n step TD 
    
    self.epsilon = EXPLORATION_MAX
    self.n_steps = N_STEP_TD
    self.is_init = True
    
    # For Training evaluation purposes:
    self.score_in_round = 0
    self.number_game = 0
    self.collected_coins_in_game  = 0
    
    self.scores = []
    self.games = []
    self.exploration_rate = []
    self.collected_coins = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    ################# (1) Add own events to hand out rewards #################
                
    if old_game_state:
        _, score, bombs_left, (x, y) = old_game_state['self']
        closest_coin_info_old = closets_coin_distance(old_game_state)

        # penalty on loops:
        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 1:
            events.append(ALREADY_KNOW_FIELD)
            if self.coordinate_history.count((x, y)) > 2:
                events.append(BACK_AND_FORTH)

        self.coordinate_history.append((x, y))

        # penalty on going away from coin vs reward for going closer: 
        if new_game_state:
            closest_coin_info_new = closets_coin_distance(new_game_state)

            if (closest_coin_info_old - closest_coin_info_new) < 0:
                events.append(CLOSER_TO_COIN)
            else:
                events.append(AWAY_FROM_COIN)

        if 'GOT_KILLED' in events:
            # closer to bomb gives higher penalty:
            bombs = old_game_state['bombs']
            bomb_xys = [xy for (xy, t) in bombs]

            step_distance_bombs = []
            for bomb in bombs:
                x_rel_bomb = bomb[0][0] - x
                y_rel_bomb = bomb[0][1] - y
                step_distance_bombs.append(np.abs(x_rel_bomb) +  np.abs(y_rel_bomb))
            step_distance_closest_bomb = sorted(step_distance_bombs)[0]

            if step_distance_closest_bomb < 2:
                events.append(DIED_DIRECT_NEXT_TO_BOMB)

    # reward for surviving:          
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)
        
    if 'COIN_COLLECTED' in events:
        self.collected_coins_in_game += 1
    
    ################## (2) Store Transision: #################
    
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    ################## (2) Do q-learning with n step TD: #################
    
    q_learning_with_n_step_TD(self, n = self.n_steps )
    
    ################# (3) For evaluation purposes: #################
    self.score_in_round += reward_from_events(self, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    ################# (1) Decrease the exploration rate: #################
    if self.epsilon > EXPLORATION_MIN:
        self.epsilon *= EXPLORATION_DECAY
        
    ################## (2) Do q-learning with n step TD: #################
    
    # q_learning_with_n_step_TD(self, n = self.n_steps )
    if len(self.X) > BATCH_SIZE:
        
        X_batch, y_batch = zip(*random.sample(list(zip(self.X, self.targets)), BATCH_SIZE))

        self.model.partial_fit(X_batch, y_batch)
        self.is_init = False
    
    ################# (3) Store learned model: #################

    with open("my-q-learning_Mulit_SGD_agentv14_n-step_TD.pt", "wb") as file:
        pickle.dump(self.model, file)
    
    ################# (4) For evaluation purposes: #################
    score = np.sum(self.score_in_round)
    game = self.number_game
    
    self.scores.append(score)
    self.games.append(game)
    self.exploration_rate.append(self.epsilon)
    self.collected_coins.append(self.collected_coins_in_game)
    
    if game%100 == 0:
        print("game number:", game, "  score:", score, "  memory length:",
                 len(self.transitions), "  epsilon:", self.epsilon)
    
    self.score_in_round = 0
    self.number_game += 1
    self.collected_coins_in_game = 0
    
    if game%100 == 0:
        
        f = plt.figure(figsize=(10,10))
             
        ax1 = plt.subplot(311)
        ax1.title.set_text('Total Score')
        plt.plot(self.games, self.scores)
        plt.ylabel("total score")
        plt.setp(ax1.get_xticklabels(), visible=False)

        # share x only
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.title.set_text('Number of collected Coins')
        plt.plot(self.games, self.collected_coins)
        plt.ylabel("Number Coins")
        # make these tick labels invisible
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        ax3 = plt.subplot(313, sharex=ax1)
        ax3.title.set_text('Exploration rate $\epsilon$')
        plt.ylabel('Exploration rate $\epsilon$')
        plt.xlabel("game")
        plt.plot(self.games, self.exploration_rate)
        plt.savefig('TrainingEvaluation_Mulit_SGD_agentv14_n-step_TD.png') 
        
    

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    survive_step = 0.15
    game_rewards = {
        # my Events:
        SURVIVED_STEP:  survive_step,
        DIED_DIRECT_NEXT_TO_BOMB: -2*survive_step,
        ALREADY_KNOW_FIELD: -0.1,
        CLOSER_TO_COIN: 0.2,
        AWAY_FROM_COIN: -0.25,
        BACK_AND_FORTH: -0.5,
        
        # AWAY_FROM_COIN + ALREADY_KNOW_FIELD + survive_step = - (CLOSER_TO_COIN + survive_step) + survive_step
        # 2*AWAY_FROM_COIN + ALREADY_KNOW_FIELD + survive_step = - (CLOSER_TO_COIN + survive_step) + survive_step
        
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,  # could punish for waiting
        e.INVALID_ACTION: -survive_step,
        
        e.BOMB_DROPPED: -0.1,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 20,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -6* survive_step,   # maybe include later that distance to bomb is included in penatly 

        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: survive_step,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def closets_coin_distance(game_state: dict) -> int:
    '''
    return the relative total distance from the 
    agent to the closest Coin to check where Agent got closer to Coin.
    '''
    
    _, score, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    coins_dis = []
    for coin in coins:
        total_step_distance = abs(coin[0]-x) + abs(coin[1]-y)
        coin_dis = (total_step_distance)
        coins_dis.append(coin_dis)
    closest_coin_dis = sorted(coins_dis)[0]
    
    return closest_coin_dis 


def q_learning_with_n_step_TD(self, n = 20):
    ''' Do the q learning with a n-step TD,
    train the model with a random batch of n-step TD rewards and corresponding states
    '''
     ################## Do n-step TD q-learning with batch: #################
    
    state, action, state_next, reward = self.transitions[-1]

    if state is not None:
        if state_next is not None:

            # adapt to only o the below if none is not in reward_history, since otherwiese games do overlap....

            self.reward_history.append((reward , state, action, state_next))

            #print("There is no None", list(self.reward_history)[0])

            if self.is_init:
                y_per_state = np.zeros(self.action_size).reshape(1, -1)
                y_per_action =  reward
                

            else:
                #print(len(list(self.reward_history)) , np.array(self.reward_history)[:,0])

                y_per_state = self.model.predict(state.reshape(1, -1))

                maximal_response = np.max(self.model.predict(state_next.reshape(1, -1))) 
                n_step_reward = np.sum( ((GAMMA)**np.arange(len(list(self.reward_history)))).dot(np.array(self.reward_history)[:,0]) )
                y_per_action = n_step_reward + GAMMA**n *  maximal_response

            #print(action)
            y_per_state[0][self.actions.index(action)] = y_per_action

            self.X.append(state)
            self.targets.append(y_per_state[0])

       