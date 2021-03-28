import os
import pickle
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.5

from sklearn.base import clone

import settings as s
import events as e
from .callbacks import (transform, state_to_features, has_object,
                        is_lethal, fname, FILENAME)

# Transition tuple. (s, a, s', r)
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

# ------------------------ HYPER-PARAMETERS -----------------------------------
# General hyper-parameters:
TRANSITION_HISTORY_SIZE = 1000  # Keep only ... last transitions.
BATCH_SIZE              = 500   # Size of batch in TD-learning.
TRAIN_FREQ              = 1     # Train model every ... game.

# N-step TD Q-learning:
GAMMA   = 0.8  # Discount factor.
N_STEPS = 3    # Number of steps to consider real, observed rewards.

# Prioritized experience replay:
PRIO_EXP_REPLAY   = True    # Toggle on/off.
PRIO_EXP_FRACTION = 0.25    # Fraction of BATCH_SIZE to keep.

# Dimensionality reduction from learning experience.
DR_FREQ           = 1000    # Play ... games before we fit DR.
DR_EPOCHS         = 30      # Nr. of epochs in mini-batch learning.
DR_MINIBATCH_SIZE = 10000   # Nr. of states in each mini-batch.
DR_HISTORY_SIZE   = 50000   # Keep the ... last states for DR learning.

# Epsilon-Greedy: (0 < epsilon < 1)
EXPLORATION_INIT  = 1.0
EXPLORATION_MIN   = 0.005
EXPLORATION_DECAY = 0.99995

# Softmax: (0 < tau < infty)
TAU_INIT  = 15
TAU_MIN   = 0.5
TAU_DECAY = 0.999

# Auxilary:
PLOT_FREQ = 25
# -----------------------------------------------------------------------------

# File name of historical training record used for plotting.
FNAME_DATA = f"{FILENAME}_data.pt"

# Custom events:
CLOSER_TO_ESCAPE = "CLOSER_TO_ESCAPE"
FURTHER_FROM_ESCAPE = "FURTHER_FROM_ESCAPE"
BOMBED_GOAL = "BOMBED_GOAL"
MISSED_GOAL = "MISSED_GOAL"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_OTHERS = "CLOSER_TO_OTHERS"
FURTHER_FROM_OTHERS = "FURTHER_FROM_OTHERS"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
FURTHER_FROM_CRATE = "FURTHER_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Ques to store the transition tuples and coordinate history of agent.
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.n_step_transitions = deque([], N_STEPS)
    
    # Storage of states for feature extration function learning.
    if not self.dr_override:
        self.state_history = deque(maxlen=DR_HISTORY_SIZE)

    # Set inital epsilon/tau.
    if self.act_strategy == 'eps-greedy':
        self.epsilon = EXPLORATION_INIT
    elif self.act_strategy == 'softmax':
        self.tau = TAU_INIT
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")

    # For evaluation of the training progress:
    # Check if historic data file exists, load it in or start from scratch.
    if os.path.isfile(FNAME_DATA):
        # Load historical training data.
        with open(FNAME_DATA, "rb") as file:
            self.historic_data = pickle.load(file)
        self.game_nr = max(self.historic_data['games']) + 1
    else:
        # Start a new historic record.
        self.historic_data = {
            'score'  : [],       # subplot 1
            'coins'  : [],       # subplot 2
            'crates' : [],       # subplot 3
            'enemies': [],       # subplot 4
            'exploration' : [],  # subplot 5
            'games'  : []        # subplot 1,2,3,4,5 x-axis
        }
        self.game_nr = 1

    # Initialization
    self.score_in_round    = 0
    self.collected_coins   = 0
    self.destroyed_crates  = 0
    self.killed_enemies    = 0
    self.bomb_loop_penalty = 0
    self.perform_export    = False

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

    # ---------- (1) Add own events to hand out rewards: ----------
    # If the old state is not before the beginning of the game:
    if old_game_state:
        # Extract feature vector:
        state_old = state_to_features(old_game_state)

        # Extract the lethal indicator from the old state.
        islethal_old = state_old[0,0] == 1
        if islethal_old:
            # ---- WHEN IN LETHAL ----
            # When in lethal danger, we only care about escaping. Following the
            # escape direction is rewarded, anything else is penalized.
            escape_dir_old = state_old[0,2:4]
            if check_direction(escape_dir_old, self_action):
                events.append(CLOSER_TO_ESCAPE)
            else:
                events.append(FURTHER_FROM_ESCAPE)

        else:
            # ---- WHEN IN NON-LETHAL ----
            # When not in lethal danger, we are less stressed to make the right
            # decision. Our order of prioritization is: others > coins > crates.

            # Extracting information from the old game state.
            target_acq_old = state_old[0,1]            
            others_dir_old = state_old[0,4:6]
            coins_dir_old  = state_old[0,6:8]
            crates_dir_old = state_old[0,8:10]

            # If we chose to bomb in the previous state:
            if self_action == 'BOMB':
                # Reward if we successfully bombed the target, else penalize.
                if target_acq_old == 1:
                    events.append(BOMBED_GOAL)
                else:
                    events.append(MISSED_GOAL)

            # If we chose to wait in the previous state:
            elif self_action == 'WAIT':
                # Reward the agent if the waiting was neccesary, but penalize
                # if a direction for others, coins or crates was suggested.
                if (all(others_dir_old == (0,0)) and
                    all(coins_dir_old  == (0,0)) and
                    all(crates_dir_old == (0,0)) and
                    target_acq_old == 0):
                    events.append(WAITED_NECESSARILY)
                else:
                    events.append(WAITED_UNNECESSARILY)

            # If we chose to move in the previous state:
            else:
                # Penalize if we were standing at the bomb goal, but moved.
                if target_acq_old == 1:
                    events.append(MISSED_GOAL)

                # Movement priority: others > coins > crates.
            
                # Reward/penalize for moving towards/away from the offensive target.
                if not all(others_dir_old == (0,0)):
                    if check_direction(others_dir_old, self_action):
                        events.append(CLOSER_TO_OTHERS)
                    else:
                        events.append(FURTHER_FROM_OTHERS)

                # Reward/penalize for moving towards/away from the closest coin.
                elif not all(coins_dir_old == (0,0)):
                    if check_direction(coins_dir_old, self_action):
                        events.append(CLOSER_TO_COIN)
                    else:
                        events.append(FURTHER_FROM_COIN)

                # Reward/penalize for moving towards/away from the crate target.
                elif not all(crates_dir_old == (0,0)):
                    if check_direction(crates_dir_old, self_action):
                        events.append(CLOSER_TO_CRATE)
                    else:
                        events.append(FURTHER_FROM_CRATE)
            

    # Reward for surviving (effectively a passive reward).
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)         
    
    # ---------- (2) Compute n-step reward and store tuple in Transitions: ----------
    # Transition tuple structure: (s, a, s', r)
    self.n_step_transitions.append((transform(self, old_game_state), self_action, transform(self, new_game_state), reward_from_events(self, events)))
    
    # When buffer is filled:
    if len(self.n_step_transitions) == N_STEPS:
        # The given starting state with corresponding action for which we want
        # to estimate the Q-value function.
        n_step_old_state = self.n_step_transitions[0][0]
        n_step_action = self.n_step_transitions[0][1]

        # Rewards observed in N_STEPS after the starting state and action.
        reward_arr = np.array([self.n_step_transitions[i][-1] for i in range(N_STEPS)])
        # Sum with the discount factor to get the accumulated rewards over N_STEP transitions.
        n_step_reward = ((GAMMA)**np.arange(N_STEPS)).dot(reward_arr)

        # The new state after N_STEPS transitions following the policy.
        n_step_new_state = self.n_step_transitions[-1][2]

        # (s, a, s', r) where s' is the state after N_STEPS, and r is the accumulation of rewards until s'.
        self.transitions.append(Transition(n_step_old_state, n_step_action, n_step_new_state, n_step_reward))
    
    # ---------- (3) Store the game state for feature extration function learning: ----------
    # Store the game state for learning of feature extration function.
    if old_game_state and not self.dr_override:
        self.state_history.append(state_to_features(old_game_state)[0])

    # ---------- (4) For evaluation purposes: ----------
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_enemies += events.count('KILLED_OPPONENT')
    self.score_in_round += reward_from_events(self, events)


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    
    # ---------- (1) Compute last n-step reward and store tuple in Transitions ----------
    # Transition tuple structure: (s, a, s', r)
    self.n_step_transitions.append((transform(self, last_game_state), last_action, None, reward_from_events(self, events)))
    
    # When buffer is filled:
    if len(self.n_step_transitions) == N_STEPS:
        # The given starting state with corresponding action for which we want
        # to estimate the Q-value function.
        n_step_old_state = self.n_step_transitions[0][0]
        n_step_action    = self.n_step_transitions[0][1]

        # Rewards observed in N_STEPS after the starting state and action.
        reward_arr = np.array([self.n_step_transitions[i][-1] for i in range(N_STEPS)])
        # Sum with the discount factor to get the accumulated rewards over N_STEPS transitions.
        n_step_reward = ((GAMMA)**np.arange(N_STEPS)).dot(reward_arr)

        # (s, a, s', r) where s' is the state after N_STEPS, and r is the accumulation of rewards until s'.
        self.transitions.append(Transition(n_step_old_state, n_step_action, None, n_step_reward))
    
    # Store the game state for learning of feature extration function.
    if last_game_state and not self.dr_override:
        self.state_history.append(state_to_features(last_game_state)[0])

    # ---------- (2) Decrease the exploration rate: ----------
    if len(self.transitions) > BATCH_SIZE:
        if self.act_strategy == 'eps-greedy':    
            if self.epsilon > EXPLORATION_MIN:
                self.epsilon *= EXPLORATION_DECAY
        elif self.act_strategy == 'softmax':
            if self.tau > TAU_MIN:
                self.tau *= TAU_DECAY
        else:
            raise ValueError(f"Unknown act_strategy {self.act_strategy}")

    # ---------- (3) N-step TD Q-learning with batch: ----------
    if len(self.transitions) > BATCH_SIZE and self.game_nr % TRAIN_FREQ == 0:
        # Create a random batch from the transition history.
        batch = random.sample(self.transitions, BATCH_SIZE)
            
        # Initialization.
        X = [[] for i in range(len(self.actions))] # Feature matrix for each action
        y = [[] for i in range(len(self.actions))] # Target vector for each action
        residuals = [[] for i in range(len(self.actions))] # Corresponding residuals.
        
        # For every transition tuple in the batch:
        for state, action, next_state, reward in batch:
            # Current state cannot be the state before game start.
            if state is not None:
                # Index of action taken in 'state'.
                action_idx = self.actions.index(action)

                # Q-value for the given state and action.
                if self.model_is_fitted and next_state is not None:
                    # Non-terminal next state and pre-existing model.
                    maximal_response = np.max(self.model.predict(next_state))
                    q_update = (reward + GAMMA**N_STEPS * maximal_response)
                else:
                    # Either next state is terminal or a model is not yet fitted.
                    q_update = reward # Equivalent to a Q-value of zero for the next state.

                # Append feature data and targets for the regression,
                # corresponding to the current action.
                X[action_idx].append(state[0])
                y[action_idx].append(q_update)

                # Prioritized experience replay.
                if PRIO_EXP_REPLAY and self.model_is_fitted:
                    # Calculate the residuals for the training instance.
                    X_tmp = X[action_idx][-1].reshape(1, -1)
                    target = y[action_idx][-1]
                    q_estimate = self.model.predict(X_tmp, action_idx=action_idx)[0]
                    res = (target - q_estimate)**2
                    residuals[action_idx].append(res)
        
        # Prioritized experience replay.
        if PRIO_EXP_REPLAY and self.model_is_fitted:
            # Initialization
            X_new = [[] for i in range(len(self.actions))]
            y_new = [[] for i in range(len(self.actions))]
            
            # For the training set of every action:
            for i in range(len(self.actions)):    
                # Keep the specifed fraction of samples with the largest squared residuals.
                prio_exp_size = int(len(residuals[i]) * PRIO_EXP_FRACTION)
                idx = np.argpartition(residuals[i], -prio_exp_size)[-prio_exp_size:]
                X_new[i] = [X[i][j] for j in list(idx)]
                y_new[i] = [y[i][j] for j in list(idx)]
            
            # Update the training set.
            X = X_new
            y = y_new

        # Regression fit.
        self.model.partial_fit(X, y)
        self.model_is_fitted = True

        # Raise flag for export of the learned model.
        self.perform_export = True

        # Logging
        self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # ---------- (4) Improve dimensionality reduction: ----------
    # Learn a new (hopefully improved) model for dimensionality reduction.
    if ((not self.dr_override) and
        (self.game_nr % DR_FREQ == 0) and
        (len(self.state_history) > DR_MINIBATCH_SIZE)):
        
        # Minibatch learning on the collected samples.
        for _ in range(DR_EPOCHS):
            batch = random.sample(self.state_history, DR_MINIBATCH_SIZE)
            self.dr_model.partial_fit(np.vstack(batch))
        self.dr_model_is_fitted = True

        # Since the feature extraction function is now changed, we need to start
        # the learning process of the Q-value function over from scratch.
        # Create a new, but unfitted, Q-value model of the same type as before.
        self.model = clone(self.model)
        self.model_is_fitted = False

        # Empty lists of transitions, bomb history and game states.
        self.transitions.clear()
        #self.state_history.clear()

        # Reset epsilon/tau to their inital values
        if self.act_strategy == 'eps-greedy':
            self.epsilon = EXPLORATION_INIT
        elif self.act_strategy == 'softmax':
            self.tau = TAU_INIT

        # Raise flag for export of full model.
        self.perform_export = True

    # ---------- (5) Clear N-step transition history: ----------
    # Do not compute the aggregated rewards beyond one game.
    self.n_step_transitions.clear()

    # ---------- (6) Model export: ----------
    # Check if a full model export has been requested.
    if self.perform_export:
        export = self.model, self.dr_model
        with open(fname, "wb") as file:
            pickle.dump(export, file)
        self.perform_export = False # Reset export flag

    # ---------- (7) Performance evaluation: ----------
    # Get the numbers from the last round.
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_enemies += events.count('KILLED_OPPONENT')
    self.score_in_round += reward_from_events(self, events)

    # Total score in this game.
    score = np.sum(self.score_in_round)
   
    # Append results to each specific list.
    self.historic_data['score'].append(score)
    self.historic_data['coins'].append(self.collected_coins)
    self.historic_data['crates'].append(self.destroyed_crates)
    self.historic_data['enemies'].append(self.killed_enemies)
    self.historic_data['games'].append(self.game_nr)   
    if self.act_strategy == 'eps-greedy':
        self.historic_data['exploration'].append(self.epsilon)
    elif self.act_strategy == 'softmax':
        self.historic_data['exploration'].append(self.tau)

    # Store the historic record.
    with open(FNAME_DATA, "wb") as file:
        pickle.dump(self.historic_data, file)
    
    # Reset game score, coins collected and one up the game count.
    self.score_in_round   = 0
    self.collected_coins  = 0
    self.destroyed_crates = 0
    self.killed_enemies   = 0
    self.game_nr += 1
    
    # Plot training progress every n:th game.
    if self.game_nr % PLOT_FREQ == 0:
        # Incorporate the full training history.
        games_list = self.historic_data['games']
        score_list = self.historic_data['score']
        coins_list = self.historic_data['coins']
        crate_list = self.historic_data['crates']
        other_list = self.historic_data['enemies']
        explr_list = self.historic_data['exploration']

        # Plotting
        fig, ax = plt.subplots(5, figsize=(7.2, 5.4), sharex=True)

        # Total score per game.
        ax[0].plot(games_list, score_list)
        ax[0].set_title('Total score per game')
        ax[0].set_ylabel('Score')

        # Collected coins per game.
        ax[1].plot(games_list, coins_list)
        ax[1].set_title('Collected coins per game')
        ax[1].set_ylabel('Coins')

        # Destroyed crates per game.
        ax[2].plot(games_list, crate_list)
        ax[2].set_title('Destroyed crates per game')
        ax[2].set_ylabel('Crates')

        # Eliminiated opponents per game
        ax[3].plot(games_list, other_list)
        ax[3].set_title('Eliminated opponents per game')
        ax[3].set_ylabel('Kills')

        # Exploration rate (epsilon/tau) per game.
        ax[4].plot(games_list, explr_list)
        if self.act_strategy == 'eps-greedy':        
            ax[4].set_title('$\epsilon$-greedy: Exploration rate $\epsilon$')
            ax[4].set_ylabel('$\epsilon$')
        elif self.act_strategy == 'softmax':
            ax[4].set_title('Softmax: Exploration rate $\\tau$')
            ax[4].set_ylabel('$\\tau$')
        ax[4].set_xlabel('Game #')

        # Export the figure.
        fig.tight_layout()
        plt.savefig(f'TrainEval_{FILENAME}.pdf')
        plt.close('all')

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    # escape > kill > coin > crate
    
    # Base rewards:
    kill  = s.REWARD_KILL
    coin  = s.REWARD_COIN
    crate = 0.1 * coin
    
    escape_movement    = 0.1  * kill
    bombing            = 0.1  * kill
    waiting            = 0.1  * kill
    offensive_movement = 0.05 * kill
    coin_movement      = 0.1  * coin
    crate_movement     = 0.05 * coin

    passive = 0

    # Game reward dictionary:
    game_rewards = {
        # ---- CUSTOM EVENTS ----
        # escape movement
        CLOSER_TO_ESCAPE    : escape_movement,
        FURTHER_FROM_ESCAPE : -escape_movement,

        # bombing
        BOMBED_GOAL         : bombing,
        MISSED_GOAL         : -4*escape_movement, # Needed to prevent self-bomb-laying loops.

        # waiting
        WAITED_NECESSARILY  : waiting,
        WAITED_UNNECESSARILY: -waiting,

        # offensive movement
        CLOSER_TO_OTHERS    : offensive_movement,
        FURTHER_FROM_OTHERS : -offensive_movement,

        # coin movement
        CLOSER_TO_COIN      : coin_movement,
        FURTHER_FROM_COIN   : -coin_movement,
        
        # crate movement
        CLOSER_TO_CRATE     : crate_movement,
        FURTHER_FROM_CRATE  : -crate_movement,
        
        # passive
        SURVIVED_STEP       : passive,

        # ---- DEFAULT EVENTS ----
        # movement
        e.MOVED_LEFT         :  0,
        e.MOVED_RIGHT        :  0,
        e.MOVED_UP           :  0,
        e.MOVED_DOWN         :  0,
        e.WAITED             :  0,
        e.INVALID_ACTION     : -1,
        
        # bombing
        e.BOMB_DROPPED       : 0,
        e.BOMB_EXPLODED      : 0,

        # crates, coins
        e.CRATE_DESTROYED    : crate,
        e.COIN_FOUND         : 0,
        e.COIN_COLLECTED     : coin,

        # kills
        e.KILLED_OPPONENT    : kill,
        e.KILLED_SELF        : -kill,
        e.GOT_KILLED         : -kill,
        e.OPPONENT_ELIMINATED: 0,

        # passive
        e.SURVIVED_ROUND     : passive,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def check_direction(direction: np.array, self_action: str) -> bool:
    """
    Check if a taken action was the appropriate one given a certain
    direction vector.
    
    Parameters:
    -----------
    direction: np.array shape=(2,)
        Binary direction vector indicating the optimal direction to take.
    
    Returns:
    --------
    is_correct_action: bool
        True if the action taken was the correct one given the direction vector.
    """
    return ((all(direction == ( 0, 1)) and self_action == 'DOWN' ) or
            (all(direction == ( 1, 0)) and self_action == 'RIGHT') or
            (all(direction == ( 0,-1)) and self_action == 'UP'   ) or
            (all(direction == (-1, 0)) and self_action == 'LEFT' ))