import os
import pickle
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt

from sklearn.base import clone

import settings as s
import events as e
from .callbacks import transform, state_to_features, state_to_vect, has_object, is_lethal, fname, FILENAME

# Transition tuple. (s, a, r, s')
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward', 'n_step_next_state'))

# ------------------------ HYPER-PARAMETERS -----------------------------------
# General hyper-parameters:
TRANSITION_HISTORY_SIZE = 10000 # Keep only ... last transitions.
BATCH_SIZE              = 5000  # Size of batch in TD-learning.
TRAIN_FREQ              = 10    # Train model every ... game.

# N-step TD Q-learning:
GAMMA   = 0.95  # Discount factor.
N_STEPS = 10    # Number of steps to consider real, observed rewards.

# Prioritized experience replay:
PRIO_EXP_REPLAY = True                # Toggle on/off.
PRIO_EXP_SIZE   = int(0.5*BATCH_SIZE) # Size of the chosen subset of TS.

# Dimensionality reduction from learning experience.
DR_FREQ           = 1000    # Play ... games before we fit DR.
DR_EPOCHS         = 30      # Nr. of epochs in mini-batch learning.
DR_MINIBATCH_SIZE = 10000   # Nr. of states in each mini-batch.
DR_HISTORY_SIZE   = 50000   # Keep the ... last states for DR learning.

# Epsilon-Greedy: (0 <= epsilon <= 1)
EXPLORATION_INIT  = 1.0
EXPLORATION_MIN   = 0.1
EXPLORATION_DECAY = 0.999

# Softmax: (0 <= tau < infty)
TAU_INIT  = 15
TAU_MIN   = 0.5
TAU_DECAY = 0.999

# Auxilary:
PLOT_FREQ = 25
# -----------------------------------------------------------------------------

# File name of historical training record used for plotting.
FNAME_DATA = f"{FILENAME}_data.pt"

# Custom events:
SURVIVED_STEP = "SURVIVED_STEP"
ALREADY_BOMBED_FIELD = "ALREADY_BOMBED_FIELD"
ESCAPED_LETHAL = "ESCAPED_LETHAL"
ENTERED_LETHAL = "ENTERED_LETHAL"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
FURTHER_FROM_CRATE = "FURTHER_FROM_CRATE"
CERTAIN_SUICIDE = "CERTAIN_SUICIDE"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Ques to store the transition tuples and coordinate history of agent.
    self.transitions        = deque(maxlen=TRANSITION_HISTORY_SIZE) # long term memory of complete step
    self.bomb_history = deque([], 5)                    # short term memory of bomb placements by agent.
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
            'score' : [],       # subplot 1
            'coins' : [],       # subplot 2
            'crates': [],       # subplot 3
            'exploration' : [], # subplot 4
            'games' : []        # subplot 1,2,3,4 x-axis
        }
        self.game_nr = 1

    # Initialization
    self.score_in_round    = 0
    self.collected_coins   = 0
    self.destroyed_crates  = 0
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # ---------- (1) Add own events to hand out rewards: ----------
    if old_game_state:
        # Extract feature vector:
        state_old = state_to_features(old_game_state)
        
        # Incur penalty by bombs laid repeatedly on the same set of squares.
        # Penalty scales quadratically by the number of repeated bombings.
        _, _, _, (x, y) = old_game_state['self']
        if self_action == 'BOMB':    
            self.bomb_history.append((x, y))
            _, counts = np.unique(self.bomb_history, return_counts=True)
            self.bomb_loop_penalty = np.sum((counts-1)**2)
            if self.bomb_loop_penalty > 0:
                events.append(ALREADY_BOMBED_FIELD)

        # Punish if bomb was placed in a position where escape was impossible.
        inescapable = state_old[0,1] == 0
        if inescapable and self_action == 'BOMB':
            events.append(CERTAIN_SUICIDE)

        # Punish if agent chose to wait, but no tile in its surrounding was lethal.
        if self_action == 'WAIT':
            arena_old = old_game_state['field']
            bombs_old = [xy for (xy, t) in old_game_state['bombs']]
            directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
            lethal_directions = np.zeros(5)
            for i, (ix, iy) in enumerate(directions): 
                if not has_object(ix, iy, arena_old, 'wall'):
                    lethal_directions[i] = int(is_lethal(ix, iy, arena_old, bombs_old))
                else:
                    lethal_directions[i] = -1
            if not any(lethal_directions == 1):
                events.append(WAITED_UNNECESSARILY)

        if new_game_state:
            # Extract feature vector:
            state_new = state_to_features(new_game_state)

            # The agent's lethal status at its position in each game state.
            lethal_new = state_new[0,2]
            lethal_old = state_old[0,2]
            if lethal_new > lethal_old:
                events.append(ENTERED_LETHAL)
            elif lethal_old > lethal_new:
                events.append(ESCAPED_LETHAL)

            # Closer to/further from the closest coin.
            coin_step_new = state_new[0,5]
            coin_step_old = state_old[0,5]
            if coin_step_new != -1 and coin_step_old != -1:
                if coin_step_old > coin_step_old:
                    events.append(CLOSER_TO_COIN)
                elif coin_step_old < coin_step_new:
                    events.append(FURTHER_FROM_COIN)

            # Closer to/further from best crate position.
            crate_step_new = state_new[0,8]
            crate_step_old = state_old[0,8]
            if crate_step_new != -1 and crate_step_old != -1:
                if crate_step_old > crate_step_new:
                    events.append(CLOSER_TO_CRATE)
                elif crate_step_old < crate_step_new:
                    events.append(FURTHER_FROM_CRATE)

    # Reward for surviving:
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)         
    
    # ---------- (2) Compute n-step reward and store tuple in Transitions: ----------
    self.n_step_transitions.append((transform(self, old_game_state), self_action, transform(self, new_game_state), reward_from_events(self, events)))
    
    if len(self.n_step_transitions) == N_STEPS:
        
        reward_arr = np.array([self.n_step_transitions[i][-1] for i in range(N_STEPS)])
        n_step_reward = ((GAMMA)**np.arange(N_STEPS)).dot(reward_arr)
        
        n_step_old_feature_state = self.n_step_transitions[0][0]
        n_step_new_feature_state = self.n_step_transitions[0][2]
        n_step_action = self.n_step_transitions[0][1]
        
        after_n_step_new_feature_state = self.n_step_transitions[-1][2]
        
        self.transitions.append(Transition(n_step_old_feature_state, n_step_action, n_step_new_feature_state, n_step_reward, after_n_step_new_feature_state))

    # ---------- (3) Store the game state for feature extration function learning: ----------
    # Store the game state for learning of feature extration function.    
    if old_game_state and not self.dr_override:
        self.state_history.append(state_to_vect(old_game_state)[0])

    # ---------- (4) For evaluation purposes: ----------
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates +=1
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
    
    # ---------- (1) Compute last n-step reward and store tuple in Transitions ----------
    self.n_step_transitions.append((transform(self, last_game_state), last_action, None, reward_from_events(self, events)))
    
    if len(self.n_step_transitions) == N_STEPS:
        reward_arr = np.array([self.n_step_transitions[i][-1] for i in range(N_STEPS)])
        n_step_reward = ((GAMMA)**np.arange(N_STEPS)).dot(reward_arr)
        
        n_step_old_feature_state = self.n_step_transitions[0][0]
        n_step_new_feature_state = self.n_step_transitions[0][2]
        n_step_action = self.n_step_transitions[0][1]

        self.transitions.append(Transition(n_step_old_feature_state, n_step_action, n_step_new_feature_state, n_step_reward, None))
    
    # Store the game state for learning of feature extration function.
    if last_game_state and not self.dr_override:
        self.state_history.append(state_to_vect(last_game_state)[0])

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
        X, targets, residuals = [], [], []
        for state, action, state_next, n_step_reward, n_step_next_state in batch:
            # Current state cannot be the state before game start.
            if state is not None:
                # Q-values for the current state.
                if self.model_is_fitted:
                    # Q-value function estimate.
                    q_values = self.model.predict(state)
                else:
                    # Zero initialization.
                    q_values = np.zeros(self.action_size).reshape(1, -1)

                # Q-value update for the given state and action.
                if self.model_is_fitted and n_step_next_state is not None:
                    # Non-terminal next state and pre-existing model.
                    maximal_response = np.max(self.model.predict(n_step_next_state))
                    q_update =  (n_step_reward + GAMMA**N_STEPS *  maximal_response)
                else:
                    # Either next state is terminal or a model is not yet fitted.
                    q_update = n_step_reward

                # Assign Q-value update. # TODO: possible to introduce a learning rate.
                q_values[0][self.actions.index(action)] = q_update

                # Append feature data and targets for the regression.
                X.append(state[0])
                targets.append(q_values[0])

                # Prioritized experience replay.
                if PRIO_EXP_REPLAY and self.model_is_fitted:
                    # Calculate the residuals for the training instance.
                    action_idx = self.actions.index(action)
                    X_tmp = X[-1].reshape(1, -1)
                    target = targets[-1][action_idx]
                    q_estimate = self.model.predict(X_tmp)[0][action_idx]
                    res = (target - q_estimate)**2
                    residuals.append(res)

        # Prioritized experience replay.
        if PRIO_EXP_REPLAY and self.model_is_fitted:
            # Keep the N samples with the largest squared residuals.
            idx = np.argpartition(residuals, -PRIO_EXP_SIZE)[-PRIO_EXP_SIZE:]

            # Update the training set.            
            X = [X[i] for i in list(idx)]
            targets = [targets[i] for i in list(idx)]

        # Regression fit.
        self.model.fit(X, targets) 
        #self.model.partial_fit(X, targets)
        self.model_is_fitted = True

        # Raise flag for export of the learned model.
        self.perform_export = True

    # ---------- (4) Improve dimensionality reduction: ----------
    # Learn a new (hopefully improved) model for dimensionality reduction.
    if ((not self.dr_override) and
        (self.game_nr % DR_FREQ == 0) and 
        (len(self.state_history) > DR_MINIBATCH_SIZE)):
        
        # Minibatch learning on the collected samples # TODO: Try out sampling with/without replacement.
        for _ in range(DR_EPOCHS):
            batch = random.sample(self.state_history, DR_MINIBATCH_SIZE)
            self.dr_model.partial_fit(np.vstack(batch)) #TODO: Fix this broken POS.
        self.dr_model_is_fitted = True

        # Since the feature extraction function is now changed, we need to start
        # the learning process of the Q-value function over from scratch.
        # Create a new, but unfitted, Q-value model of the same type as before.
        self.model = clone(self.model)
        self.model_is_fitted = False

        # Empty lists of transitions, bomb history and game states.
        self.transitions.clear()
        self.bomb_history.clear()
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
    # Total score in this game.
    score = np.sum(self.score_in_round)
   
    # Append results to each specific list.
    self.historic_data['score'].append(score)
    self.historic_data['coins'].append(self.collected_coins)
    self.historic_data['crates'].append(self.destroyed_crates)
    self.historic_data['games'].append(self.game_nr)   
    if self.act_strategy == 'eps-greedy':
        self.historic_data['exploration'].append(self.epsilon)
    elif self.act_strategy == 'softmax':
        self.historic_data['exploration'].append(self.tau)

    # Store the historic record.
    with open(FNAME_DATA, "wb") as file:
        pickle.dump(self.historic_data, file)
    
    # Reset game score, coins collected and one up the game count.
    self.score_in_round  = 0
    self.collected_coins = 0
    self.destroyed_crates = 0
    self.game_nr += 1
    
    # Plot training progress every n:th game.
    if self.game_nr % PLOT_FREQ == 0:
        # Incorporate the full training history.
        games_list = self.historic_data['games']
        score_list = self.historic_data['score']
        coins_list = self.historic_data['coins']
        crate_list = self.historic_data['crates']
        explr_list = self.historic_data['exploration']

        # Plotting
        fig, ax = plt.subplots(4, sharex=True)

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

        # Exploration rate (epsilon/tau) per game.
        ax[3].plot(games_list, explr_list)
        if self.act_strategy == 'eps-greedy':        
            ax[3].set_title('$\epsilon$-greedy: Exploration rate $\epsilon$')
            ax[3].set_ylabel('$\epsilon$')
        elif self.act_strategy == 'softmax':
            ax[3].set_title('Softmax: Exploration rate $\\tau$')
            ax[3].set_ylabel('$\\tau$')
        ax[3].set_xlabel('Game #')

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
    passive_constant = -0.1 # Always added to rewards for every step in game.
    lethal_movement  = 0.5   # Moving in/out of lethal range.
    coin_movement    = 0.25  # Moving closer to/further from closest coin.
    crate_movement   = 0.1  # Moving closer to/further from best crate position.
    loop_factor      = -0.1  # Scale factor for loop penalty.

    # Game reward dictionary:
    game_rewards = {
        # Custom events:
        SURVIVED_STEP:  passive_constant,
        ALREADY_BOMBED_FIELD: loop_factor * self.bomb_loop_penalty,
        ENTERED_LETHAL: -lethal_movement,
        ESCAPED_LETHAL: lethal_movement,
        CLOSER_TO_COIN: coin_movement,
        FURTHER_FROM_COIN: -coin_movement,
        CLOSER_TO_CRATE: crate_movement,
        FURTHER_FROM_CRATE: -crate_movement,
        CERTAIN_SUICIDE: -5,
        WAITED_UNNECESSARILY: -0.5,
        
        # Default events:
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,
        
        e.BOMB_DROPPED: 0.5,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 0.2,
        e.COIN_COLLECTED: 5,

        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,

        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: passive_constant,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
