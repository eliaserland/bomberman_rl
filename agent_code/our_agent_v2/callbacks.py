import os
import pickle
import random
from operator import itemgetter

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

import settings as s
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# ---------------- Parameters ----------------
FILENAME = "SGD_potential_v2"         # Base filename of model (excl. extensions).
ACT_STRATEGY = 'softmax'          # Options: 'softmax', 'eps-greedy'
# --------------------------------------------

fname = f"{FILENAME}.pt" # Adding the file extension.

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Save constants.
    self.action_size = len(ACTIONS)
    self.actions = ACTIONS

    # Assign decision strategy.
    self.act_strategy = ACT_STRATEGY

    # Incremental PCA for dimensionality reduction of game state.
    n_comp = 100
    self.dr_override = True  # if True: Use only manual feature extraction.

    # Setting up the full model.
    if os.path.isfile(fname):
        self.logger.info("Loading model from saved state.")
        with open(fname, "rb") as file:
            self.model, self.dr_model = pickle.load(file)
        self.model_is_fitted = True
        if self.dr_model is not None:
            self.dr_model_is_fitted = True
        else:
            self.dr_model_is_fitted = False

    elif self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.01, warm_start=True, penalty='elasticnet'))
        if not self.dr_override:
            self.dr_model = IncrementalPCA(n_components=n_comp)
        else:
            self.dr_model = None
        self.model_is_fitted = False
        self.dr_model_is_fitted = False
    else:
        raise ValueError(f"Could not locate saved model {fname}")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # --------- (1) Only allow valid actions: -----------------
    mask, valid_actions =  get_valid_action(game_state)

    # --------- (2a) Softmax decision strategy: ---------------
    if self.act_strategy == 'softmax':
        # Softmax temperature. During training, we anneal the temperature. In
        # game mode, we use a predefined (optimal) temperature. Limiting cases:
        # tau -> 0 : a = argmax Q(s,a) | tau -> +inf : uniform prob dist P(a).
        if self.train:
            tau = self.tau
        else:
            tau = 0.1 # TODO: Hyper-parameter which needs optimization.
        if self.model_is_fitted:
            self.logger.debug("Choosing action from softmax distribution.")
            # Q-values for the current state.
            q_values = self.model.predict(transform(self, game_state))[0][mask]
            # Normalization for numerical stability.
            qtau = q_values/tau - np.max(q_values/tau)
            # Probabilities from Softmax function.
            p = np.exp(qtau) / np.sum(np.exp(qtau))        
        else:
            # Uniformly random action when Q not yet initialized.
            self.logger.debug("Choosing action uniformly at random.")
            p = np.ones(len(valid_actions))/len(valid_actions)
        # Pick choice from valid actions with the given probabilities.
        return np.random.choice(valid_actions, p=p)

    # --------- (2b) Epsilon-Greedy decision strategy: --------
    elif self.act_strategy == 'eps-greedy':
        if self.train:
            random_prob = self.epsilon
        else:
            random_prob = 0.1 # TODO: Hyper-parameter which needs optimization.    
        if random.random() < random_prob or not self.model_is_fitted:
            self.logger.debug("Choosing action uniformly at random.")
            execute_action = np.random.choice(valid_actions)
        else:
            self.logger.debug("Choosing action with highest q_value.")
            q_values = self.model.predict(transform(self, game_state))[0][mask]
            execute_action = valid_actions[np.argmax(q_values)]
        return execute_action
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")

def transform(self, game_state: dict) -> np.array:
    """
    Feature extraction from the game state dictionary. Wrapper that toggles
    between automatic and manual feature extraction.
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    if self.dr_model_is_fitted and not self.dr_override:
        # Automatic dimensionality reduction.
        return self.dr_model.transform(state_to_vect(game_state))
    else:
        # Hand crafted feature extraction function.
        return state_to_features(game_state)

def state_to_vect(game_state: dict) -> np.array:
    """
    Converts the game state dictionary to a feature vector. Used
    as pre-proccessing before an automatic feature extraction method.
    """
    # TODO: Redo this function, this is terrible.
    
    # Flat array of base arena.
    arena = game_state['field'].flatten()

    # Flat array with info of own agent.
    _, _, bombs_left, (x, y) = game_state['self']
    self_info = np.array([int(bombs_left), x, y])

    # 
    bombs    = game_state['bombs']
    bomb_xys = np.array([xy for (xy, t) in bombs]).flatten()
    bombs = np.zeros(8)
    bombs[:len(bomb_xys)] = bomb_xys 

    # Flat array with postions of other agents.
    others = np.zeros(6)
    others_xy = np.array([xy for (n, s, b, xy) in game_state['others']]).flatten()
    others[:len(others_xy)] = others_xy

    coins_xys = np.array(game_state['coins']).flatten()
    coins = np.zeros(18)
    coins[:len(coins_xys)] = coins_xys
    
    # Flat explosion map.
    bomb_map = game_state['explosion_map'].flatten()
    
    out = np.concatenate((arena, self_info, bombs, others, coins, bomb_map), axis=None)

    return out.reshape(1, -1)


def state_to_features(game_state: dict) -> np.array:
    """

    """
    # Self info
    _, _, bombs_left, (x, y) = game_state['self']

    #            ['UP', 'RIGHT', 'DOWN', 'LEFT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    pot_crates = np.zeros(len(directions))
    pot_coins  = np.zeros(len(directions))
    pot_bombs  = np.zeros(len(directions))

    arena = game_state['field']

    # ---- Potential-based ----
    # Crates:
    x_crates, y_crates = np.where(arena == 1)
    for i, d in enumerate(directions):
        pot_crates[i] += np.sum(gaussian(x_crates-d[0], y_crates-d[1], sigma=3, height=0.25))

    # Coins
    coins = game_state['coins']
    if coins:
        coins = np.array(coins).T
        for i, d in enumerate(directions):
            pot_coins[i] += np.sum(gaussian(coins[0]-d[0], coins[1]-d[1], sigma=7, height=5)) 

    # Bombs
    bombs = game_state['bombs']
    if bombs:
        bombs = np.array([xy for (xy, t) in bombs]).T
        for i, d in enumerate(directions):
            pot_bombs[i] += np.sum(bomb_pot(bombs[0]-d[0], bombs[1]-d[1])) 


    # TODO: GET GRADIENT ESTIMATES
    feat_crates = np.zeros(2)
    feat_coins = np.zeros(2)
    feat_bombs = np.zeros(2)

    feat_crates = np.array([pot_crates[0]-pot_crates[2], pot_crates[1]-pot_crates[3]])
    
    
    # -------------------------

    # Number of creates that would get destroyed from this position
    crates = 0
    directions_str = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for direction in directions_str:
        ix, iy = x, y
        ix, iy = increment_position(ix, iy, direction)
        while has_object(ix, iy, arena, 'crate') and abs(x-ix) < 4 and abs(y-iy) < 4:
            crates += 1
            ix, iy = increment_position(ix, iy, direction)
    
    # Count escape ways for each direction.
    escapable = np.zeros(len(direction))
    for i, direction in enumerate(directions_str):
        ix, iy = x, y
        ix, iy = increment_position(ix, iy, direction)
        while has_object(ix, iy, arena, 'free'):
            if abs(x-ix) > 3 or abs(y-iy) > 3:
                escapable[i] += 1 # Possible to run in a straight line outside of blast radius.
                break
            jx, jy, kx, ky = check_sides(ix, iy, direction)
            if has_object(jx, jy, arena, 'free'):
                escapable[i] += 1
            if has_object(kx, ky, arena, 'free'):
                escapable[i] += 1
            ix, iy = increment_position(ix, iy, direction)

    # Converting scalar features to numpy vectors.
    # Own bomb indicator, and number of crates reachable from this position:
    bombs_crates = np.array([int(bombs_left), crates])

    features = np.concatenate((bombs_crates, pot, escapable), axis=None)

    return features.reshape(1,-1)


def gaussian(x: np.array, y: np.array, sigma: float=1, height: float=1) -> float:
    return height*np.exp(-0.5*(x**2+y**2)/sigma**2)

def bomb_pot(x: np.array, y: np.array, diag: float=20, height: float=10) -> float:
    return height*(np.clip((np.abs(x)+np.abs(y)+diag*np.abs(x*y))/4, None, 1)-1)

def has_object(x: int, y: int, arena: np.array, object: str) -> bool:
    if object == 'crate':
        return arena[x,y] == 1
    elif object == 'free':
        return arena[x,y] == 0
    elif object == 'wall':
        return arena[x,y] == -1
    else:
        raise ValueError(f"Invalid object {object}")

def increment_position(x: int, y: int, direction: str) -> (int, int):
    if direction == 'UP':
        y -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'DOWN':
        y += 1
    elif direction == 'LEFT':
        x -= 1
    else:
        raise ValueError(f"Invalid direction {direction}")
    return x, y

def check_sides(x: int, y: int, direction: str) -> (int, int, int, int):
    if direction == 'UP' or direction == 'DOWN':
        jx, jy, kx, ky = x+1, y, x-1, y
    elif direction == 'RIGHT' or direction == 'LEFT':
        jx, jy, kx, ky = x, y+1, x, y-1
    else:
        raise ValueError(f"Invalid direction {direction}")
    return jx, jy, kx, ky

'''
def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state dictionary to a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """    
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']    
    
    max_distance_x = s.ROWS - 2 #s.ROWS - 3 ? 
    max_distance_y = s.COLS - 2

    # (1) get relative,normlaized step distances to closest coin
    coins_info = []
    for coin in coins:
        x_coin_dis = coin[0] - x
        y_coin_dis = coin[1] - y
        total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
        coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
        coins_info.append(coin_info)

    # #TODO:FIX THIS
    h = 0
    v = 0
    if coins_info:
        closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]  # TODO: This breaks with no coins.
        #print(closest_coin_info)
        if closest_coin_info[2] == 0:
            h = 0
            v = 0
        else:
            h = closest_coin_info[0]/closest_coin_info[2]  #normalize with total difference to coin   
            v = closest_coin_info[1]/closest_coin_info[2]  

    # (2) encounter for relative postion of agent in arena: 
    # is between two invalide field horizontal (not L and R, do U and D)
    # is between two invalide field vertical (do L and R, not U and D)
    # somewhere else (not L and R, not U and D)
    # will increase number of states with a factor 3
    mask, valid_actions =  get_valid_action(game_state)
    
    relative_position_vertical   = 0
    relative_position_horizontal = 0

    if 'RIGHT' not in valid_actions and 'LEFT' not in valid_actions:
        relative_position_horizontal = 1
    if 'UP' not in valid_actions and 'DOWN' not in valid_actions:
        relative_position_vertical = 1
    features = np.array([h , v , relative_position_horizontal , relative_position_vertical])
    return features.reshape(1, -1)
'''

def get_valid_action(game_state: dict):
    """
    Given the gamestate, check which actions are valide.

    :param game_state:  A dictionary describing the current game board.
    :return: mask which ACTIONS are executable
             list of VALID_ACTIONS
    """
    aggressive_play = True # Allow agent to drop bombs.

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']
    
    # Check for valid actions.
    #            ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    valid_actions = []
    mask = np.zeros(len(ACTIONS))

    # Movement:
    for i, d in enumerate(directions):
        if ((arena[d] == 0)    and # Is a free tile
            (bomb_map[d] <= 1) and # No ongoing explosion
            (not d in others)  and # Not occupied by other player
            (not d in bomb_xys)):  # No bomb placed

            valid_actions.append(ACTIONS[i]) # Append the valid action.
            mask[i] = 1                      # Binary mask
            
    # Bombing:
    if bombs_left and aggressive_play: 
        valid_actions.append(ACTIONS[-1])
        mask[-1] = 1

    # Convert binary mask to boolean mask of the valid moves.
    mask = (mask == 1)
    
    # Convert list to numpy array (# TODO Is this neccesary?)
    valid_actions = np.array(valid_actions)

    return mask, valid_actions
