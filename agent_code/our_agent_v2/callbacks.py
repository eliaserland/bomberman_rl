import os
import pickle
import random
from operator import itemgetter
import warnings
from collections import namedtuple, deque

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

import settings as s
import events as e

warnings.simplefilter(action='ignore', category=FutureWarning)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# ---------------- Parameters ----------------
FILENAME = "SGD_agent_v2"         # Filename of for model output (excl. extension).
ACT_STRATEGY = 'eps-greedy'         # Options: 'softmax', 'eps-greedy'

DR_BATCH_SIZE = 1000
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

    # TODO: Load in old DR model.
    # Incremental PCA for dimensionality reduction of game state.
    n_comp = 100
    self.inkpca = IncrementalPCA(n_components=n_comp, batch_size=DR_BATCH_SIZE) 
    self.dr_override = True  # if True: Use only manual feature extraction.
    self.tx_is_fitted = False   # TODO: REDO THIS, ONLY TEMPORARY

    # Setting up the model.
    if os.path.isfile(fname):
        self.logger.info("Loading model from saved state.")
        with open(fname, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = True
        """
        if self.tx is not None: 
            self.tx_is_fitted = True
        else: 
            self.tx_is_fitted = False
        """
    elif self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.0001, warm_start=True))
        self.model_is_fitted = False
        """
        # TODO: Need additional if statement here?
        if not self.dr_override:
            self.tx = IncrementalPCA(n_components=n_comp, batch_size=DR_BATCH_SIZE) 
        else: 
            self.tx = None
        self.tx_is_fitted = False 
        """
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
    if self.tx_is_fitted and not self.dr_override:
        # Automatic dimensionality reduction.
        return self.tx.transform(state_to_vect(game_state))
    else:
        # Hand crafted feature extraction function.
        return state_to_features(game_state)

def state_to_vect(game_state: dict) -> np.array:
    """
    Converts the game state dictionary to a feature vector. Used
    as pre-proccessing before an automatic feature extraction method.
    """
    return np.array([]).reshape(1, -1) 


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*
    Converts the game state to the input of your model, i.e.
    a feature vector.
    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # Make a simple encoder to give every distance to closest coin a own state number :
    # e.g agent position (x,y) = (1,1) and closest coin position (1,5) -> distance (0 , 4) state number 4  
    # e.g agent position (x,y) = (15,15) and closest coin position (1,1) -> distance (-14 , -14) state number 4 = 23 
    # e.g agent position (x,y) = (1,1) and closest coin position (15,15) -> distance (14 , 14) state number 4 = 23 
    
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']  #is weirdly implemented: Does not stop when wall is in betweeen?
    
    # print("The bomb map is - given \n " , game_state['explosion_map'])
    
    max_distance_x = s.ROWS - 2 #s.ROWS - 3 ? 
    max_distance_y = s.COLS - 2

    # ---------- (1) get relative, normlaized step distances to closest coin: ----------
    
    if len(coins) == 0:
        h = 0   # what to give if no coins left
        v = 0
    else: 
        coins_info = []
        for coin in coins:
            x_coin_dis = coin[0] - x
            y_coin_dis = coin[1] - y
            total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
            coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
            coins_info.append(coin_info)

        closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]
        # print(closest_coin_info)
        if closest_coin_info[2] == 0:
            h = 0
            v = 0
        else:
            h = closest_coin_info[0]/closest_coin_info[2]  #normalize with total difference to coin   
            v = closest_coin_info[1]/closest_coin_info[2]  


    # ---------- (2) encounter for relative postion of agent in arena:  + (later added) is in corner: ----------
    
    # is between two invalide field horizontal (not L and R, do U and D)
    # is between two invalide field vertical (do L and R, not U and D)
    # somewhere else (not L and R, not U and D)
    # will increase number of states with a factor 3
    mask, valid_actions =  get_valid_action(game_state)
    
    relative_position_vertical = 0
    relative_position_horizintal = 0
    is_in_corner = 0
    
    if 'RIGHT' not in valid_actions and 'LEFT' not in valid_actions:
        relative_position_horizintal = 1  # between_invalide_horizintal
    
    if 'UP' not in valid_actions and 'DOWN' not in valid_actions:
        relative_position_vertical = 1  # between_invalide_vertical
        
    if 'UP' not in valid_actions or 'DOWN' not in valid_actions:
        if 'RIGHT' not in valid_actions or 'LEFT' not in valid_actions:
            is_in_corner = 1  # in corner

# ---------- (3) get relative, normalized step distance to closest bomb and its timer: ----------        
    bombs_info = []
    
    if len(bombs) == 0:
        bomb_h = 0   #what to give if no bomb - should be fixed by incountering if in dead zone
        bomb_v = 0
        bomb_timer = 0
        
    else:
        for bomb in bombs:
            x_bomb_dis = bomb[0][0] - x
            y_bomb_dis = bomb[0][1] - y
            timer = bomb[1]
            total_step_distance = abs(x_bomb_dis) + abs(y_bomb_dis)
            bomb_info = (x_bomb_dis , y_bomb_dis , total_step_distance, timer)
            bombs_info.append(bomb_info)
            
        closest_bomb_info = sorted(bombs_info, key=itemgetter(2))[0]

        if closest_bomb_info[2] == 0:
            bomb_h = 0
            bomb_v = 0
            bomb_timer = 0
        else:
            bomb_h = closest_bomb_info[0]/closest_bomb_info[2]  #normalize with total difference to bomb   
            bomb_v = closest_bomb_info[1]/closest_bomb_info[2]  
            bomb_timer = closest_bomb_info[3]

# ---------- (4) get info if agent is in dead zone or not: ---------- 
    bomb_map = np.zeros_like(arena)
    for bomb_xy in bomb_xys:
        bomb_map += get_explosion_map(arena,bomb_xy)

    if bomb_map[x,y] != 0:
        dead_zone = bomb_map[x,y]
    else: dead_zone = 0

# ---------- (5) get number of crates in explosion radius: ---------- 
    
    explosion_rad = get_explosion_map(arena,(x,y))
   
    n_crates_in_explosion_rad = np.count_nonzero((arena[explosion_rad == 1] == 1))

# ---------- (6) to do: check whether escape route  possible if action executed ---------- 
    
    (escape_left, escape_right, escape_down, escape_up ) = possible_escape_route(arena, (x,y))


# ---------- (7) add how far from danger zone: ---------- 
    # don't add, just cannot execute action into bomb zone
    

# ---------- (last) stack all featres together: ----------
    features = np.array([h , v , relative_position_horizintal , relative_position_vertical, is_in_corner , bomb_h , bomb_v , \
                         bomb_timer, dead_zone, n_crates_in_explosion_rad, escape_left, escape_right, escape_down, escape_up])
    
    # print("The bomb map is - mine \n " , bomb_map)
    # print(features)
    return features.reshape(1, -1)


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
    # Use my bomb_map instead:
    # bomb_map = np.zeros_like(arena)

    for (bomb_xy, t) in bombs:
        if t == 0: #explodes in next step
            bomb_map += get_explosion_map(arena,bomb_xy)
    
    # Check for valid actions.
    #            ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    valid_actions = []
    mask = np.zeros(len(ACTIONS))
    
    # Movement:
    for i, d in enumerate(directions):
        if ((arena[d] == 0)    and # Is a free tile
            (bomb_map[d]  == 0) and # No ongoing explosion
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


def get_explosion_map(arena, position):
    ''' 
    Define a map which shows explosion map/radius of a bomb at 'position'
    '''
    
    x, y = position
    blast_coords = [(x, y)]
    power = s.BOMB_POWER 

    for i in range(1, power + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))
    explosion_map = np.zeros_like(arena)
    for blast_coord in blast_coords:
        explosion_map[blast_coord] = 1

    return explosion_map


def possible_escape_route(arena, position):
    ''' 
    Find out if surrounding invalide_tiles block possible escape route in direction x
    '''

    x, y = position
    
    # can escape when action is executed = 1, cannot escape when action is executed = 0:
    escape_left = 1
    escape_right = 1
    escape_down = 1
    escape_up = 1
    
    power = s.BOMB_POWER 

    for i in range(1, power + 2):
        if arena[x + i, y] in list([-1, +1]):
            escape_right = 0
            break
        try:
            if arena[x + i, y + i] == 0 or arena[x + i, y - i] == 0:
                break
        except: 
            pass

    for i in range(1, power + 2):
        if arena[x - i, y]  in list([-1, +1]):
            escape_left = 0
            break
        try:
            if arena[x - i, y + i] == 0 or arena[x - i, y - i] == 0:
                break
        except: 
            pass

    for i in range(1, power + 2):
        if arena[x, y + i]  in list([-1, +1]):
            escape_down = 0
            break
        try:
            if arena[x - i, y + i] == 0 or arena[x + i, y + i] == 0:
                break
        except: 
            pass

    for i in range(1, power + 2):
        if arena[x, y - i]  in list([-1, +1]):
            escape_up = 0
            break
        try:
            if arena[x - i, y - i] == 0 or arena[x + i, y - i] == 0:
                break
        except: 
            pass

    return (escape_left, escape_right, escape_down, escape_up )
