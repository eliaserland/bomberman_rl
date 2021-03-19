import os
import pickle
import random
from operator import itemgetter

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from queue import Queue

import settings as s
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# ---------------- Parameters ----------------
FILENAME = "SGD_pot_v1"         # Base filename of model (excl. extensions).
ACT_STRATEGY = 'softmax'        # Options: 'softmax', 'eps-greedy'
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
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.001, warm_start=True))#, penalty='elasticnet'))
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
    # Extracting information from the game state dictionary.
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = [xy for (xy, t) in game_state['bombs']]         
    others = [xy for (n, s, b, xy) in game_state['others']]

    #            ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]

    # Initialization of the potetials.
    pot_crates = np.zeros(len(directions))
    pot_coins  = np.zeros(len(directions))
    #pot_bombs  = np.zeros(len(directions))

    """
    # ---- Potential-based ----
    # Crates:
    x_crates, y_crates = np.where(arena == 1)
    for i, d in enumerate(directions):
        pot_crates[i] += np.sum(gaussian(x_crates-d[0], y_crates-d[1], sigma=3, height=0.25))
    """

    # Coins
    """
    if coins:
        coins_arr = np.array(coins).T
        for i, d in enumerate(directions):
            pot_coins[i] += np.sum(inv_squared(coins_arr[0]-d[0], coins_arr[1]-d[1]))
    coins_grad = np.zeros(2)
    [(pot_coins[1]-pot_coins[3]),  (pot_coins[0]-pot_coins[2])]
    coins_grad = np.array([])
    """

    """
    # Bombs
    if bombs:
        bomb_arr = np.array(bombs).T
        for i, d in enumerate(directions):
            pot_bombs[i] += np.sum(bomb_pot(bomb_arr[0]-d[0], bomb_arr[1]-d[1])) 
    
    # TODO: GET GRADIENT ESTIMATES
    feat_crates = np.zeros(2)
    feat_coins = np.zeros(2)
    feat_bombs = np.zeros(2)crates_direction
    """
    # -------------------------
    # -------------------------
    # -------------------------
    
    # Normalized vector indicating the direction of the closest coin.
    coin_dir = closest_coin_dir(x, y, coins)

    # No. of creates that would get destroyed by a bomb at the agent's position.
    crates = destructible_crates(x, y, arena)
    # TODO: Features using destructible_crates() for tiles in the agent's immedate surrounding.

    # Find tile within a specified radius which can be reached by agent and destroyes the most crates
    crates_direction = crates_dir(x, y, 8, arena, bombs, others)

    # Agent can escape from a potential bomb placed at its current position.
    escapable = int(is_escapable(x, y, arena))

    # Normalized vector indicating the direction of the nearest non-lethal tile.
    escape_direction = escape_dir(x, y, arena, bombs, others)

    # Check if the tiles in the agent's surrounding are lethal.
    lethal_directions = np.zeros(5)
    for i, (ix, iy) in enumerate(directions):
        lethal_directions[i] = int(is_lethal(ix, iy, arena, bombs))
    
    # Joining the scalar features into a numpy vector.
    scalar_feat = np.array([int(bombs_left), crates, escapable])
    
    # Concatenating all vectors into the final feature vector.
    features = np.concatenate((scalar_feat, escape_direction, lethal_directions, coin_dir, crates_direction), axis=None)
    # [bombs_left, crates, escapable, escape_dir_x, escape_dir_y, lethal_1, lethal_2, lethal_3, lethal_4, lethal_own, coin_x, coin_y, coin_step, crate_x, crate_y, crate_step]
    # [         0,      1,         2,            3,            4,        5,        6,        7,        8,          9,     10,     11,        12,      13,      14,         15]

    return features.reshape(1,-1)


def gaussian(x: np.array, y: np.array, sigma: float=1, height: float=1) -> np.array:
    return height*np.exp(-0.5*(x**2+y**2)/sigma**2)

def bomb_pot(x: np.array, y: np.array, diag: float=20, height: float=10) -> np.array:
    return height*(np.clip((np.abs(x)+np.abs(y)+diag*np.abs(x*y))/4, None, 1)-1)

def inv_squared(x: np.array, y: np.array, height: float=1) -> np.array:
    return height*1/(1+x**2+y**2)

def closest_coin_dir(x: int, y: int, coins: list) -> np.array:    
    """
    Given the agent's position at (x,y) get the normalized position vector
    towards the closest revealed coin and the l1 distance to this coin. Returns 
    the zero vector with -1 as the l1 distance if no coins are present.  
    """
    if coins:
        l1_dist = []
        for cx, cy in coins:
            l1_dist.append(abs(cx-x)+abs(cy-y))
        (cx, cy), l1 = coins[np.argmin(l1_dist)], np.min(l1_dist) 
        rel_pos = (cx-x, cy-y)
        if rel_pos != (0, 0):
            rel_dir = rel_pos / np.linalg.norm(rel_pos) 
            return np.concatenate((rel_pos, l1), axis=None)

        else:
            return np.zeros(3)
    return np.array([0, 0, -1])

def has_object(x: int, y: int, arena: np.array, object: str) -> bool:
    """
    Check if tile at position (x,y) is of the specified type.
    """
    if object == 'crate':
        return arena[x,y] == 1
    elif object == 'free':
        return arena[x,y] == 0
    elif object == 'wall':
        return arena[x,y] == -1
    else:
        raise ValueError(f"Invalid object {object}")

def increment_position(x: int, y: int, direction: str) -> (int, int):
    """
    Standing at position (x,y), take a step in the specified direction.
    """
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
    """
    Standing at position (x,y) and facing the direction specified, get the
    position indices of the two tiles directly to the sides of (x,y).
    """
    if direction == 'UP' or direction == 'DOWN':
        jx, jy, kx, ky = x+1, y, x-1, y
    elif direction == 'RIGHT' or direction == 'LEFT':
        jx, jy, kx, ky = x, y+1, x, y-1
    else:
        raise ValueError(f"Invalid direction {direction}")
    return jx, jy, kx, ky

def is_lethal(x: int, y: int, arena: np.array, bombs: list) -> bool:
    """
    Check if position (x,y) is within the lethal range of any of the ticking
    bombs. Returns True if the position is within blast radius.
    """
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']        
    if bombs:        
        for (bx, by) in bombs:
            if bx == x and by == y:
                return True
            for direction in directions:
                ix, iy = bx, by
                ix, iy = increment_position(ix, iy, direction)
                while (not has_object(ix, iy, arena, 'wall') and
                       abs(ix-bx) <= 3 and abs(iy-by) <= 3):
                    if ix == x and iy == y:
                        return True
                    ix, iy = increment_position(ix, iy, direction)
    return False

def destructible_crates(x: int, y: int, arena: np.array) -> int:
    """
    Count the no. of crates that would get destroyed by a bomb placed at (x,y).
    Returns -1 if (x,y) is an invalid bomb placement.
    """
    if has_object(x, y, arena, 'free'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        crates = 0
        for direction in directions:
            ix, iy = x, y
            ix, iy = increment_position(ix, iy, direction)
            while (not has_object(ix, iy, arena, 'wall') and
                abs(x-ix) <= 3 and abs(y-iy) <= 3):
                if has_object(ix, iy, arena, 'crate'):
                    crates += 1
                ix, iy = increment_position(ix, iy, direction)
        return crates
    else:
        return -1

def is_escapable(x: int, y: int, arena: np.array) -> bool:
    """
    Assuming the agent is standing at (x,y), check if an escape from a bomb
    dropped at its own position is possible (not considering other agents'
    active bombs). Returns True if an escape from own bomb is possible.
    # TODO: is_escapable() needs to consider that another agents might block its way.
    # TODO: is_escapable() should consider the effect of other agent's active bombs.
    """
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for direction in directions:
        ix, iy = x, y
        ix, iy = increment_position(ix, iy, direction)
        while has_object(ix, iy, arena, 'free'):
            if abs(x-ix) > 3 or abs(y-iy) > 3:
                return True
            jx, jy, kx, ky = check_sides(ix, iy, direction)
            if (has_object(jx, jy, arena, 'free') or
                has_object(kx, ky, arena, 'free')):
                return True
            ix, iy = increment_position(ix, iy, direction)
    return False

def get_free_neighbours(x: int, y: int, arena: np.array, bombs: list, others: list) -> list:
    """
    Get a list the positions of all free tiles neighbouring position (x,y).
    """
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    neighbours = []
    for ix, iy in directions:
        if (has_object(ix, iy, arena, 'free') and
            not (ix, iy) in bombs and
            not (ix, iy) in others):
            neighbours.append((ix, iy))
    return neighbours

def escape_dir(x: int, y: int, arena: np.array, bombs: list, others: list) -> np.array:
    """
    Given agent's position at (x,y) find the direction to the closest non-lethal
    tile. Returns a normalized vector indicating the direction. Returns the zero
    vector if the bombs cannot be escaped or if there are no active bombs.
    """
    escapable = False # initialization
    if bombs:
        # Breadth-first search for the closest non-lethal position.
        q = Queue()  # Create a queue.
        visited = [] # List to keep track of visited positions.
        root = (x,y)
        visited.append(root)
        q.put(root)
        while not q.empty():
            ix, iy = q.get()
            if not is_lethal(ix, iy, arena, bombs):
                escapable = True
                break
            neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
            for neighbour in neighbours:
                if not neighbour in visited:
                    visited.append(neighbour)
                    q.put(neighbour)
        if escapable:
            rel_pos = (ix-x, iy-y)
            if rel_pos != (0, 0):
                return rel_pos / np.linalg.norm(rel_pos) 
    return np.zeros(2)


def crates_dir(x: int, y: int, n: int, arena: np.array, bombs: list, others: list) -> np.array:
    """
    Given the agent's position at (x,y) find the tile within n steps which
    would yield the largest amount of destroyed crates if a bomb where to be
    placed there. Returns a vector with the normalized direction to this tile
    and the number of steps to get there. Returns the zero vector with -1 as the
    step count if no crates could be found within the n step radius.
    """
    candidates = [] 
    alpha = 0.25 # Weighing the importance of short distance to no. of crates.
                 # alpha -> 0   : only considering no. of crates
                 # alpha -> inf : only considering shortest distance

    # Breadth-first search for tile with most effective bomb placement.
    q = Queue()
    visited = [] 
    root = ((x, y), 0) # ((x, y), steps)
    visited.append(root[0])
    q.put(root)
    while not q.empty():
        (ix, iy), steps = q.get()
        if steps > n:
            continue
        crates = destructible_crates(ix, iy, arena)
        if crates > 0:
            candidates.append((crates, steps, (ix, iy)))
        neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
        for neighb in neighbours:
            if not neighb in visited:
                visited.append(neighb)
                q.put((neighb, steps+1))
    if candidates:
        w_max = 0
        for crates, steps, (ix, iy) in candidates:
            w = crates * np.exp(-alpha*steps)
            if w > w_max:
                w_max = w
                cx, cy, c_steps = ix, iy, steps
        rel_pos = (cx-x, cy-y) 
        if rel_pos != (0, 0):
            rel_dir = rel_pos / np.linalg.norm(rel_pos) 
            return np.concatenate((rel_dir, c_steps), axis=None)
        else: 
            return np.zeros(3)
    return np.array([0, 0, -1])








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
