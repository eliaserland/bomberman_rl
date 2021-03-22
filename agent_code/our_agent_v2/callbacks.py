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
FILENAME = "crates_nightrun"  # Base filename of model (excl. extensions).
ACT_STRATEGY = 'eps-greedy'        # Options: 'softmax', 'eps-greedy'
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
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.0001, warm_start=True)) #, penalty='elasticnet'))
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
    mask, valid_actions = get_valid_action(game_state)

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

    # Flat array with info of own agent.pos, y])

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
    Converts the game state dictionary to a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # Extracting information from the game state dictionary.
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = [xy for (xy, t) in game_state['bombs']]         
    others = [xy for (n, s, b, xy) in game_state['others']]

    #            ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    #directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
 
    # ---- COINS ----
    # Normalized vector indicating the direction of the closest coin and the no.
    # of steps to get there
    coin_dir = closest_coin_dir(x, y, coins, arena, bombs, others)

    # ---- CRATES ----
    # Find tile within a specified radius which can be reached by agent and destroyes the most crates
    crates_direction = crates_dir(x, y, 30, arena, bombs, others)

    # ---- ESCAPE FROM BOMB ----
    # Agent can escape from its own bomb, if placed at its current position.
    escapable = int(is_escapable(x, y, arena))

    # Normalized vector indicating the direction of the nearest non-lethal tile.
    escape_direction = escape_dir(x, y, arena, bombs, others)

    # ---- LETHAL INDICATOR  -----
    # Check if the agent's position is lethally dangerous.
    lethal = int(is_lethal(x, y, arena, bombs))
 
    # ---------------------------
    # Joining the scalar features into a numpy vector.
    scalar_feat = np.array([int(bombs_left), escapable, lethal])
    # Concatenating all subvectors into the final feature vector.
    features = np.concatenate((scalar_feat, coin_dir, crates_direction, escape_direction), axis=None)
    # [bombs_left, escapable, lethal, coin_x, coin_y, coin_step, crate_x, crate_y, crate_step, escape_x, escape_y]
    # [         0,         1,      2,      3,      4,         5,       6,       7,          8,        9,       10]
    return features.reshape(1, -1)

# ---- Potentials ----
def gaussian(x: np.array, y: np.array, sigma: float=1, height: float=1) -> np.array:
    return height*np.exp(-0.5*(x**2+y**2)/sigma**2)

def bomb_pot(x: np.array, y: np.array, diag: float=20, height: float=10) -> np.array:
    return height*(np.clip((np.abs(x)+np.abs(y)+diag*np.abs(x*y))/4, None, 1)-1)

def inv_squared(x: np.array, y: np.array, height: float=1) -> np.array:
    return height*1/(1+x**2+y**2)
# --------------------

def closest_coin_dir(x: int, y: int, coins: list, arena: np.array, bombs: list, others: list) -> np.array:
    """
    # TODO: Outdated function description, redo.
    Given the agent's position at (x,y) get the normalized position vector
    towards the closest revealed coin and the l1 distance to this coin. Returns
    the zero vector with -1 as the l1 distance if no coins are present.
    """
    reachable = False # initialization
    if coins:
        # Perform a breadth-first search for the closest coin.
        q = Queue()
        visited = []
        graph = {}  
        root = ((x, y), 0, (None, None)) # ((x, y), steps, (parent_x, parent_y))
        visited.append(root[0])
        q.put(root)
        while not q.empty():
            (ix, iy), steps, parent = q.get()
            graph[(ix, iy)] = parent
            if (ix, iy) in coins:
                reachable = True
                cx, cy, c_steps = ix, iy, steps
                break
            neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
            for neighb in neighbours:
                if not neighb in visited:
                    visited.append(neighb)
                    q.put((neighb, steps+1, (ix, iy)))
        if reachable:
            # Traverse graph backwards to recover the path to the closest coin.
            s = []          # empty sequence
            node = (cx, cy) # target node
            if graph[node] != (None, None) or node == (x, y):
                while node != (None, None):
                    s.insert(0, node)  # insert at the front of the sequence
                    node = graph[node] # get the parent node
            # Assignment of direction towards the coin.
            if len(s) > 1:
                next_node = s[1] # The very next node on path towards the coin.
                rel_pos = (next_node[0]-x, next_node[1]-y)
                return np.concatenate((rel_pos, c_steps), axis=None)
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
    if not has_object(x, y, arena, 'wall'):
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
    raise ValueError("Lethal status undefined for tile of type 'wall'.")


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
    if has_object(x, y, arena, 'free'):
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
    else:
        raise ValueError("Can only check escape status on free tiles.")

def get_free_neighbours(x: int, y: int, arena: np.array, bombs: list, others: list) -> list:
    """
    Get a list of all free and unoccupied tiles directly neighbouring the
    position with indices (x,y).
    """
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    neighbours = []
    random.shuffle(directions) # Randomize such that no direction is prioritized.
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
    # TODO: Update with direction to closest tile

    escapable = False # initialization
    if bombs:
        # Breadth-first search for the closest non-lethal position.
        q = Queue()  # Create a queue.
        visited = [] # List to keep track of visited positions.
        graph = {}   # Saving node-parent relationships.
        root = ((x, y), (None, None)) # ((x, y), (parent_x, parent_y))
        visited.append(root[0])       # Mark as visited.
        q.put(root)                   # Put in queue.
        while not q.empty():             
            (ix, iy), parent = q.get()              
            graph[(ix, iy)] = parent
            if not is_lethal(ix, iy, arena, bombs):
                escapable = True
                break
            neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
            for neighbour in neighbours:
                if not neighbour in visited:
                    visited.append(neighbour)
                    q.put((neighbour, (ix, iy)))
        if escapable:
            # Traverse the graph backwards from the target node to the source node.
            s = []          # empty sequence
            node = (ix, iy) # target node
            if graph[node] != (None, None) or node == (x, y):
                while node != (None, None):
                    s.insert(0, node)  # Insert at the front of the sequence.
                    node = graph[node] # Get the parent.
            # Assigning a direction towards the escape tile.
            if len(s) > 1:
                next_node = s[1] # The very next node towards the escape tile.
                rel_pos = (next_node[0]-x, next_node[1]-y)
                return np.array(rel_pos)
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
   
    # Breadth-first search for the tile with most effective bomb placement.
    q = Queue()
    visited, graph = [], {}
    root = ((x, y), 0, (None, None)) # ((x, y), steps, (parent_x, parent_y))
    visited.append(root[0]) # Keeping track of visited nodes.
    q.put(root)             
    while not q.empty():
        (ix, iy), steps, parent = q.get() # Taking the next node from the queue.
        if steps > n:                     # Stop condition.
            continue
        graph[(ix, iy)] = parent          # Save the node with its parent.
        
        # Determine no. of destructible crates and escape status.
        crates = destructible_crates(ix, iy, arena)
        if crates > 0 and is_escapable(ix, iy, arena):
            candidates.append((crates, steps, (ix, iy)))
        
        # Traversing to the neighbouring nodes.
        neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
        for neighb in neighbours:
            if not neighb in visited:
                visited.append(neighb)
                q.put((neighb, steps+1, (ix, iy)))
    
    if candidates:
        # Finding the best tile from the candidates.
        w_max = 0
        for crates, steps, (ix, iy) in candidates:
            w = crates/(4+steps) # Average no. of destroyed crates per step.
            if w > w_max:
                w_max = w
                cx, cy, c_steps = ix, iy, steps
        
        # Traverse the graph backwards from the target node to the source node.
        s = []          # empty sequence
        node = (cx, cy) # target node
        if graph[node] != (None, None) or node == (x, y):
            while node != (None, None):
                s.insert(0, node)  # Insert at the front of the sequence.
                node = graph[node] # Get the parent.

        # Assigning a direction and distance to the tile.
        if len(s) > 1:
            next_node = s[1] # The very next node towards the best tile.
            rel_pos = (next_node[0]-x, next_node[1]-y)
            return np.concatenate((rel_pos, c_steps), axis=None)
        else:
            return np.zeros(3)
    return np.array([0, 0, -1])

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
    if bombs_left and is_escapable(x, y, arena) and aggressive_play:
        valid_actions.append(ACTIONS[-1])
        mask[-1] = 1

    # Convert binary mask to boolean mask of the valid moves.
    mask = (mask == 1)
    
    # Convert list to numpy array (# TODO Is this neccesary?)
    valid_actions = np.array(valid_actions)

    return mask, valid_actions
