import os
import pickle
import random

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.base import clone
from queue import Queue

import settings as s
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# ---------------- Parameters ----------------
FILENAME = "final_agent"        # Base filename of model (excl. extensions).
ACT_STRATEGY = 'eps-greedy'     # Options: 'softmax', 'eps-greedy'
ONLY_USE_VALID_ACTIONS = False  # Enable/disable filtering of invalid actions.
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
    # Save the actions.
    self.actions = ACTIONS

    # Assign the decision strategy.
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
        self.model = CustomRegressor(SGDRegressor(alpha=0.0001, warm_start=True))
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
    # --------- (1) Optionally, only allow valid actions: -----------------
    # Switch to enable/disable filter of valid actions.
    if ONLY_USE_VALID_ACTIONS:
        mask, valid_actions = get_valid_actions(game_state, filter_level='full')
    else:
        mask, valid_actions = np.ones(len(ACTIONS)) == 1, ACTIONS

    # --------- (2a) Softmax decision strategy: ---------------
    if self.act_strategy == 'softmax':
        # Softmax temperature. During training, we anneal the temperature. In
        # game mode, we use a predefined (optimal) temperature. Limiting cases:
        # tau -> 0 : a = argmax Q(s,a) | tau -> +inf : uniform prob dist P(a).
        if self.train:
            tau = self.tau
        else:
            tau = 0.1
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
            random_prob = 0.01
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
    # This is the dict before the game begins and after it ends.
    if game_state is None:
        return None
    if self.dr_model_is_fitted and not self.dr_override:
        # Automatic dimensionality reduction.
        return self.dr_model.transform(state_to_features(game_state))
    else:
        # Hand crafted feature extraction function.
        return state_to_features(game_state)

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state dictionary to a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # ---- INFORMATION EXTRACTION ----
    # Getting all useful information from the game state dictionary.
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = [xy for (xy, t) in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    #--------------------------------------------------------------------------
    # ---- DANGER ----
    # Boolean indicator telling if agent is currently standing in mortal danger.
    lethal = is_lethal(x, y, arena, bombs)

    # ---- ESCAPE ----
    # Direction towards the closest escape from imminent danger.
    escape_direction = escape_dir(x, y, arena, bombs, others)
    
    # ---- OTHERS ----
    # Direction towards the best offensive tile against other agents.
    others_direction, others_reached = others_dir(x, y, 5, arena, bombs, others)

    # ---- COINS ----
    # Direction towards the closest reachable coin.
    coin_direction = coins_dir(x, y, coins, arena, bombs, others)

    # ---- CRATES ----
    # Direction towards the best offensive tile for destroying crates.
    crates_direction, crates_reached = crates_dir(x, y, 10, arena, bombs, others)
 
    # ---- ATTACK ----
    # Enemy or crate, the optimum position for bomb-laying has been reached,
    # while having an available bomb and not currently having a lethal status.
    # Prioritizes agents over crates, and as such will ignore crate target tiles
    # if a direction towards an enemy agent is given.
    target_acquired = int((others_reached or (crates_reached and all(others_direction == (0,0))))
                          and bombs_left and not lethal)
    #--------------------------------------------------------------------------
    # [DANGER,         ATTACK,             ESCAPE,           OTHERS,          COINS,           CRATES]
    # [lethal, target_aquired, escape_x, escape_y, other_x, other_y, coin_x, coin_y, crate_x, crate_y]
    # [     0,              1,        2,        3,       4,       5,      6,      7,       8,       9]
    features = np.concatenate((int(lethal), target_acquired,
                                escape_direction, others_direction,
                                coin_direction, crates_direction), axis=None)
    return features.reshape(1, -1)
   

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

def is_escapable(x: int, y: int, arena: np.array) -> bool:
    """
    Assuming the agent is standing at (x,y), check if an escape from a bomb
    dropped at its own position is possible (not considering other agents'
    active bombs). Returns True if an escape from own bomb is possible.
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

def crates_dir(x: int, y: int, n: int, arena: np.array, bombs: list, others: list) -> (np.array, bool):
    """
    Given the agent's position at (x,y) find the tile within a given amount of
    steps which would yield the largest amount of destroyed crates if a bomb
    where to be at this location.
    
    Parameters:
    -----------
    x: int
        Agent's x-coordinate.
    y: int
        Agent's y-coordinate.
    n: int
        The max number of steps from the agent's position to consider.
    arena: np.array shape=(width, height)
        Game state information: walls, crates and free tiles.
    bombs: list
        List of coordinate tuples (x, y) for all currently active bombs.
    others: list
        List of coordinate tuples (x, y) for all other agents.

    Returns:
    --------
    rel_pos: np.array shape=(2,)
        Relative position vector from the agent's position, indicating the next
        tile on the path towards the best crate-destroying position.
        Returns the zero-vector if no candidates could be found within the
        specifed search radius.
    target_reached: bool
        True if we have successfully reached the optimal position.
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
        if steps > n:                     # Stopping condition.
            continue
        graph[(ix, iy)] = parent          # Save the node with its parent.
        
        # Determine no. of destructible crates at the current position.
        crates = destructible_crates(ix, iy, arena)
        # Only save escapable candidates with a non-zero amount of crates.
        if crates > 0 and is_escapable(ix, iy, arena):
            candidates.append((crates, steps, (ix, iy)))
        
        # Traversing to the neighbouring nodes.
        neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
        for neighb in neighbours:
            if not neighb in visited:
                visited.append(neighb)
                q.put((neighb, steps+1, (ix, iy)))
    
    if candidates:
        # Find the best tile from the candidates.
        w_max = 0
        for crates, steps, (ix, iy) in candidates:
            w = crates/(4+steps) # Average no. of destroyed crates per step.
            if w > w_max:
                w_max = w 
                cx, cy = ix, iy 
        # Traverse the graph backwards from the target node to the source node,
        # to recover the best path for the agent.
        s = []          # empty sequence
        node = (cx, cy) # target node
        if graph[node] != (None, None) or node == (x, y):
            while node != (None, None):
                s.insert(0, node)  # Insert at the front of the sequence.
                node = graph[node] # Get the parent.

        if len(s) > 1:
            # We have found a candidate tile, and we are not at this position.
            # Get the very next tile on the path towards the best crate position.
            nx, ny = s[1]
            # Only suggest crate direction if next tile is not lethal.
            if not is_lethal(nx, ny, arena, bombs):
                rel_pos = np.array([nx-x, ny-y])
                return rel_pos, False
        elif len(s) == 1:
            # We have a candidate tile, and we are standing at this position.
            return np.zeros(2), True
    # We have either found no candidate tiles, or a lethal tile is blocking the
    # path towards the best candidate tile.
    return np.zeros(2), False

def coins_dir(x: int, y: int, coins: list, arena: np.array, bombs: list, others: list) -> np.array:
    """
    Find the direction towards the closest revealed and reachable coin.

    Parameters:
    -----------
    x: int
        Agent's x-coordinate.
    y: int
        Agent's y-coordinate.
    coins: list
        List of coordinate tuples (x, y) for all revealed coins.
    arena: np.array shape=(width, height)
        Game state information: walls, crates and free tiles.
    bombs: list
        List of coordinate tuples (x, y) for all currently active bombs.
    others: list
        List of coordinate tuples (x, y) for all other agents.

    Returns:
    --------
    rel_pos: np array shape=(2,)
        Relative position vector from the agent's position, indicating the next
        tile on the path towards the closest revealed and reachable coin.
        Returns the zero-vector if no coins are present or can be reached.
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
                reachable = True # Found the closest reachable coin.
                cx, cy = ix, iy  # Save position. 
                break            # Stop the search.
            neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
            for neighb in neighbours:
                if not neighb in visited:
                    visited.append(neighb)
                    q.put((neighb, steps+1, (ix, iy)))
        if reachable:
            # Traverse graph backwards to recover the path to the closest coin
            # from the position of the agent.
            s = []          # List to hold the sequence of tiles.
            node = (cx, cy) # Target node, coin position.
            if graph[node] != (None, None) or node == (x, y):
                while node != (None, None):
                    s.insert(0, node)  # Insert at the front of the sequence.
                    node = graph[node] # Get the parent node.
            # If we are not already standing at the coin position.
            if len(s) > 1:
                # Get the very next tile on best path towards the coin.
                nx, ny = s[1]
                # Only suggest coin direction if next tile is not lethal.
                if not is_lethal(nx, ny, arena, bombs):
                    rel_pos = np.array([nx-x, ny-y])
                    return rel_pos
    return np.zeros(2)

def is_lethal_for_others(x: int, y: int, arena: np.array, others: list) -> bool:
    """
    Determine if a bomb placed at position (x, y) would place any of the 
    other agents in lethal danger.

    Parameters:
    -----------
    x: int
        x-coordinate for bomb placement.
    y: int
        y-coordinate for bomb placement.
    arena: np.array shape=(width, height)
        Game state information: walls, crates and free tiles.
    others: list
        List of coordinate tuples (x, y) for all other agents.

    Returns:
    --------
    lethal_for_others: bool
        True if a bomb in (x, y) would be lethally dangerous for any agent.
    """
    if has_object(x, y, arena, 'free'):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        lethal_for_others = False
        for direction in directions:
            ix, iy = x, y
            ix, iy = increment_position(ix, iy, direction)
            while (not has_object(ix, iy, arena, 'wall') and
                    abs(x-ix) <= 3 and abs(y-iy) <= 3):
                if (ix, iy) in others:
                    lethal_for_others = True
                    break
                ix, iy = increment_position(ix, iy, direction)
        return lethal_for_others
    else:
        raise ValueError("Can only place bombs at tiles of type 'free'.")

def others_dir(x: int, y: int, n: int, arena: np.array, bombs: list, others: list) -> (np.array, bool):
    """
    Find the best tile for offensive bomb placement against other agents, within
    a search radius of n steps from the agent's position at (x, y).
    
    Parameters:
    -----------
    x: int
        Agent's x-coordinate.
    y: int
        Agent's y-coordinate.
    n: int
        The max number of steps from the agent's position to consider.
    arena: np.array shape=(width, height)
        Game state information: walls, crates and free tiles.
    bombs: list
        List of coordinate tuples (x, y) for all currently active bombs.
    others: list
        List of coordinate tuples (x, y) for all other agents.

    Returns:
    --------
    rel_pos: np.array shape=(2,)
        Relative position vector from the agent's position, indicating the next
        tile on the path towards the best agent-defeating position. Returns
        the zero-vector if no other agents are present or can be reached.
    target_reached: bool
        True if we have successfully reached the optimal position.
    """
    if others:
        # Initialization
        reachable = False

        # Breadth-first search for the closest reachable tile that would inflict
        # damage to any other agent.
        q = Queue()
        visited, graph = [], {}
        root = ((x, y), 0, (None, None)) # ((x, y), steps, (parent_x, parent_y))
        visited.append(root[0])
        q.put(root)
        while not q.empty():
            (ix, iy), steps, parent = q.get() 
            graph[(ix, iy)] = parent
            if steps > n: # Only consider a certain search radius.
                continue

            # Check if the current tile would be lethal for some other agent
            # and simultaneously escapable for our agent:
            if (is_lethal_for_others(ix, iy, arena, others) and
                is_escapable(ix, iy, arena)):
                reachable = True    # Mark as reachable.
                cx, cy = ix, iy     # Save the position.
                break               # Stop the search.

            # Traversing to the neighbouring nodes.
            neighbours = get_free_neighbours(ix, iy, arena, bombs, others)
            for neighb in neighbours:
                if not neighb in visited:
                    visited.append(neighb)
                    q.put((neighb, steps+1, (ix, iy)))

        if reachable:
            # Traverse the graph backwards to recover the path from the agent
            # to the target tile for the best bomb placement.
            s = []
            node = (cx, cy)
            if graph[node] != (None, None) or node == (x, y):
                while node != (None, None):
                    s.insert(0, node)   # Insert at the front of the sequence.
                    node = graph[node]  # Get the parent.

            if len(s) > 1:
                # We have found a tile for the best placement and we are not currently
                # there. Get the very next tile on the path towards this position.
                nx, ny = s[1]
                # Only suggest movement if the next tile is not lethal.
                if not is_lethal(nx, ny, arena, bombs):
                    rel_pos = np.array([nx-x, ny-y])
                    return rel_pos, False
            elif len(s) == 1:
                # We have found a tile, and we are standing at this position.
                return np.zeros(2), True
    # We have either found no tiles for a good offensive bomb placement, or
    # there is a lethal tile blocking the path towards this position.
    return np.zeros(2), False

def get_valid_actions(game_state: dict, filter_level: str='basic'):
    """
    Given the gamestate, check which actions are valid. Has two filtering levels,
    'basic' where only purely invalid moves are disallowed and 'full' where also
    bad moves (bombing at inescapable tile, moving into lethal region) are 
    forbidden.

    :param game_state:  A dictionary describing the current game board.
    :param filter_level: Either 'basic' or 'full'
    :return: mask which ACTIONS are executable
             list of VALID_ACTIONS
    """
    aggressive_play = True # Allow agent to drop bombs.

    # Gather information about the game state
    _, _, bombs_left, (x, y) = game_state['self'] 
    arena  = game_state['field']
    coins  = game_state['coins']
    bombs  = [xy for (xy, t) in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    bomb_map = game_state['explosion_map']
    
    # Check for valid actions.
    #            [    'UP',  'RIGHT',   'DOWN',   'LEFT', 'WAIT']
    directions = [(x, y-1), (x+1, y), (x, y+1), (x-1, y), (x, y)]
    
    # Initialization.
    valid_actions = []
    mask = np.zeros(len(ACTIONS))
    disallow_bombing = False
    lethal_status = np.zeros(len(directions))

    # Check the filtering level.
    if filter_level == 'full':
        # Check lethal status in all directions.
        for i, (ix, iy) in enumerate(directions):
            if not has_object(ix, iy, arena, 'wall'):
                lethal_status[i] = int(is_lethal(ix, iy, arena, bombs))
            else:
                lethal_status[i] = -1
        # Verify that there is at least one non-lethal tile in the surrounding.
        if not any(lethal_status == 0):
            # No non-lethal tile detected, we can only disallow waiting.
            lethal_status = np.zeros(len(directions))
            lethal_status[-1] = 1
        
        # Check escape status on the current tile.
        if not is_escapable(x, y, arena):
            disallow_bombing = True

    elif filter_level == 'basic':
        # Could to other things here.
        pass
    else:
        raise ValueError(f"Invalid option filter_level={filter_level}.")

    # Movement:
    for i, d in enumerate(directions):        
        if (arena[d] == 0    and    # Is a free tile
            bomb_map[d] <= 1 and    # No ongoing explosion
            not d in others  and    # Not occupied by other player
            not d in bombs   and    # No bomb placed
            lethal_status[i] == 0): # Is non-lethal.
            valid_actions.append(ACTIONS[i]) # Append the valid action.
            mask[i] = 1                      # Binary mask
            
    # Bombing:
    if bombs_left and aggressive_play and not disallow_bombing:
        valid_actions.append(ACTIONS[-1])
        mask[-1] = 1

    mask = (mask == 1) # Convert binary mask to boolean mask.
    valid_actions = np.array(valid_actions) # Convert list to numpy array

    if len(valid_actions) == 0:
        # The list is empty, there are no valid actions. Return all actions as
        # to not break the code by returning an empty list.
        return np.ones(len(ACTIONS)) == 1, ACTIONS
    else:
        return mask, valid_actions

class CustomRegressor:
    def __init__(self, estimator):
        # Create one regressor for each action separately.
        self.reg_model = [clone(estimator) for i in range(len(ACTIONS))]

    def partial_fit(self, X, y):
        '''
        Fit each regressor individually on its set of data.

        Parameters:
        -----------
        X: list
            List of length len(ACTIONS), where each entry is a 2d array of
            shape=(n_samples, n_features) with feature data corresponding to the
            given regressor. While n_features must be the same for all arrays,
            n_samples can optionally be different in every array.
        y: list
            List of length len(ACTIONS), where each entry is an 1d array of
            shape=(n_samples,) corresponding to each regressor. Since each
            regressor is fully independent, n_samples need not be equal for
            every array in y, but must however match in size to the
            corresponding array in X mentioned above.
        
        Returns:
        --------
        Nothing.
        '''
        # For every action:
        for i in range(len(ACTIONS)):
            # Verify that we have data.
            if X[i] and y[i]:
                # Perform one epoch of SGD.
                self.reg_model[i].partial_fit(X[i], y[i])


    def predict(self, X, action_idx=None):
        '''
        Get predictions from all regressors on a set of samples. Can also return
        predictions by a single regressor.

        Parameters:
        -----------
        X: np.array shape=(n_samples, n_features)
            Feature matrix for the n_samples each with n_features as the number
            of dimensions.
        action_idx: int
            (Optional) if action_idx is specified, only get predictions from
            the chosen regressor.
        
        Returns:
        --------
        y_predict: np.array
            If action_idx is unspecified, return the predictions by all regressors
            for all samples in an array of shape=(n_samples, len(ACTIONS)). Else
            return predictions for the single specified regressor, in an array
            of shape=(n_samples,).
        '''
        if action_idx is None:
            y_predict = [self.reg_model[i].predict(X) for i in range(len(ACTIONS))]
            return np.vstack(y_predict).T # shape=(n_samples, len(ACTIONS))
        else:
            return self.reg_model[action_idx].predict(X) # shape=(n_samples,)

