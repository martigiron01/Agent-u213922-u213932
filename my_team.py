# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import heapq
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.distance_calculator import Distancer
from util import nearest_point

SOME_DEFAULT_VALUE = 5

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class BaseAgent(CaptureAgent):
    """
    Base class for offensive and defensive agents with common methods.
    """
    def register_initial_state(self, game_state):
        """
        Initializes the agent.
        """
        super().register_initial_state(game_state)
        self.distancer = Distancer(game_state.data.layout)
        self.distancer.get_maze_distances()  # Pre-compute maze distances

    def get_current_position(self, game_state):
        """
        Returns the current position of the agent.
        """
        return game_state.get_agent_position(self.index)

class OffensiveAgent(BaseAgent):
    """
    Offensive agent that collects food and safely returns to its midfield when it has collected enough food.
    """
    def choose_action(self, game_state):
        """
        Selects the optimal action for collecting food, avoiding enemies, and returning to midfield after collecting 8 food.
        """
        enemies = self.get_visible_enemies(game_state)
        food_list = self.get_food(game_state).as_list()
        my_pos = self.get_current_position(game_state)

        # Get carried food count
        carried_food = game_state.get_agent_state(self.index).num_carrying
        score = game_state.get_score()
        threshold = self.calculate_food_threshold(score)

        # Define the boundary (midfield) to return to
        midfield = self.get_midfield_positions(game_state)
        closest_midfield = min(midfield, key=lambda b: self.distancer.get_distance(my_pos, b))

        # If the agent has enough food or is threatened, return to midfield
        if carried_food >= threshold or self.is_threatened(my_pos, enemies):
            # High likelihood of returning to midfield when carrying enough food
            if random.random() < 0.8:  # 80% chance to return to midfield
                path = self.a_star_search(game_state, my_pos, closest_midfield, enemies)
                if path and len(path) > 1:
                    return self.get_direction(my_pos, path[1])
            else:
                # Still try to collect food in some cases with a lower chance
                if food_list:
                    closest_food = min(food_list, key=lambda f: self.distancer.get_distance(my_pos, f))
                    path = self.a_star_search(game_state, my_pos, closest_food, enemies)
                    if path and len(path) > 1:
                        return self.get_direction(my_pos, path[1])

        elif food_list:
            # Collect food if it's safe and we haven't reached the threshold
            closest_food = min(food_list, key=lambda f: self.distancer.get_distance(my_pos, f))
            path = self.a_star_search(game_state, my_pos, closest_food, enemies)
            if path and len(path) > 1:
                return self.get_direction(my_pos, path[1])

        # Fallback: Choose a safe random action if no clear path
        safe_actions = self.get_safe_actions(game_state, enemies)
        if safe_actions:
            return random.choice(safe_actions)

        return Directions.STOP  # Stop if no safe action is available

    def get_midfield_positions(self, game_state):
        """
        Returns a list of positions in the midfield of the agent's field of play.
        Midfield is typically near the center of the agent's side.
        """
        layout = game_state.data.layout
        mid_x = layout.width // 2
        team_side = range(mid_x) if self.red else range(mid_x, layout.width)
        
        # Midfield will be around the center of the agent's half
        midfield = [(mid_x, y) for y in range(layout.height) if not game_state.has_wall(mid_x, y)]
        return midfield

    def is_threatened(self, position, enemies):
        """
        Determines if the agent is under threat based on enemy proximity.
        """
        return any(self.distancer.get_distance(position, enemy) <= 4 for enemy in enemies)

    def get_visible_enemies(self, game_state):
        """
        Returns a list of positions of visible enemies who are not scared.
        """
        enemies = []
        opponents = self.get_opponents(game_state)
        for opponent in opponents:
            enemy_state = game_state.get_agent_state(opponent)
            if (
                enemy_state
                and enemy_state.get_position() is not None  # Visible
                and not enemy_state.is_pacman  # Not in Pacman mode
                and enemy_state.scared_timer == 0  # Not scared
            ):
                enemies.append(enemy_state.get_position())
        return enemies

    def get_safe_actions(self, game_state, enemies):
        """
        Returns a list of actions that keep the agent at a safe distance from enemies.
        """
        safe_actions = []
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_position(self.index)
            if pos and all(self.distancer.get_distance(pos, enemy) > 2 for enemy in enemies):
                safe_actions.append(action)
        return safe_actions
    
    def calculate_food_threshold(self, score):
        """
        Calculates the dynamic food threshold based on the agent's score.
        The more points the agent has, the less food it needs to return with.
        """
        if score > 10:  # High score, return with 2 food
            return 1
        elif score > 5:  # Mid-range score, return with 4 food
            return 2
        else:  # Low score, return with 6 food
            return 3
        
    def a_star_search(self, game_state, start, goal, enemies):
        """
        Implements A* algorithm to find the shortest path from start to goal while avoiding enemies.
        """
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start]))
        closed_set = set()
        g_scores = {start: 0}

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self.get_successors_positions(game_state, current):
                if neighbor in closed_set:
                    continue

                # Avoid positions too close to enemies
                if any(self.distancer.get_distance(neighbor, enemy) <= 2 for enemy in enemies):
                    continue

                new_cost = cost + 1
                if neighbor not in g_scores or new_cost < g_scores[neighbor]:
                    g_scores[neighbor] = new_cost
                    priority = new_cost + self.distancer.get_distance(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))

        return []  # Return empty if no path is found

    def get_successors_positions(self, game_state, position):
        """
        Returns a list of accessible adjacent positions from the given position.
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(position[0] + dx), int(position[1] + dy)
            if not game_state.has_wall(next_x, next_y):
                successors.append((next_x, next_y))
        return successors

    def get_direction(self, current, next_step):
        """
        Returns the direction from current to next_step.
        """
        dx = next_step[0] - current[0]
        dy = next_step[1] - current[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        return Directions.STOP




class DefensiveAgent(BaseAgent):
    """
    Agente defensivo que utiliza la búsqueda alfa-beta para anticipar y bloquear movimientos enemigos.
    """
    def choose_action(self, game_state):
        """
        Selecciona la acción óptima para defender utilizando la búsqueda alfa-beta.
        """
        invaders = self.get_invaders(game_state)

        if invaders:
            # Si hay invasores, interceptarlos
            invader = min(invaders, key=lambda inv: self.distancer.get_distance(self.get_current_position(game_state), inv))
            action = self.alpha_beta_search(game_state, depth=2, target=invader)
            return action if action else Directions.STOP
        else:
            # No hay invasores, moverse hacia la posición de defensa seleccionada dinámicamente
            defend_position = self.select_defend_position(game_state)
            action = self.alpha_beta_search(game_state, depth=2, target=defend_position)
            return action if action else Directions.STOP

    def get_invaders(self, game_state):
        """
        Retorna una lista de posiciones de los invasores (enemigos que son Pac-Man).
        """
        invaders = []
        opponents = self.get_opponents(game_state)
        for opponent in opponents:
            state = game_state.get_agent_state(opponent)
            if state.is_pacman and state.get_position() is not None:
                invaders.append(state.get_position())
        return invaders

    def select_defend_position(self, game_state):
        """
        Selecciona dinámicamente una posición en nuestro territorio para defender, basada en la distribución de la comida.
        """
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if food_list:
            # Seleccionar la comida propia más cercana a la frontera
            mid_x = game_state.data.layout.width // 2
            if self.red:
                min_distance = min([abs(food[0] - (mid_x - 1)) for food in food_list])
                defend_food = [food for food in food_list if abs(food[0] - (mid_x - 1)) == min_distance]
            else:
                min_distance = min([abs(food[0] - mid_x) for food in food_list])
                defend_food = [food for food in food_list if abs(food[0] - mid_x) == min_distance]
            # Seleccionar la posición más céntrica verticalmente
            defend_position = min(defend_food, key=lambda f: abs(f[1] - game_state.data.layout.height // 2))
            return defend_position
        else:
            # Si no hay comida, retornar al centro de nuestro territorio
            mid_x = game_state.data.layout.width // 2
            if self.red:
                return (mid_x - 1, game_state.data.layout.height // 2)
            else:
                return (mid_x, game_state.data.layout.height // 2)

    def alpha_beta_search(self, game_state, depth, target=None):
        """
        Implementa la búsqueda alfa-beta para determinar la mejor acción hacia el objetivo.
        """
        def max_value(state, depth, alpha, beta):
            if depth == 0 or state.is_over():
                return self.evaluate(state, target)
            value = float('-inf')
            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                value = max(value, min_value(successor, depth -1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state, depth, alpha, beta):
            if depth ==0 or state.is_over():
                return self.evaluate(state, target)
            value = float('inf')
            opponents = self.get_opponents(state)
            for opponent in opponents:
                opponent_state = state.get_agent_state(opponent)
                if opponent_state is None or opponent_state.get_position() is None:
                    continue
                for action in state.get_legal_actions(opponent):
                    successor = state.generate_successor(opponent, action)
                    value = min(value, max_value(successor, depth -1, alpha, beta))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
            return value

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            score = min_value(successor, depth -1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action

    def evaluate(self, game_state, target):
        """
        Evalúa el estado del juego desde la perspectiva defensiva.
        """
        my_pos = self.get_current_position(game_state)
        invaders = self.get_invaders(game_state)
        score = 0

        # Penalización por número de invasores
        score -= 1000 * len(invaders)

        # Minimizar la distancia a los invasores
        if invaders:
            distances = [self.distancer.get_distance(my_pos, inv) for inv in invaders]
            score -= 10 * min(distances)

        # Si no hay invasores, moverse hacia la posición de defensa
        elif target is not None:
            distance_to_target = self.distancer.get_distance(my_pos, target)
            score -= distance_to_target

        return score