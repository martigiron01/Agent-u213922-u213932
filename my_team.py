# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import heapq
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.distance_calculator import Distancer

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
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
        self.distancer.get_maze_distances()  # Pre-computes maze distances

    def get_current_position(self, game_state):
        """
        Returns the current position of the agent.
        """
        return game_state.get_agent_position(self.index)

    def get_visible_enemies(self, game_state):
        """
        Returns a list of visible enemy positions that are not scared.
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

    def get_action_to_position(self, game_state, start, target):
        """
        Gets the best action to move towards a target position.
        """
        actions = game_state.get_legal_actions(self.index)
        best_dist = float('inf')
        best_action = Directions.STOP

        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(pos, target)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action

    def get_midfield_positions(self, game_state):
        """
        Returns a list of positions at the agent's side of the field.
        """
        layout = game_state.data.layout
        mid_x = layout.width // 2
        patrol_points = []

        # Adjust x-coordinate based on team (one step back from middle line)
        if self.red:
            x = mid_x - 1  # One step back from middle for red team
        else:
            x = mid_x  # One step back from middle for blue team

        # Define points along the adjusted line
        for y in range(1, layout.height - 1):
            if not game_state.has_wall(x, y):
                patrol_points.append((x, y))

        return patrol_points


class OffensiveAgent(BaseAgent):
    """
    An offensive agent designed to efficiently collect food while managing risks.
    Implements sophisticated decision-making for food collection, capsule usage, and escape strategies.
    """
    def choose_action(self, game_state):
        """
        Selects the optimal action based on the current game state.
        
        Strategic priorities:
        1. Collect food efficiently when safe
        2. Return home when carrying sufficient food (threshold = 3)
        3. Evaluate escape strategies when threatened:
           - Consider capsule acquisition vs returning home
           - Use probabilistic decision making for unpredictability
        """
        enemies = self.get_visible_enemies(game_state)
        food_list = self.get_food(game_state).as_list()
        my_pos = self.get_current_position(game_state)

        # Track food carrying status and threshold
        carried_food = game_state.get_agent_state(self.index).num_carrying
        threshold = 3

        # Calculate strategic positions
        midfield = self.get_midfield_positions(game_state)
        closest_midfield = min(midfield, key=lambda b: self.distancer.get_distance(my_pos, b))
        capsules = self.get_capsules(game_state)

        # Strategic decision making when carrying sufficient food or under threat
        if carried_food >= threshold or self.is_threatened(my_pos, enemies):
            # Evaluate escape options: capsule acquisition vs returning home
            if capsules:
                closest_capsule = min(capsules, key=lambda c: self.distancer.get_distance(my_pos, c))
                capsule_path = self.a_star_search(game_state, my_pos, closest_capsule, enemies)
            else:
                capsule_path = None

            home_path = self.a_star_search(game_state, my_pos, closest_midfield, enemies)

            # Choose between capsule and home based on path efficiency
            if capsule_path and (not home_path or len(capsule_path) < len(home_path)):
                if len(capsule_path) > 1:
                    return self.get_direction(my_pos, capsule_path[1])
            elif home_path and len(home_path) > 1:
                # Implement probabilistic return strategy for unpredictability
                if random.random() < 0.8:  # 80% chance to return when carrying sufficient food
                    return self.get_direction(my_pos, home_path[1])

        # Food collection strategy when safe
        elif food_list:
            closest_food = min(food_list, key=lambda f: self.distancer.get_distance(my_pos, f))
            path = self.a_star_search(game_state, my_pos, closest_food, enemies)
            if path and len(path) > 1:
                return self.get_direction(my_pos, path[1])

        # Fallback: Choose a safe random action if no clear path exists
        safe_actions = self.get_safe_actions(game_state, enemies)
        if safe_actions:
            return random.choice(safe_actions)

        return Directions.STOP

    def is_threatened(self, position, enemies):
        """
        Determines if the agent is under threat based on enemy proximity.
        """
        return any(self.distancer.get_distance(position, enemy) <= 4 for enemy in enemies)

    def get_safe_actions(self, game_state, enemies):
        """
        Identifies actions that maintain a safe distance from enemies.
        """
        safe_actions = []
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_position(self.index)
            if pos and all(self.distancer.get_distance(pos, enemy) > 2 for enemy in enemies):
                safe_actions.append(action)
        return safe_actions

    def a_star_search(self, game_state, start, goal, enemies):
        """
        Implements A* pathfinding algorithm to find optimal paths while avoiding enemies.
        """
        open_set = []
        heapq.heappush(open_set, (0 + self.distancer.get_distance(start, goal), 0, start, [start]))
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

        return []  # Return empty list if no path exists

class DefensiveAgent(BaseAgent):
    """
    An advanced defensive agent that implements a decision-making strategy based on a decision tree.
    Focuses on protecting territory, managing threats, and strategic patrolling.
    """
    def choose_action(self, game_state):
        """
        Implements the main decision-making logic for the defensive agent.
        
        Strategic priorities:
        1. Return to home territory if out of bounds
        2. Seek safe positions when scared
        3. Chase visible invaders
        4. Investigate missing food
        5. Patrol strategically
        """
        # Initialize agent state if not already done
        if not hasattr(self, 'initial_food'):
            self.opponents = self.get_opponents(game_state)
            self.initial_food = len(self.get_food_you_are_defending(game_state).as_list())
            self.patrol_points = self.get_midfield_positions(game_state)
            self.current_patrol_point = self.get_best_patrol_point(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # 1. Check if the agent is in enemy territory
        if self.is_in_enemy_territory(my_pos, game_state):
            return self.get_action_to_home(game_state)
        
        # 2. Handle scared state
        if my_state.scared_timer > 0:
            # If there are enemies, move to a safe position
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            visible_enemies = [e for e in enemies if e.get_position() is not None]
            
            if visible_enemies:
                # Find safe spots and move to the best one
                safe_spots = self.get_midfield_positions(game_state)
                if safe_spots:
                    best_spot = max(safe_spots, 
                                  key=lambda x: min(self.get_maze_distance(x, e.get_position()) 
                                                  for e in visible_enemies))
                    return self.get_action_to_position(game_state, my_pos, best_spot)
            return self.get_safe_action(game_state)
        
        # 3. Detect and chase invaders or investigate missing food
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        
        if invaders:
            # Prioritize invaders carrying food
            target_invader = max(invaders, key=lambda x: x.num_carrying)
            return self.get_action_to_position(game_state, my_pos, target_invader.get_position())
        
        # 4. Investigate missing food if no visible invaders
        defending_food = self.get_food_you_are_defending(game_state)
        current_food_count = len(defending_food.as_list())
        if current_food_count < self.initial_food:
            self.initial_food = current_food_count
            closest_food = min(defending_food.as_list(), 
                             key=lambda x: self.get_maze_distance(my_pos, x))
            return self.get_action_to_position(game_state, my_pos, closest_food)
        
        # 5. Strategic patrolling
        if self.should_change_patrol_point(game_state):
            self.current_patrol_point = self.get_best_patrol_point(game_state)
        
        return self.patrol_action(game_state)

    def is_in_enemy_territory(self, position, game_state):
        """
        Checks if a position is in enemy territory.
        """
        mid_x = game_state.data.layout.width // 2
        return (self.red and position[0] > mid_x) or (not self.red and position[0] <= mid_x)

    def get_action_to_home(self, game_state):
        """
        Calculates the best action to return to home territory.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        midfield = self.get_midfield_positions(game_state)
        target = min(midfield, key=lambda x: self.get_maze_distance(my_pos, x))
        return self.get_action_to_position(game_state, my_pos, target)

    def get_safe_action(self, game_state):
        """
        Finds an action that keeps the agent near enemies but at a safe distance.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [e for e in enemies if e.get_position() is not None]

        # If no visible enemies, return STOP
        if not visible_enemies:
            return Directions.STOP

        # Filter positions that keep the agent close but at a safe distance
        safe_positions = self.get_successors_positions(game_state, my_pos)
        if not safe_positions:
            return Directions.STOP

        # Filter positions based on distance to enemies
        safe_positions = [pos for pos in safe_positions if all(
            2 < self.get_maze_distance(pos, enemy.get_position()) <= 4
            for enemy in visible_enemies
        )]

        # If no safe positions found, try to move away from enemies
        if not safe_positions:
            actions = game_state.get_legal_actions(self.index)
            if actions:
                # Find the action that maximizes distance from enemies
                best_action = max(actions, key=lambda action: min(
                    self.get_maze_distance(
                        game_state.generate_successor(self.index, action).get_agent_state(self.index).get_position(),
                        enemy.get_position()
                    ) for enemy in visible_enemies
                ))
                return best_action
            return Directions.STOP

        # Choose the position closest to enemies but still safe
        target = min(safe_positions, key=lambda pos: min(
            self.get_maze_distance(pos, enemy.get_position()) for enemy in visible_enemies
        ))
        return self.get_action_to_position(game_state, my_pos, target)

    def get_best_patrol_point(self, game_state):
        """
        Selects the best patrol point based on current position, defended food, and visible enemies.
        """
        my_pos = self.get_current_position(game_state)
        food = self.get_food_you_are_defending(game_state).as_list()
        patrol_points = self.get_midfield_positions(game_state)
        
        if not patrol_points:
            return None

        # Get visible enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [e.get_position() for e in enemies if e.get_position() is not None]
        
        # Evaluate each patrol point
        best_score = float('-inf')
        best_point = None
        
        for point in patrol_points:
            score = 0
            
            # Factor 1: Distance to the nearest food
            if food:
                min_food_dist = min(self.get_maze_distance(point, food_pos) for food_pos in food)
                score += 10.0 / (min_food_dist + 1)  # Closer to food = better
            
            # Factor 2: Distance to visible enemies
            if visible_enemies:
                min_enemy_dist = min(self.get_maze_distance(point, e_pos) for e_pos in visible_enemies)
                if min_enemy_dist < 2:  # Too close to enemies
                    continue
                score += 5.0 / (min_enemy_dist + 1)  # Maintain distance but not too far
            
            # Factor 3: Distance from current position
            current_dist = self.get_maze_distance(my_pos, point)
            score -= current_dist * 0.1  # Penalize points that are too far
            
            if score > best_score:
                best_score = score
                best_point = point
        
        return best_point or patrol_points[0]  # Fallback to the first point if no better option

    def patrol_action(self, game_state):
        """
        Determines the best action for patrolling based on the current situation.
        """
        my_pos = self.get_current_position(game_state)
        
        # If no patrol point or we're at it, get a new one
        if (not self.current_patrol_point or 
            self.get_maze_distance(my_pos, self.current_patrol_point) < 2):
            self.current_patrol_point = self.get_best_patrol_point(game_state)
        
        # If we have a valid point, move towards it
        if self.current_patrol_point:
            return self.get_action_to_position(game_state, my_pos, self.current_patrol_point)
        
        # If no valid point, hold position
        return Directions.STOP

    def should_change_patrol_point(self, game_state):
        """
        Determines if the current patrol point should be changed.
        """
        if not self.current_patrol_point:
            return True
            
        my_pos = self.get_current_position(game_state)
        
        # Change if:
        # 1. We're very close to the current point
        if self.get_maze_distance(my_pos, self.current_patrol_point) < 2:
            return True
            
        # 2. There are enemies dangerously close
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [e for e in enemies if e.get_position() is not None and e.is_pacman]
        if visible_enemies:
            for enemy in visible_enemies:
                if self.get_maze_distance(self.current_patrol_point, enemy.get_position()) < 3:
                    return True
        
        # 3. The nearest food is too far from the current point
        defending_food = self.get_food_you_are_defending(game_state).as_list()
        if defending_food:
            closest_food_dist = min(self.get_maze_distance(self.current_patrol_point, food) 
                                  for food in defending_food)
            if closest_food_dist > 5:  # If the nearest food is too far
                return True
        
        return False