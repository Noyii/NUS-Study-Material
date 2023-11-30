import queue
import numpy as np

from lanedetector import get_yellow_lanes
from utils.intentions import intention_to_idx
from termcolor import colored


INTENTION_FORWARD = 'forward'
INTENTION_LEFT = 'left'
INTENTION_RIGHT = 'right'
TANK_LEFT = np.array([0, +1])
TANK_RIGHT = np.array([0, -1])


class HighLevelPlanner():

    def __init__(self, env, init_pos, second_pos, goal_pos) -> None:
        self.env = env
        self.init_pos = tuple(init_pos)
        self.second_pos = tuple(second_pos)
        self.goal_pos = tuple(goal_pos)
        self.grid_height = env.grid_height
        self.grid_width = env.grid_width
        
        self.drivable_tiles = {}
        for tiles in self.env.drivable_tiles:
            self.drivable_tiles[tiles['coords']] = {'kind': tiles['kind'], 'angle': tiles['angle']}

        self.high_level_path = None
        self.action_seq = None

    def perform_a_star(self):
        # Calculate A* from 1 step ahead
        distance = {
            self.init_pos: 0,
            self.second_pos: 1
            }
        visited = [self.init_pos]
        parent = {self.second_pos: self.init_pos}
        pq = queue.PriorityQueue()
        pq.put((0, self.second_pos))

        # Normal A*
        while not pq.empty():
            _, curr_state = pq.get()

            if curr_state in visited:
                continue
            
            if curr_state == self.goal_pos:
                self._get_action_sequence(self.init_pos, curr_state, parent)
                return self.action_seq

            visited.append(curr_state)
            
            # Expand neighbour
            neighbors = self._get_neighbors(curr_state)
            
            prev_state = parent[curr_state]
            for neighbor in neighbors:
                prev_orientation = self._get_orientation(curr_state, prev_state)
                next_orientation = self._get_orientation(neighbor, curr_state)
                is_forward = prev_orientation == next_orientation
                if is_forward:
                    g = distance[curr_state] + 1
                else:
                    is_right = self._take_a_turn(prev_orientation, next_orientation) == 'right'
                    g = distance[curr_state] + (4 if is_right else 2)
    
                # Turn A Star to BFS
                # f = g + 0
                f = g + self._manhatten(neighbor, self.goal_pos)
                pq.put((f, neighbor))

                if (neighbor not in distance) or (distance[curr_state]+f < distance[neighbor]):
                    distance[neighbor] = distance[curr_state] + f
                    parent[neighbor] = curr_state


    def _get_path(self, init_pose, goal, parent):
        reverse_path = [goal]

        while goal != init_pose:
            goal = parent[goal]
            reverse_path.append(goal)

        return list(reversed(reverse_path))
    

    def _get_orientation(self, curr, prev):
        diff_x = curr[0] - prev[0]
        diff_y = curr[1] - prev[1]
        
        if diff_x > 0:
            return 0
        if diff_y < 0:
            return 3
        elif diff_x < 0:
            return 2
        return 1


    def _take_a_turn(self, prev_orientation, next_orientation):
        if prev_orientation == 0:
            if next_orientation == 1:
                return INTENTION_RIGHT
            else:
                return INTENTION_LEFT
        elif prev_orientation == 1:
            if next_orientation == 0:
                return INTENTION_LEFT
            else:
                return INTENTION_RIGHT
        elif prev_orientation == 2:
            if next_orientation == 1:
                return INTENTION_LEFT
            else:
                return INTENTION_RIGHT
        else:
            if next_orientation == 0:
                return INTENTION_RIGHT
            else:
                return INTENTION_LEFT
            

    def _get_action_sequence(self, init_pose, goal, parent):
        self.high_level_path = self._get_path(init_pose, goal, parent)
        prev_orientation = self._get_orientation(self.high_level_path[1], init_pose)
        current_intention = INTENTION_FORWARD
        self.action_seq = [(init_pose, INTENTION_FORWARD)]

        for i in range(1, len(self.high_level_path)):
            prev = self.high_level_path[i-1]
            curr = self.high_level_path[i]
            next_orientation = self._get_orientation(curr, prev)

            if next_orientation != prev_orientation:
                neighbors = self._get_neighbors(prev)

                # If only 1 action is available, choose Forward
                # if len(neighbors) == 1:
                #     current_intention = INTENTION_FORWARD
                # else:
                current_intention = self._take_a_turn(prev_orientation, next_orientation)
                prev_orientation = next_orientation
            else:
                current_intention = INTENTION_FORWARD

            self.action_seq.append((curr, current_intention))


    def _get_neighbors(self, current):
        (x, y) = current
        candidate_neighbors = [(x, y+1), (x+1, y), (x, y-1), (x-1, y)]
        final_neighbors = []

        for n in candidate_neighbors:
            (x, y) = n

            if x < 0 or x >= self.grid_width:
                continue
            if y < 0 or y >= self.grid_height:
                continue
            if n not in self.drivable_tiles.keys():
                continue

            final_neighbors.append(n)
        
        return final_neighbors
            
    
    def _euclidean(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def _manhatten(self, a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    