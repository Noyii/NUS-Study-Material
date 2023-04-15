import Queue as queue
import pickle
import os
import math

from base_planner import *


class DSPA_Planner(Planner, object):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(DSPA_Planner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)

    def setup_map(self):
        """Get the occupancy grid and inflate the obstacle by some pixels.

        You should implement the obstacle inflation yourself to handle uncertainty.
        """
        # Hint: search the ROS message defintion of OccupancyGrid
        occupancy_grid = rospy.wait_for_message('/map', OccupancyGrid)
        self.map = occupancy_grid.data
        
        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask
        map_path = "../maps/aug_maps/com1building.pkl"
        new_mask = None
        with open(os.path.join(map_path), "rb") as fin:
            new_mask = pickle.load(fin)
        
        if new_mask is None:
            new_mask = self._inflate_obstacle()
            with open(map_path,'wb') as f: pickle.dump(new_mask, f)

        self.aug_map = new_mask
        print "Inflation done"

    def _inflate_obstacle(self):
        original_map = np.reshape(list(self.map), (self.world_height, self.world_width)) # 2D grid
        new_mask = copy.deepcopy(original_map)

        for i in range(self.world_height):
            for j in range(self.world_width):
                if original_map[i, j] != 0: # obstacle is present
                    for neighbor in self._d_neighbors(i, j, self.inflation_ratio):
                        new_mask[neighbor] = original_map[i, j] # inflate
        
        return np.transpose(new_mask)

    def _d_neighbors(self, row, col, d):
        neighbors = []

        for i in range(row-d, row+d+1):
            if i >= 0 and i < self.world_height:
                for j in range(col-d, col+d+1):
                    if j >= 0 and j < self.world_width:
                        if i != row and j != col:
                            neighbors.append((i, j))

        return neighbors

    def generate_plan(self, init_pose):
        """TODO: FILL ME! This function generates the plan for the robot, given
        an initial pose and a goal pose.

        You should store the list of actions into self.action_seq, or the policy
        into self.action_table.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """
        actions = [(1,0), (0, 1), (0, -1)]
        max_iterations = 100
        gamma = 0.9
        delta = 1e-1
        probabilities = {
            (1, 0): {
                (1, 0): 0.9,
                (1, 1): 0.05,
                (1, -1): 0.05
            },
            (0, 1): {(0, 1): 1},
            (0, -1): {(0, -1): 1}
        }
        cols = int(math.ceil(self.world_width * self.resolution))
        rows = int(math.ceil(self.world_height * self.resolution))
        V = np.zeros((cols, rows, 4))

        print "Value Iteration begins"
        for iter in range(max_iterations):
            print "Iteration: ", iter
            max_diff = 0
            V_new = copy.deepcopy(V)

            for i, j, theta in np.ndindex(V.shape):
                values = []
                for action in actions:
                    sum = 0.0
                    for (v, w) in probabilities[action].keys():
                        neighbor = self.discrete_motion_predict(i, j, theta, v, w)
                        if neighbor is None or self.collision_checker(neighbor[0], neighbor[1]):
                            # Penalize action if it results in a collision
                            sum -= 0.5
                            continue

                        sum += probabilities[action][(v, w)] * V[neighbor] 
                    values.append(sum)
                
                V_new[i, j, theta] = self._reward((i, j, theta), goal) + gamma * np.max(values)
                if self.collision_checker(i, j):
                    V_new[i, j, theta] = self._reward((i, j, theta), goal)  # Keeping it constant for obstacle
                max_diff = max(max_diff, abs(V_new[i, j, theta] - V[i, j, theta]))

            V = copy.deepcopy(V_new)  
            print "Delta: ", max_diff
            if max_diff < delta:
                break      
        
        self._get_action_table(V, actions)
    
    def _get_action_table(self, V, actions):
        print "Generating action table"
        self.action_table = {}

        for x, y, theta in np.ndindex(V.shape):
            nxt_poses = [self.discrete_motion_predict(x, y, theta, v, w) for (v, w) in actions]
            values = []
            for n in nxt_poses:
                if n is None:
                    values.append(-np.inf)
                else:
                    values.append(V[n])

            if len(values) == 0:
                self.action_table[(x, y, theta)] = (0, 0)
            else:
                self.action_table[(x, y, theta)] = actions[np.argmax(values)]

    def _reward(self, state, goal):
        if self._euclidean(state, goal) == 0:
            return 10
        if self.collision_checker(state[0], state[1]):
            return -10
        return 0
        
    def _euclidean(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size
        
        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
        
        Returns:
            bool -- True for collision, False for non-collision
        """
        x_pixel = int(x / self.resolution)
        y_pixel = int(y / self.resolution)

        if x_pixel < 0 or x_pixel >= self.world_width:
            return True

        if y_pixel < 0 or y_pixel >= self.world_height:
            return True

        return self.aug_map[x_pixel, y_pixel] != 0


if __name__ == "__main__":
    # TODO: You can generate and save the plan using the code below
    rospy.init_node('planner')
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    robot = RobotClient()
    inflation_ratio = 15  # TODO: You should change this value accordingly
    planner = DSPA_Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan(robot.get_current_discrete_state())

    # for MDP, please dump your policy table into a json file
    filename = "../Controls/DSPA_com1building_" + str(goal[0]) + "_" + str(goal[1]) + ".json"
    dump_action_table(planner.action_table, filename)

    # TODO: FILL ME!
    robot.execute_policy(planner.action_table, goal)
