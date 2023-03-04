import Queue as queue
import pickle
import os

from base_planner import *

ANGLE = np.pi/4
DISCRETIZE = 8


class CSDA_Planner(Planner, object):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(CSDA_Planner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)

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
        print "Generating plan"

        goal = self._get_goal_position()
        visited = {self._discretize_state(init_pose): self._euclidean(init_pose, goal)}
        parent = {}
        distance = {init_pose: 0}
        v_w = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1)]
        additional_costs = [0.5, 0.5, 0, 1, 1]
        pq = queue.PriorityQueue()
        pq.put((0, init_pose))

        while not pq.empty():
            _, curr_state = pq.get()
            print "Current State: ", curr_state

            # Goal reached
            if self._euclidean(curr_state, goal) < 0.3:
                self._get_action_sequence(init_pose, curr_state, parent)
                break

            neighbors = []
            for (v, w) in v_w:
                w = w * ANGLE
                first_step = self.motion_predict(curr_state[0], curr_state[1], curr_state[2], v, w)
                if first_step:
                    n = self.motion_predict(first_step[0], first_step[1], first_step[2], v, w)
                    neighbors.append(n)
            
            for (neighbor, (v, w), c) in zip(neighbors, v_w, additional_costs):
                if neighbor == None:
                    continue
                
                neighbor = self._round_off_theta(neighbor)
                g = distance[curr_state] + 1 + c
                f = g + self._euclidean(neighbor, goal)
                d_neighbor = self._discretize_state(neighbor)

                if d_neighbor in visited.keys():
                    prev_f = visited[d_neighbor] 
                    if prev_f < f:
                        # ignore this neighbor as a better one exists for the same discrete state
                        continue

                print "Neighbor: ", neighbor, "f value: ", f, "v_w: ", (v, w)
                visited[d_neighbor] = f
                distance[neighbor] = g
                parent[neighbor] = (curr_state, (v, w))
                pq.put((f, neighbor))

    def _discretize_state(self, pose):
        return (round(pose[0]), round(pose[1]), round(pose[2]/ANGLE) % DISCRETIZE)

    def _round_off_theta(self, state):
        theta = (round(state[2]/ANGLE) % DISCRETIZE) * ANGLE
        return (state[0], state[1], theta)

    def _get_action_sequence(self, init_pose, goal, parent):
        reverse_path = [goal]
        self.action_seq = []

        while goal != init_pose:
            goal, (v, w) = parent[goal]
            reverse_path.append(goal)
            self.action_seq.append((v, w * ANGLE))
            self.action_seq.append((v, w * ANGLE))
        
        self.action_seq = list(reversed(self.action_seq))
        path = list(reversed(reverse_path))
        print "Path: ", path
        print "Action sequence: "

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
    inflation_ratio = 5  # TODO: You should change this value accordingly
    planner = CSDA_Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan(robot.get_current_discrete_state())

    # TODO: FILL ME!
    print planner.action_seq
    robot.publish_continuous_control(planner.action_seq, goal)

    # save your action sequence
    result = np.array(planner.action_seq)
    filename = "../Controls/CSDA_com1building_" + str(goal[0]) + "_" + str(goal[1]) + ".txt"
    np.savetxt(filename, result, fmt="%.2e")
