import Queue as queue
import pickle
import os

from base_planner import *


class DSDA_Planner(Planner, object):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        super(DSDA_Planner, self).__init__(world_width, world_height, world_resolution, inflation_ratio)

    def setup_map(self):
        """Get the occupancy grid and inflate the obstacle by some pixels.

        You should implement the obstacle inflation yourself to handle uncertainty.
        """
        # Hint: search the ROS message defintion of OccupancyGrid
        occupancy_grid = rospy.wait_for_message('/map', OccupancyGrid)
        self.map = occupancy_grid.data
        
        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask
        # you should inflate the map to get self.aug_map
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
                    if j >= 0 and j < self.world_width and i!=row and j!=col:
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

        visited = []
        parent = {}
        v_w = [(1,0), (0, 1), (0, 3)]
        goal = self._get_goal_position()
        distance = {init_pose: 0}
        pq = queue.PriorityQueue()
        pq.put((0, init_pose))

        while not pq.empty():
            _, curr_state = pq.get()
            if curr_state in visited:
                continue
            
            if (curr_state[0], curr_state[1]) == goal:
                self._get_action_sequence(init_pose, curr_state, parent)
                break

            visited.append(curr_state)
            neighbors = [ \
                self.discrete_motion_predict(curr_state[0], curr_state[1], curr_state[2], v, w) \
                    for (v, w) in v_w]
            
            for neighbor in neighbors:
                if neighbor == None:
                    continue 

                f = distance[curr_state] + 1 + self._euclidean(neighbor, goal)
                pq.put((f, neighbor))

                if (neighbor not in distance) or (distance[curr_state]+1 < distance[neighbor]):
                    distance[neighbor] = distance[curr_state] + 1
                    parent[neighbor] = curr_state
    
    def _get_path(self, init_pose, goal, parent):
        reverse_path = [goal]

        while goal != init_pose:
            goal = parent[goal]
            reverse_path.append(goal)

        return list(reversed(reverse_path))

    def _get_action_sequence(self, init_pose, goal, parent):
        print "Generating action sequence"
        path = self._get_path(init_pose, goal,parent)
        print path
        self.action_seq = []

        for i in range(len(path)-1):
            curr_state = path[i]
            next_state = path[i+1]
            orientation1 = curr_state[2]
            orientation2 = next_state[2]

            if orientation1 != orientation2:
                if orientation1 == 0:
                    if orientation2 == 1:
                        self.action_seq.append((0, 1))
                    else:
                        self.action_seq.append((0, -1))
                elif orientation1 == 1:
                    if orientation2 == 0:
                        self.action_seq.append((0, -1))
                    else:
                        self.action_seq.append((0, 1))
                elif orientation1 == 2:
                    if orientation2 == 1:
                        self.action_seq.append((0, -1))
                    else:
                        self.action_seq.append((0, 1))
                else:
                    if orientation2 == 0:
                        self.action_seq.append((0, 1))
                    else:
                        self.action_seq.append((0, -1))
            else:
                self.action_seq.append((1, 0))
    
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
    planner = DSDA_Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan(robot.get_current_discrete_state())

    # TODO: FILL ME!
    print len(planner.action_seq)
    robot.publish_discrete_control(planner.action_seq, goal)

    # save your action sequence
    result = np.array(planner.action_seq)
    filename = "../Controls/DSDA_com1building_" + str(goal[0]) + "_" + str(goal[1]) + ".txt"
    np.savetxt(filename, result, fmt="%.2e")
