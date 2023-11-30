import math
import numpy as np
from gym_duckietown.simulator import AGENT_SAFETY_RAD, LanePosition

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15

DEFAULT_DISCRETE_INTENTIONS = ["forward", "right", "left"]

def intention_to_idx(intention): 
    return DEFAULT_DISCRETE_INTENTIONS.index(intention)

def idx_to_intention(idx): 
    return DEFAULT_DISCRETE_INTENTIONS[idx]

class PurePursuitPolicyAligner:
    """
    A Pure Pusuit controller class to act as an expert to the model
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images
    
    loss(*args)
        takes images and target action to compute the loss function used in optimization

    predict(observation)
        takes an observation image and predicts using env information the action
    """
    def __init__(self, env, ref_velocity=REF_VELOCITY, following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        """
        Parameters
        ----------
        ref_velocity : float
            duckiebot maximum velocity (default 0.7)
        following_distance : float
            distance used to follow the trajectory in pure pursuit (default 0.24)
        """
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity

    def predict(self, observation, intention_tuple=[0,0]):
        """
        Parameters
        ----------
        observation : image
            image of current observation from simulator
        intention_tuple: List[Int] of size 2. [intention_idx, intention_timestep]
            intention_timestep ranges from 0 to 1
        Returns
        -------
        action: list
            action having velocity and omega of current observation
        """
        intention = idx_to_intention(int(intention_tuple[0]))
        intention_timestep = intention_tuple[1]

        closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        if closest_point is None or closest_tangent is None:
            self.env.reset()
            closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        current_world_objects = self.env.objects

        lookup_distance = self.following_distance

        current_tile_pos = self.env.get_grid_coords(self.env.cur_pos)
        current_tile = self.env._get_tile(*current_tile_pos)
        # print(f"Current tile kind \033[94m {current_tile['kind']} \033[0m  Curr Pos {self.env.cur_pos}, {self.env.cur_angle}")
        

        velocity = self.ref_velocity

        _, closest_point, curve_point= self._get_projected_angle_difference(lookup_distance, intention, intention_timestep, current_tile['kind'])

        if closest_point is None:  # if cannot find a curve point in max iterations
            return [0,0]

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        right_vec = np.array([math.sin(self.env.cur_angle),0,math.cos(self.env.cur_angle)])
        dot = np.dot(right_vec, point_vec)
        omega = -1 * dot

        
        

        # velocity_scale *= (1 - np.abs(omega))**2

        velocity = self.ref_velocity 
        velocity *= (1.5 - np.abs(omega))**2
        # velocity = np.clip(velocity, 0.4, self.ref_velocity)
        action = [velocity , omega]

        return action
    

    def _get_projected_angle_difference(self, lookup_distance, intention="forward", intention_timestep=0, tile_kind='straight'):
        # Find the projection along the path
        cur_angle = self.env.cur_angle

        # if tile_kind.startswith('4way') or tile_kind.startswith('3way'):
        #     if intention == "right" and intention_timestep<=0.8:
        #         cur_angle += (-np.pi/2) 
        #     elif intention == "left" and intention_timestep<=0.8:
        #         cur_angle += (np.pi/4) - min(0, intention_timestep-0.3)
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, cur_angle)
        
        iterations = 0
        curve_angle = None

        while iterations < 10:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_angle = self.env.closest_curve_point(follow_point, cur_angle)

            # If we have a valid point on the curve, stop
            if curve_angle is not None and curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_angle is None:  # if cannot find a curve point in max iterations
            return None, None, None

        else:
            return np.dot(curve_angle, closest_tangent), closest_point, curve_point
