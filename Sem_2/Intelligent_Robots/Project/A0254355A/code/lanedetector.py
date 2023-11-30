import os
import argparse
import cv2
import numpy as np
import time
from termcolor import colored
import webcolors

from gym_duckietown.envs import DuckietownEnv
from utils.milestones import load_milestone
from intelligent_robots_project import LaneFollower

TESTCASE_PATH = "../../testcases"


def _draw_line(image, x1, y1, x2, y2, text="", color=None):
    if color is None:
        color = webcolors.name_to_rgb("purple")

    cv2.line(image, (x1,y1), (x2,y2), color=color, thickness=2)
   
    cv2.putText(image, text, ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1, color=color, thickness=2)
    return image


def gauss(image):
  return cv2.GaussianBlur(image,(5,5),0)


def extract_color_pixels(image, color_name):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if color_name == "yellow":
        # Mask Yellow
        lower = np.uint8([26, 75, 100])
        upper = np.uint8([30, 255, 255])
        structuring_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elif color_name == "white":
        lower  = np.uint8([0, 0, 125])
        upper  = np.uint8([179, 75, 255])
        structuring_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    else:
        raise NotImplementedError()

    extracted_image = cv2.inRange(image_hsv, lower, upper)
    extracted_image_cleaned = cv2.morphologyEx(extracted_image, cv2.MORPH_OPEN, structuring_elem)
    extracted_image_smoothed = cv2.morphologyEx(extracted_image_cleaned, cv2.MORPH_CLOSE, structuring_elem)

    return extracted_image_smoothed


# def color_mask(image, display=False):
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    

#     # Mask Yellow
#     lower_yellow = np.uint8([20, 100, 100])
#     upper_yellow = np.uint8([30, 255, 255])

#     mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

#     # Mask White
#     sensitivity = 100
#     lower_white  = np.uint8([0, 0, 255-sensitivity])
#     upper_white  = np.uint8([255, sensitivity, 255])
#     mask_white = cv2.inRange(image_hsv, lower_white, upper_white)

#     if display:
#         cv2.imshow("Yellow", mask_yellow)
#         cv2.imshow("White", mask_white)
#         cv2.waitKey(10)

#     return mask_yellow, mask_white

def region_mask(image, mask):
    """Mask out region from an 2D image"""

    height, width = image.shape

    # triangle = np.array([
    #                    [(0, height*0.8), (0, height*0.3), (width, height*0.3), (width, height*0.8)]
    #                    ], dtype=np.int32)
    
    obs = np.zeros_like(image)
    
    obs = cv2.fillPoly(obs, mask, 255)
    obs = cv2.bitwise_and(image, obs)
    return obs

def get_yellow_lanes(image, display=False):

    mask_yellow = extract_color_pixels(image, "yellow")

    height, width = image.shape[0], image.shape[1]

    region_mask_polygon = np.array([
                       [(width//2, height*0.8),
                        (width//2, height*0.05),
                        (width, height*0.3),
                        (width, height*0.8)]
                       ], dtype=np.int32)
    yellow_edges = cv2.Canny(gauss(region_mask(mask_yellow, region_mask_polygon)), 150, 200)
    yellow_lines = cv2.HoughLinesP(yellow_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    
    if display:
        if yellow_lines is not None:
            for l in yellow_lines:
                (x1, y1, x2, y2) = l[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(400)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 30, 0, 0, 0, np.pi);
    # lines = cv2.HoughLines(edges, 2, np.pi / 180, 100)
    return yellow_lines




def get_white_lanes(image, display=False):

    mask_yellow, mask_white = color_mask(image, display=display)
    
    white_edges = cv2.Canny(gauss(region(mask_white)), 150, 200)
    white_lines = cv2.HoughLinesP(white_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    if display:
        if white_lines is not None:
            for l in white_lines:
                (x1, y1, x2, y2) = l[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(400)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 30, 0, 0, 0, np.pi);
    # lines = cv2.HoughLines(edges, 2, np.pi / 180, 100)
    return white_lines

def get_lines(image, color_name, region_mask_polygon):

    image = extract_color_pixels(image, color_name)
    image = region_mask(image, region_mask_polygon)
    image = gauss(image)
    
    edges = cv2.Canny(image, 150, 200)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=40)

    if lines is None:
        lines = []
    return lines
    

def is_rightlane_aligned(obs):
    """Inputs a image and outputs if it's in right lane or now"""

    mask_yellow = extract_color_pixels(obs, color_name="yellow")
    
    mask_white = extract_color_pixels(obs, color_name="white")
    
    # mask_red = extract_color_pixels(obs, color_name="red")

    height, width = obs.shape[0], obs.shape[1]

    left_side_mask_polygon = np.array([
                       [(0, height*0.8),
                        (0, height*0.3),
                        (width//2, height*0.05),
                        (width//2, height*0.8)]
                       ], dtype=np.int32)
    right_side_mask_polygon = np.array([
                       [(width//2, height*0.8),
                        (width//2, height*0.05),
                        (width, height*0.3),
                        (width, height*0.8)]
                       ], dtype=np.int32)
    

    

    # Draw Mask and lines in them
    cv2.polylines(obs, left_side_mask_polygon, isClosed=True, color=(255, 0, 0))
    yellow_lanes = get_lines(obs, "yellow", left_side_mask_polygon)
    max_yellow_slope = 0
    max_yellow_slope_sign = 0
    
    for lane in yellow_lanes:
        x1, y1, x2, y2 = lane[0]
        slope = np.arctan2((y2-y1), (x2-x1))
        if np.abs(slope) < 0.2:
            continue
        if np.abs(slope) > max_yellow_slope:
            max_yellow_slope = np.abs(slope)
            max_yellow_slope_sign = np.sign(slope)
            
        _draw_line(obs, x1, y1, x2, y2, f"Slope {slope:0.3f}", color=webcolors.name_to_rgb("green"))

    
    cv2.polylines(obs, right_side_mask_polygon, isClosed=True, color=(0, 0, 0))
    white_lanes = get_lines(obs, "white", right_side_mask_polygon)
    max_white_slope = 0
    max_white_slope_sign = 0

    for lane in white_lanes:
        x1, y1, x2, y2 = lane[0]
        slope = (y2-y1) / (x2-x1)
        if np.abs(slope) < 0.2:
            continue
        if np.abs(slope) > max_white_slope:
            max_white_slope = np.abs(slope)
            max_white_slope_sign = np.sign(slope)

        _draw_line(obs, x1, y1, x2, y2, f"Slope {slope:0.3f}", color=webcolors.name_to_rgb("blue"))
    
    print(max_yellow_slope*max_yellow_slope_sign, max_white_slope*max_white_slope_sign)
    if (max_yellow_slope > 0.5 or max_white_slope > 0.5) and (max_yellow_slope_sign * max_white_slope_sign < 0):
        return True
    return False


if __name__ == "__main__":
    """Loads a map and alignes the Duckie bot into right lane
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', '-m', default="map5_0", type=str)

    config = parser.parse_args()

    # Load Maps
    testcases = load_milestone(os.path.join(TESTCASE_PATH, "milestone2.json"))
    milestone = testcases[config.map_name]

    env = DuckietownEnv(
        domain_rand=False,
        max_steps=50000,
        map_name=config.map_name,
        seed=int(milestone['seed'][0]),
        user_tile_start=milestone['start'],
        goal_tile=milestone['goal'],
        randomize_maps_on_reset=False
    )

    env.seed(int(milestone['seed'][0]))
    obs = env.reset()

    TANK_LEFT = np.array([0, +1.5])
    TANK_RIGHT = np.array([0, -1.5])
    action = TANK_LEFT

    alignment_timesteps = 0
    while alignment_timesteps < 90:
        flag = is_rightlane_aligned(obs)
        
        cv2.imshow("Lane Detection", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        action = TANK_LEFT if alignment_timesteps < 30 else TANK_RIGHT
        
            
        obs, reward, done, new_info = env.step(action)
        print(f"Action {action[0]:.3f},{action[1]:.3f} |\t {colored(new_info['curr_pos'], 'green')} |\t {colored(reward, 'blue')} | In Lane ", end="")
        if flag:
            print(colored(flag, "green"))
            time.sleep(3)
        else:
            print(colored(flag, "red"))
            time.sleep(0.05)
        alignment_timesteps += 1




