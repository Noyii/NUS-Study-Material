import cv2
import webcolors

def viz_map(env, info, new_info):
        x, _,  y = env.cur_pos

        map_img, _, _ = env.get_task_info()
        cv2.circle(map_img, [int(x*100),int(y*100)], 
                    radius=2,
                    color=webcolors.name_to_rgb("purple"),
                    thickness=2)
        cv2.rectangle(map_img, 
                    (info["curr_pos"][0] * 100, info["curr_pos"][1] * 100),
                    ((info["curr_pos"][0]+1) * 100, (info["curr_pos"][1]+1) * 100),
                    color=webcolors.name_to_rgb("yellow"), thickness=4)
        cv2.rectangle(map_img, 
                    (new_info["curr_pos"][0] * 100, new_info["curr_pos"][1] * 100),
                    ((new_info["curr_pos"][0]+1) * 100, (new_info["curr_pos"][1]+1) * 100),
                    color=webcolors.name_to_rgb("green"), thickness=4)
        
        aspect_ratio = map_img.shape[0] / map_img.shape[1]
        max_width = 3000 if "map1" in env.map_name else 800
        cv2.imshow("map", cv2.cvtColor(cv2.resize(map_img, (max_width, int(max_width * aspect_ratio))), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

def viz_obs(obs, intention_tuple=None, action=None):
        if intention_tuple is not None:
                cv2.putText(obs, text=f"Intention {intention_tuple[0]} T={intention_tuple[1]:0.3f}", 
                                org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=webcolors.name_to_rgb("white"), thickness=1, lineType=cv2.LINE_AA)
        if action is not None:
                cv2.putText(obs, text=f"Action {action}", 
                                org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=webcolors.name_to_rgb("white"), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("Obs", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))