import json
import os

TESTCASE_PATH = "../../testcases"

def load_milestone(milestone_path):
    with open(milestone_path) as fp:
        testcases = json.load(fp)
        for key in testcases.keys():
            testcases[key]["map_name"] = key
        return testcases
    
def get_milestone_paths(milestone):
    path_name = f"{milestone['map_name']}_seed{milestone['seed'][0]}_start_{milestone['start'][0]},{milestone['start'][1]}_goal_{milestone['goal'][0]},{milestone['goal'][1]}.txt"

    subgoals = {}
    with open(os.path.join(TESTCASE_PATH, "milestone2_paths", path_name)) as fp:
        subgoals_strings = fp.read().split("\n")

        intention_list = []
        state_list = []
        for s in subgoals_strings:
            if len(s) > 0:
                intention = s.split(" ")[2].strip(" ")
                intention_list.append(intention)

                state = (int(s.split(',')[0][1:]), int(s.split(',')[1].split(')')[0].strip(" ")))
                state_list.append(state)

        for idx, s in enumerate(state_list[:-1]):
            subgoals[s] = intention_list[idx+1] 

    return subgoals