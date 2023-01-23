import os
import json
import subprocess

import numpy as np

def check_answers(answer_dir = 'data/answers', pred_dir = 'data/predictions', test_range = None):
    """
    Checks the answers in the answer_dir against the predictions in the pred_dir.
    """
    
    answer_files = os.listdir(answer_dir)
    pred_files = os.listdir(pred_dir)
    answer_files.sort()
    pred_files.sort()

    if(len(answer_files) != len(pred_files)):
        print("Number of answer files and prediction files do not match!")
        return

    file_tuples = list(zip(answer_files, pred_files))
    for i in range(len(file_tuples) if test_range == None else test_range+1):
        # subprocess.run(f'python3 main.py --case {i+1}', capture_output=True)
        # print(f"Case {i+1}: ", end = '')

        with open(os.path.join(answer_dir, file_tuples[i][0]), 'r') as f:
            answer = json.load(f)

        with open(os.path.join(pred_dir, file_tuples[i][1]), 'r') as f:
            pred = json.load(f)

        keylist = answer.keys()
        answers = np.array([answer[key] for key in keylist])
        preds = np.array([pred[key] for key in keylist])
        
        match = [np.allclose(answers[j], preds[j]) for j in range(len(keylist))]
        if(np.all(match)):
            print(f"{True}")
        else:
            print(f"{False}")
            print(f"Answer: {answer}")
            print(f"Prediction: {pred}")

if __name__ == '__main__':
    check_answers()