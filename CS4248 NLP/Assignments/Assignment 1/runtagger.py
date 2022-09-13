# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import ast
import sys
import datetime
import json

PROBABILITY = "probability"
BACKPOINTER = "backpointer"


def viterbi(observation, vocabulary, states, \
initial_state_probability, emission_probability, transition_probability, \
word_tag_frequency):

    distinct_tag_count = len(states)
    distinct_word_count = len(vocabulary)

    unknown_word_prob = math.log(1/distinct_word_count)
    unknown_state_prob = math.log(1/distinct_tag_count)
    
    V = [{}]

    for state in states:
        V[0][state] = {
            PROBABILITY: initial_state_probability[state] + \
                                emission_probability[state].get(observation[0], unknown_word_prob),
            BACKPOINTER: None
        }
    
    for t in range(1, len(observation)):
        V.append({})

        for state in states:
            probs = [(V[t-1][prev_state][PROBABILITY] + \
                        transition_probability[prev_state].get(state, unknown_state_prob)) \
                            for prev_state in states]
            
            max_transition_prob = max(probs)
            max_emission_prob = max_transition_prob + emission_probability[state].get(observation[t], unknown_word_prob)
            
            for prev_state in states:
                transition_prob = V[t-1][prev_state][PROBABILITY] + transition_probability[prev_state].get(state, unknown_state_prob)
                if (max_transition_prob == transition_prob):
                        V[t][state] = {PROBABILITY: max_emission_prob, BACKPOINTER: prev_state}

    pre_tag = "RDM"
    output = ""
    tags = []

    max_probability = max(stage[PROBABILITY] for stage in V[-1].values())
    for state, stage in V[-1].items():
        if stage[PROBABILITY] == max_probability:
            tags.append(state)
            pre_tag = state
            break

    for t in range(len(V) - 2, -1, -1):
        x = V[t+1][pre_tag][BACKPOINTER]
        tags.append(x)
        pre_tag = x

    tags = tags[::-1]

    for i in range(len(observation)):
        output += observation[i] + '/' + tags[i] + " " 

    return output[:-1]


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    result = []

    print("Reading model")
    with open(model_file, 'r') as model:
        lines = model.readlines()
        
        vocabulary = ast.literal_eval(lines[0])
        states = ast.literal_eval(lines[1])
        initial_state_probability = json.loads(lines[2])
        transition_probability = json.loads(lines[3])
        emission_probability = json.loads(lines[4])
        word_tag_frequency = json.loads(lines[5])

    print("Performing Viterbi algorithm")
    with open(test_file, 'r') as test:
        for line in test.readlines():
            observation = line.strip().split(" ")
            output = viterbi(observation, vocabulary, states, initial_state_probability, emission_probability, transition_probability, word_tag_frequency)
            result.append(output)

    print("Writing answer in out_file")
    with open(out_file, 'w') as answer:
        for line in result:
            answer.write(line)
            answer.write('\n')

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
