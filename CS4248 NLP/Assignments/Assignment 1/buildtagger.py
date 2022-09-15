# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import json

from collections import defaultdict, Counter

# Global constants
K = 1

def calculate_unigram_probability(counter, total_count, distinct_count):
    for i in counter.keys():
        counter[i] = math.log((counter[i] + 1*K) / (total_count + distinct_count*K))


def calculate_probability_matrix(matrix):
    for counter in matrix.values():
        total = sum(counter.values())
        
        for key in counter.keys():
            counter[key] = math.log((counter[key] + 1*K) / (total + len(counter)*K))


def write_to_file(file, data):
    file.write(json.dumps(data))
    file.write('\n')


def train_model(train_file, model_file):
    # Write your code here. You can add functions as well.
    vocabulary = set()
    states = set()
    total_tag_count = 0
    
    unigram_tag_probability = Counter()
    transition_probability = defaultdict(Counter)
    emission_probability = defaultdict(Counter)

    with open(train_file, 'r') as training:
        for line in training.readlines():
            line = line.strip().split(" ")
            pre_tag = "RDM" # Random word
            word, tag = None, None

            for entry in line:
                if entry.count('/') > 1: # E.g., 1/2-year/CD
                    idx = entry[::-1].find('/')
                    word = entry[: -idx-1]
                    tag = entry[-idx:]
                else:
                    word, tag = entry.split('/')

                vocabulary.add(word)
                states.add(tag)

                unigram_tag_probability[tag] += 1
                transition_probability[pre_tag][tag] += 1
                emission_probability[tag][word] += 1

                pre_tag = tag
                total_tag_count += 1

    distinct_tag_count = len(states) # 45 for Penn TreeBank
    distinct_word_count = len(vocabulary)

    calculate_unigram_probability(unigram_tag_probability, total_tag_count, distinct_tag_count)
    calculate_probability_matrix(transition_probability)
    calculate_probability_matrix(emission_probability)
   
    # emission_probability['RDM'] = {'RDM': math.log(1.0)}

    with open(model_file, 'w') as model:
        print("Writing vocabulary")
        write_to_file(model, distinct_word_count)

        print("Writing states")
        model.write(str(states))
        model.write('\n')

        print("Writing initial state probabilities")
        write_to_file(model, unigram_tag_probability)

        print("Writing transition probabilities")
        write_to_file(model, transition_probability)

        print("Writing emission probabilities")
        write_to_file(model, emission_probability)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)