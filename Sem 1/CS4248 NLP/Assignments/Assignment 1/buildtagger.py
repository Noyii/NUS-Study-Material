# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import json

from collections import defaultdict, Counter

# Global constants
K = 1 # Laplace K-Smoothing constant

# Probability P(word) or P(tag) in the entire training corpus
def calculate_unigram_probability(counter, total_count, distinct_count):
    for i in counter.keys():
        # Smooth each probability by add-one and discounting (normalization)
        # Convert to log-space to avoid underflow
        counter[i] = math.log((counter[i] + 1*K) / (total_count + distinct_count*K))


# Probability matrix P[tag][word] or P[tag][tag]
def calculate_probability_matrix(matrix):
    for counter in matrix.values():
        total = sum(counter.values())
        
        for key in counter.keys():
            # Smooth each probability by add-one and discounting (normalization)
            # Convert to log-space to avoid underflow
            counter[key] = math.log((counter[key] + 1*K) / (total + len(counter)*K))


# Write training statistics to model file
def write_to_file(file, data):
    file.write(json.dumps(data))
    file.write('\n')


def train_model(train_file, model_file):
    # Write your code here. You can add functions as well.
    
    vocabulary = set() # Set of all words in training corpus
    states = set() # Set of all states/tags in Penn TreeBank
    total_tag_count = 0
    
    unigram_tag_probability = Counter() # P(tag)
    transition_probability = defaultdict(Counter) # P(tag|previous_tag) or P(State N | State N-1)
    emission_probability = defaultdict(Counter) # P(word|tag) or P(word|State)

    # Parse training data
    with open(train_file, 'r') as training:
        for line in training.readlines():
            line = line.strip().split(" ")
            pre_tag = "RDM" # Random word
            word, tag = None, None

            for entry in line:
                if entry.count('/') > 1: # E.g., 1/2-year/CD
                    idx = entry[::-1].find('/') # idx = 2
                    word = entry[: -idx-1] # word = 1/2-year
                    tag = entry[-idx:] # tag = CD
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

    calculate_unigram_probability(unigram_tag_probability, total_tag_count, distinct_tag_count) # P(tag)
    calculate_probability_matrix(transition_probability) # P(State N | State N-1)
    calculate_probability_matrix(emission_probability) # P(word|State)

    # Write training statistics to model file
    with open(model_file, 'w') as model:
        print("Writing vocabulary length")
        write_to_file(model, distinct_word_count)

        print("Writing states")
        model.write(str(states))
        model.write('\n')

        print("Writing unigram tag probabilities")
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