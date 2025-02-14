# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

from ast import Return
import os
import math
import sys
import datetime
import json

from collections import defaultdict, Counter

K = 1
UNKNOWN = "UNK"

def train_model(train_file, model_file):
    # Write your code here. You can add functions as well.
    vocabulary = set()
    states = set()
    total_tag_count = 0

    initial_tag_probabilities = Counter()
    transition_probability = defaultdict(Counter)
    emission_probability = defaultdict(Counter)
    word_tag_frequency = defaultdict(Counter)

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

                initial_tag_probabilities[tag] += 1
                transition_probability[pre_tag][tag] += 1
                # emission_probability[tag][word] += 1
                word_tag_frequency[word][tag] += 1

                pre_tag = tag
                total_tag_count += 1

    # Handle unknown words by changing all words with freq = 1 to 'UNK'
    new_word_tag_freq = defaultdict(Counter)
    new_word_tag_freq[UNKNOWN]['RDM'] = 1
    vocabulary.add(UNKNOWN)

    for word, tags in word_tag_frequency.items():
        total = sum(tags.values())
        
        if total <= 1:
            new_word_tag_freq[UNKNOWN] += tags
            vocabulary.remove(word)
        else:
            new_word_tag_freq[word] = tags

    word_tag_frequency = new_word_tag_freq

    for word, tags in word_tag_frequency.items():
        for tag, value in tags.items():
            emission_probability[tag][word] += value
    
    print(sum(word_tag_frequency[UNKNOWN].values()))
    return
    
    # Calculate all probabilities
    distinct_tag_count = len(states) # 45 for Penn TreeBank
    distinct_word_count = len(vocabulary)

    for i in initial_tag_probabilities.keys():
        initial_tag_probabilities[i] = math.log((initial_tag_probabilities[i] + 1*K) / (total_tag_count + distinct_tag_count*K))

    for tag_transition_counter in transition_probability.values():
        total = sum(tag_transition_counter.values())
        for tag in tag_transition_counter.keys():
            tag_transition_counter[tag] = math.log((tag_transition_counter[tag] + 1*K) / (total + len(tag_transition_counter)*K))

    for word_emission_counter in emission_probability.values():
        total = sum(word_emission_counter.values())
        for word in word_emission_counter.keys():
            word_emission_counter[word] = math.log((word_emission_counter[word] + 1*K) / (total + len(word_emission_counter)*K))
        
    emission_probability['RDM'] = {'RDM': math.log(1.0)}

    # print(emission_probability)
    # return

    with open(model_file, 'w') as model:
        print("Writing vocabulary")
        model.write(str(vocabulary))
        model.write('\n')

        print("Writing states")
        model.write(str(states))
        model.write('\n')

        print("Writing initial state probabilities")
        model.write(json.dumps(initial_tag_probabilities))
        model.write('\n')

        print("Writing transition probabilities")
        model.write(json.dumps(transition_probability))
        model.write('\n')

        print("Writing emission probabilities")
        model.write(json.dumps(emission_probability))
        model.write('\n')

        print("Writing word-tag frequencies")
        model.write(json.dumps(word_tag_frequency))
        model.write('\n')

        print("Writing distinct states count")
        model.write(json.dumps(distinct_tag_count))
        model.write('\n')

        print("Writing distinct word counts")
        model.write(json.dumps(distinct_word_count))
        model.write('\n')

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
