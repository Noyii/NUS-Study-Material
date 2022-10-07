import os
import re
import sys
import string
import argparse
import datetime

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        def __file_to_list(file_path):
            with open(file_path) as data:
                output = []
                for line in data:
                    output.append(line.rstrip())

            return output

        self.texts = __file_to_list(text_path)
        if label_path is not None:
            self.labels = __file_to_list(label_path)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            count = 0
            self.bigram_matrix = []

            for text in self.texts:
                characters = ['$'] + [c for c in text] + ['$']
                bigram_list = []
                
                for i in range(len(characters)-1):
                    bigram = characters[i] + characters[i+1]
                    bigram_list.append(bigram)

                    if bigram not in self.vocab.keys():
                        self.vocab[bigram] = count
                        count += 1

                self.bigram_matrix.append(bigram_list)
        
        self.embedding_matrix = []
        for i, bigrams in enumerate(self.bigram_matrix):
            feature = []
            counter = Counter(bigrams)
            for key in self.vocab.keys():
                feature.append(counter[key])
            
            self.embedding_matrix.append(feature)


    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = self.vocab.__len__()
        classes = Counter(self.labels)
        num_class = classes.__len__()

        return num_vocab, num_class


    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self)


    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        label = self.labels[i]
        compact_embeddings = list(zip(*self.embedding_matrix))
        bigrams = self.bigram_matrix[i]
        feature = []

        for bigram in bigrams:
            x = compact_embeddings[self.vocab[bigram]]
            feature.append(x)

        h = []
        d = len(feature[0])
        compact_feature = list(zip(*feature))

        for j in range(d):
            average = sum(compact_feature[j]) / len(bigrams)
            h.append(average)
        
        text = h
        
        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer with dimension 16 and
    a feed-forward layers that reduce the dimension from 16 to 200 with ReLU activation
    a dropout layer, and a feed-forward layers that reduce the dimension from 200 to num_class
    """
    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        # define your model here
        self.vocab_size = num_vocab

        self.embedding = nn.Linear(16, 200)
        self.out = nn.Linear(200, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # define the forward function here
        out = self.embedding(x)
        h1 = self.relu(out)
        out = self.dropout(h1)
        h2 = self.out(out)

        output = torch.softmax(h2)

        return output


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    texts = None
    labels = None
    
    return texts, labels


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation
            predicted_labels = model.forward(texts)

            # do loss calculation
            loss = criterion(predicted_labels, labels)

            # do backward propagation
            loss.backward()

            # do parameter optimization step
            optimizer.step()

            # calculate running loss value for non padding
            running_loss = loss-padd

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = None
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        
        # you may change these hyper-parameters
        learning_rate = None
        batch_size = None
        num_epochs = None

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # create the test dataset object using LangDataset class


        # initialize and load the model


        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        lang_map = None

        # run the prediction
        preds = test(model, dataset, lang_map, device)
        
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))
    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)