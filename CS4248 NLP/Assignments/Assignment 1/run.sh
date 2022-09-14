#!/bin/zsh

python3 buildtagger.py sents.train model-file && \
python3 runtagger.py sents.test model-file sents.out && \
python3 eval.py sents.out sents.answer
