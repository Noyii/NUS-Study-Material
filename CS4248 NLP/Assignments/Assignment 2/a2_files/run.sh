#!/bin/zsh

python3 a2part2.py --train --text_path data/x_train.txt --label_path data/y_train.txt --model_path model.pt && \
python3 a2part2.py --test --text_path data/x_test.txt --model_path model.pt --output_path out.txt && \
python3 eval.py out.txt data/y_test.txt
