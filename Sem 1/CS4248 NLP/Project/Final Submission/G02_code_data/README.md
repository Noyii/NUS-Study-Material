## Grammatical Error Correction

### Improved ESC-CNN

- Follow the instructions given in the jupyter notebook

- Replace `run.py` after cloning the original ESC repository with the `run.py` given in `esc/run.py`.

### Genetic Algorithm

- Follow the instructions in `readme.txt`.

### Ensemble ESC-CNN-BART

- The code to pretrain and fine-tune BART-GEC is hosted on OneDrive: https://nusu-my.sharepoint.com/:f:/g/personal/e0954756_u_nus_edu/EtObzCnMhUtFpH1RnFyq74QBBb1yb1H6rp9dXQaa-lOmVw?e=Hhh6q6

- Download the folder. Pre-trained model is already present in `model/gec_bart/` as `checkpoint_best.pt`.
Run:

```shell
$ cd BART-GEC

# arg1 = output folder
# arg2 = gpu id
# arg3 = model folder
$ sh translate.sh 'output/' '0' 'model/gec_bart/'
```

- Output of above command is stored in `BART.txt` in Ensemble ESC-CNN-BART folder for convenience. This is same as `BART-GEC/output/hyp.txt`. To check difference in predictions with groundtruth, run:
```shell
$ python check_predictions_diff.py
```

- To check score:
```shell
$ cd m2scorer
$ python m2scorer ../BART.txt ../official-2014.combined.m2
```
