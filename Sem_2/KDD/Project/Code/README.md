# CS5228_Project

Assumes kaggle authentication `kaggle.json` is present in the root directory of this repo, and that conda or miniconda is installed:

```bash
# Only tested on ubuntu
# Creates a conda env called cs5228
$ conda env create -f environment.yml

# Updating an existing env and uninstall removed deps
# If already activated:
$ conda env update -f environment.yml --prune
# else:
$ conda env update --name cs5228 -f environment.yml --prune
```

# Initialisation

Run notebook `initialisation.ipynb` first to download the relevant datasets and clean them.

# Augmentation

Run notebook `augmentation.ipynb` next to augment the cleaned dataset with supplementary data.
