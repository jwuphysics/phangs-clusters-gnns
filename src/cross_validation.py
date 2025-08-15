import numpy as np
import json

def create_stratified_k_folds_by_distance(galaxy_metadata, k=5, seed=42):
    np.random.seed(seed)
    galaxies = galaxy_metadata.copy()
    galaxies = galaxies.sample(frac=1, replace=False)
    galaxies = galaxies.sort_values("D", ascending=True)

    folds = [[] for _ in range(k)]
    for i, galaxy_name in enumerate(galaxies.index):
        folds[i % k].append(galaxy_name)

    cv_splits = []
    for i in range(k):
        valid_galaxies = folds[i]
        train_galaxies = [g for fold in folds if fold is not valid_galaxies for g in fold]
        cv_splits.append({'train': train_galaxies, 'valid': valid_galaxies})

    return cv_splits

def create_random_k_folds(galaxy_metadata, k=5, seed=42):
    np.random.seed(seed)
    galaxies = list(galaxy_metadata.index)
    np.random.shuffle(galaxies)
    
    folds = [[] for _ in range(k)]
    for i, galaxy_name in enumerate(galaxies):
        folds[i % k].append(galaxy_name)
    
    cv_splits = []
    for i in range(k):
        valid_galaxies = folds[i]
        train_galaxies = [g for fold in folds if fold is not valid_galaxies for g in fold]
        cv_splits.append({'train': train_galaxies, 'valid': valid_galaxies})
    
    return cv_splits

def save_cv_splits(cv_splits, path):
    with open(path, 'w') as f:
        json.dump(cv_splits, f, indent=4)

def load_cv_splits(path):
    with open(path, 'r') as f:
        return json.load(f)
