from ADNI.MultiINPUT_shard_descriptor import MultiINPUTShardDescriptor
from rocket_img import ROCKET
from distributions import DistributionType
from utils import ConvolutionType, FeatureType, DilationType

import time
import os
from typing import Optional

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import nibabel as nib

import wandb

def load_nifti_to_tensor(file_paths):
    """Load multiple NIfTI files and stack them into a single tensor."""
    tensors = []
    for path in file_paths:
        img = nib.load(path)
        data = img.get_fdata()
        tensor = torch.tensor(data, dtype=torch.float32)
        tensors.append(tensor)
    return torch.stack(tensors)

def load_adni_data(
        rank_worldsize:str,
        adni_num:int,
        data_dir:str,
        img_dir:str,
        csv_path:str,
        csv_filename:str,
        data_seed:int,
    ):
     # Load dataset
    descriptor = MultiINPUTShardDescriptor(
        rank_worldsize= rank_worldsize,
        adni_num= adni_num,
        data_dir= data_dir,
        img_dir= img_dir,
        csv_path= csv_path,
        csv_filename= csv_filename,
        data_seed= data_seed,
    )

    # train e test contengono i path per le immagini da usare come input e le labels associate
    train, test = descriptor.data_by_type['train'], descriptor.data_by_type['val']
    
    base_dir = descriptor.data_dir 
    duplicate_prefix = descriptor.data_dir.split('/')[-1]  # prefisso da rimuovere se presente nei path

    train_paths = [
        os.path.join(base_dir, path.split(f'{duplicate_prefix}/', 1)[-1]) if path.startswith(f'{duplicate_prefix}/') else os.path.join(base_dir, path)
        for path in train['IMG_PATH_NORM_min-max'].tolist()
    ]
    test_paths = [
        os.path.join(base_dir, path.split(f'{duplicate_prefix}/', 1)[-1]) if path.startswith(f'{duplicate_prefix}/') else os.path.join(base_dir, path)
        for path in test['IMG_PATH_NORM_min-max'].tolist()
    ]
    
    # Carica tutte le immagini in un unico tensore
    X_train = load_nifti_to_tensor(train_paths)
    y_train = np.array(train['labels'].tolist())
    
    X_test = load_nifti_to_tensor(test_paths)
    y_test = np.array(test['labels'].tolist())

    return X_train, y_train, X_test, y_test


def measure_performance(
    model,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Optional[float]]:
    """Measure the performance of a model on training and testing datasets.
    """

    model.fit(X_train)

    # Tempo di trasformazione (feature extraction)
    start = time.time()
    X_train_transformed = model.transform(X_train)
    transform_time_train = time.time() - start
    start = time.time()
    X_test_transformed = model.transform(X_test)
    transform_time_test = time.time() - start

    # Tempo di training ridge regression
    start = time.time()
    clf = RidgeClassifier(alpha=0.1).fit(X_train_transformed, y_train)
    training_time = time.time() - start

    # Accuracy ridge regression
    predictions = clf.predict(X_test_transformed)
    acc = accuracy_score(y_test, predictions)

    return {
        'transform_time_train': round(transform_time_train, 3),
        'transform_time_test': round(transform_time_test, 3),
        'training_time': round(training_time, 3),
        'accuracy': acc,
    }


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_adni_data(
        rank_worldsize= '1, 1',
        adni_num= 1,
        data_dir= 'path/to/ADNI/data',
        img_dir= 'ADNI1_ALL_T1',
        csv_path= 'path/to/ADNI/csv',
        csv_filename= 'ADNI_ready.csv',
        data_seed= 13
    )
    
    rocket = ROCKET(
        cout=1000,
        distr_pair=(DistributionType.GAUSSIAN_01, DistributionType.UNIFORM),
        dilation=DilationType.UNIFORM_ROCKET,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    distributions = [
        (DistributionType.GAUSSIAN_01, DistributionType.UNIFORM)
    ]

    convs_type = [ConvolutionType.STANDARD, ConvolutionType.DEPTHWISE_SEP, ConvolutionType.SPATIAL]

    lengths = [[3], [5], [7], [9], [11], [5,7,9], [7,9,11]]

    for conv_type in convs_type:
        rocket.convolution_type = conv_type
        
        for d_pair in distributions:
            rocket.distr_pair = d_pair
                
            for feature_set in [[FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV]]:
                rocket.features_to_extract = feature_set
                
                for length_set in lengths:
                    rocket.candidate_lengths = length_set
                    
                    for seed in [0,1,42,7,21]:

                        rocket.random_state = seed

                        run = wandb.init(
                            project="rocket2img-ADNI-min_max",
                            entity="luca-gr",
                            config={
                                **rocket.get_params(),
                            }
                        )

                        metrics = measure_performance(
                            rocket, X_train, X_test, y_train, y_test)
                        
                        run.log(metrics)

                        run.finish()
    