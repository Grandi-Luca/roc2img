# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class MultiINPUTShardDescriptor():
    """MultiINPUT Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            adni_num: str = '',
            data_dir: str = '',
            img_dir: str = '',
            csv_path: str = 'ADNI_csv',
            csv_filename: str = 'ADNI_ready.csv',
            **kwargs

    ) -> None:
        """Initialize MultiINPUTShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.adni_num = adni_num
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.csv_filename = csv_filename

        adni_train, adni_val = self.load_data()
        self.data_by_type = {
            'train': adni_train,
            'val': adni_val
        }


    def load_data(self) -> Tuple[Any, Any]:
        """Download prepared dataset."""

        # Load tabular data from csv file
        csv_file = os.path.join(self.csv_path, self.csv_filename)

        if not csv_file:
            logger.info(f"Dataset {self.csv_filename} not found at:{self.csv_path}.\n\t")
            logger.info(f"Aborting.")
            exit

        adni_tabular = pd.read_csv(csv_file)
        adni_tabular = adni_tabular[adni_tabular['SRC']==f"ADNI{self.adni_num}"]

        # Load img paths and details from stored dataframe
        img_df_filename = f"adni{self.adni_num}_paths.pkl"
        img_df_file = os.path.join(self.data_dir, img_df_filename)
        adni_imgs = pd.read_pickle(img_df_file)

        # Combine dataframes adni_tabular and adni_images
        adni = pd.merge( left=adni_imgs, right=adni_tabular, how="inner", on="PTID",
                            suffixes=("_x", "_y"),copy=False, indicator=False, validate="one_to_one")


        # Datasplit
        labels = adni['labels'].tolist()
        train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                              shuffle=True, stratify=labels)

        adni_train = adni.iloc[train_idx]
        adni_val = adni.iloc[val_idx]

        return adni_train, adni_val
