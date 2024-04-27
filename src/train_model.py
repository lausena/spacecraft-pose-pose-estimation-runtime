import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from pathlib import Path

import numpy as np, pandas as pd

from data_utils.create_dataframe import create_image_dataframe_dict
from data_utils.load_datasets import create_path_dict, load_dataset_dfs

from models.CNN import train_translational_model


INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


data_dir = "/workspace/School/Deep_Neural_Nets/Final_Project/all_data"
output_path = "/workspace/School/Deep_Neural_Nets/Final_Project/Working_Branches/aaron_working_2024-04-26/src"


def main(data_dir, output_path):
    
    # Retrieves paths of input files as Path objects in a dictionary
    path_dict = create_path_dict(data_dir, output_path)

    # Retrieves paths of input files as Path objects in a dictionary
    input_df_dict = load_dataset_dfs(path_dict["data_dir"])

    # Creates a output dictionary of dataframes split into features and labels
    dataframe_dict = create_image_dataframe_dict(input_df_dict["train_labels"], input_df_dict["range_df"], path_dict["image_dir"])
    print(dataframe_dict)

    trans_model = train_translational_model(dataframe_dict)

    # Save Model Here





if __name__ == "__main__":
    main(data_dir, output_path)


