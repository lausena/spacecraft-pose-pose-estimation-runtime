import pandas as pd
from pathlib import Path

from loguru import logger

INDEX_COLS = ["chain_id", "i"]

def create_path_dict(data_dir, output_path):
    
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()

    assert (output_path.parent.exists()), f"Expected output directory {output_path.parent} does not exist"

    logger.info(f"using data dir: {data_dir}")
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    image_dir = data_dir / "images"

    path_dict = {
        "data_dir": data_dir,
        "output_path": output_path,
        "image_dir": image_dir        
    }

    return path_dict

def load_dataset_dfs(data_dir):

    # read in the submission format
    submission_format_path = data_dir / "submission_format.csv"
    submission_format_df = pd.read_csv(submission_format_path, index_col=INDEX_COLS)
    
    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()
    
    # read in the train labels
    train_labels_path = data_dir / "train_labels.csv"
    train_labels_df = pd.read_csv(train_labels_path, index_col=INDEX_COLS)
    
    range_path = data_dir / "range.csv"
    range_df = pd.read_csv(range_path, index_col=INDEX_COLS)
    range_df = interpolate_empty_ranges(range_df)
    range_df.to_csv("output.csv")

    input_df_dict = {
        "submission_format": submission_format_df,
        "submission": submission_df,
        "train_labels": train_labels_df,
        "range_df": range_df
    }

    return input_df_dict

def interpolate_empty_ranges(range_df):
    range_df = range_df.interpolate(method='linear')
    return range_df