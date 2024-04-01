from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger

from SIFTengine import SIFTFeatureTracker

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

#Intrinsic Matrix
fx = 5.2125371e+03
fy = 6.2550444e+03
cx = 6.4000000e+02
cy = 5.1200000e+02


K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

tracker = SIFTFeatureTracker()

def predict_chain(chain_dir: Path, range_values):
    logger.debug(f"making predictions for {chain_dir}")
    chain_id = chain_dir.name
    image_paths = list(sorted(chain_dir.glob("*.png")))
    path_per_idx = {int(image_path.stem): image_path for image_path in image_paths}
    idxs = list(sorted(path_per_idx.keys()))

    assert idxs[0] == 0, f"First index for chain {chain_id} is not 0"
    assert (
        np.diff(idxs) == 1
    ).all(), f"Skipped image indexes found in chain {chain_id}"

    # pick out the reference image
    try:
        reference_img_path = path_per_idx[0]
        _reference_img = cv2.imread(str(reference_img_path), cv2.IMREAD_GRAYSCALE)
    except KeyError:
        raise ValueError(f"Could not find reference image for chain {chain_id}")

    # create an empty dataframe to populate with values
    chain_df = pd.DataFrame(
        index=pd.Index(idxs, name="i"), columns=PREDICTION_COLS, dtype=float
    )

    # make a prediction for each image
    for i, image_path in path_per_idx.items():
        if i == 0:
            predicted_values = REFERENCE_VALUES
            previous_range = range_values.loc[i]
            last_valid_R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])#tracker.visualize_poses()
            last_valid_t = np.array([1,1,1])

        else:
            _other_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            try:
                R, t = tracker.process_pair(_reference_img, _other_image, K)
            except:
                R = None
                t = None
            
            if R is not None and t is not None:
                last_valid_R = R#tracker.visualize_poses()
                last_valid_t = t
                print(range_values.loc[i].range)
                if range_values.loc[i].range != None:
                    previous_range = range_values.loc[i]
                    t = t / np.linalg.norm(t)
                    #print(range_values.loc[i].range)
                    t_scaled = t * range_values.loc[i].range * 0.1
                else:
                    t = t / np.linalg.norm(t)
                    t_scaled = t * previous_range.range * 0.1
                R = np.array(R)
                q = tracker.r_to_q(R)
                predicted_values = np.concatenate([t_scaled.T.flatten(),q])
                print(predicted_values)
                
            else:
                if range_values.loc[i].range is not None:
                    previous_range = range_values.loc[i]
                    t = last_valid_t / np.linalg.norm(last_valid_t)
                    t_scaled = t * range_values.loc[i].range * 0.1
                else:
                    t = last_valid_t / np.linalg.norm(last_valid_t)
                    t_scaled = t * previous_range.range * 0.1
                R = np.array(last_valid_R)
                q = tracker.r_to_q(R)
                predicted_values = np.concatenate([t_scaled.T.flatten(),q])
                print(predicted_values)
                #predicted_values = np.random.rand(len(PREDICTION_COLS))
            # TODO: actually make predictions! we don't actually do anything useful here!
            
        chain_df.loc[i] = predicted_values

    # double check we made predictions for each image
    assert (
        chain_df.notnull().all(axis="rows").all()
    ), f"Found NaN values for chain {chain_id}"
    assert (
        np.isfinite(chain_df.values).all().all()
    ), f"Found NaN or infinite values for chain {chain_id}"

    return chain_df


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    assert (
        output_path.parent.exists()
    ), f"Expected output directory {output_path.parent} does not exist"

    logger.info(f"using data dir: {data_dir}")
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    # read in the submission format
    submission_format_path = data_dir / "submission_format.csv"
    submission_format_df = pd.read_csv(submission_format_path, index_col=INDEX_COLS)

    #read in the range csv
    range_path = data_dir / "range.csv"
    range_df = pd.read_csv(range_path, index_col=INDEX_COLS)
    range_df.replace(to_replace=[np.nan, 'nan', 'NaN', 'NAN'], value=None, inplace=True)

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()

    image_dir = data_dir / "images"
    chain_ids = submission_format_df.index.get_level_values(0).unique()
    #for chain_id in chain_ids:
    
    for i, chain_id in enumerate(chain_ids):
        logger.info(f"Processing chain: {chain_id}")
        chain_dir = image_dir / chain_id
        assert chain_dir.exists(), f"Chain directory does not exist: {chain_dir}"
        range_values = range_df.loc[chain_id]
        print(range_values)
        chain_df = predict_chain(chain_dir, range_values)
        submission_df.loc[chain_id] = chain_df.values

    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
