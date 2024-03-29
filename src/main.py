from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation
import time

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

focal_length_x = 5212.5371
focal_length_y = 6255.0444
principal_point_x = 640.
principal_point_y = 512.
# camera intrinsics
K = np.array([[focal_length_x, 0, principal_point_x],
                          [0, focal_length_y,principal_point_y],
                          [0, 0, 1]])
P = np.array([[focal_length_x, 0, principal_point_x, 0],
                          [0, focal_length_y,principal_point_y, 0],
                          [0, 0, 1, 0]])

def find_homography(reference_image, target_image, model):
    keypoints_reference, descriptors_reference = model.detectAndCompute(reference_image, None)
    keypoints_target, descriptors_target = model.detectAndCompute(target_image, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_reference, descriptors_target, k=2) # 2.56
    best_matches = [m for m, _ in matches]
    src_pts = np.float32([keypoints_reference[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    # get the homography matrix H
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def decompose_homography(H, K):
    _, rotation_matrix, translation_vector, _ = cv2.decomposeHomographyMat(H, K)
    return rotation_matrix[0], translation_vector[0]


def predict_chain(chain_dir: Path):
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
        _reference_img = cv2.imread(str(reference_img_path))
    except KeyError:
        raise ValueError(f"Could not find reference image for chain {chain_id}")

    # create an empty dataframe to populate with values
    chain_df = pd.DataFrame(
        index=pd.Index(idxs, name="i"), columns=PREDICTION_COLS, dtype=float
    )

    # make a prediction for each image
    start_time = time.time()
    for i, image_path in path_per_idx.items():
        if i == 0:
            predicted_values = REFERENCE_VALUES
            reference_image = cv2.imread(str(image_path))
        else:
            try:
                # target_image_path = image_path
                target_image = cv2.imread(str(image_path))
                H = find_homography(reference_image, target_image, cv2.SIFT_create())

                if H is not None:
                    inf_mask = np.isinf(H)
                    nan_mask = np.isnan(H)
                    if np.any(inf_mask) or np.any(nan_mask):
                        logger.warning('H is invalid!!')
                        predicted_values = np.random.rand(len(PREDICTION_COLS))
                    else:
                        # print(H)
                        rotation_matrix, translation_vector = decompose_homography(H, K)
                        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
                        if np.isnan(quaternion).any():
                            logger.warning('Quat is invalid!!')
                            predicted_values = np.random.rand(len(PREDICTION_COLS))
                        else:
                            qw, qx, qy, qz = quaternion
                            x, y, z = 0,0,0
                            # x, y, z = translation_vector
                            predicted_values = np.array([x, y, z, qw, qx, qy, qz])                
            except Exception as e:
                print(e)
                predicted_values = np.random.rand(len(PREDICTION_COLS))
                    
        chain_df.loc[i] = predicted_values

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: {:.2f} seconds".format(execution_time))

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

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()

    image_dir = data_dir / "images"
    chain_ids = submission_format_df.index.get_level_values(0).unique()
    for i, chain_id in enumerate(chain_ids):
        logger.info(f"Processing chain: {chain_id}. Total {i}/{len(chain_ids)}")
        chain_dir = image_dir / chain_id
        assert chain_dir.exists(), f"Chain directory does not exist: {chain_dir}"
        chain_df = predict_chain(chain_dir)
        submission_df.loc[chain_id] = chain_df.values

    submission_df.to_csv(output_path, index=True)
    # 270 seconds


if __name__ == "__main__":
    main()
