from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
import math
from scipy.spatial.transform import Rotation

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

CAMERA_MATRIX = np.array([[5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
                          [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
                          [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

def rotationMatrixToQuaternion(R):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    return qw, qx, qy, qz


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

    sift = None
    base_image = None
    keypoint_base = None
    destination_base = None
    bf_feature_matcher = None

    # make a prediction for each image
    for i, image_path in path_per_idx.items():
        if i == 0:
            predicted_values = REFERENCE_VALUES
            # sift = cv2.SIFT_create()
            sift = cv2.SIFT_create()
            orb = cv2.ORB_create()

            base_image = cv2.imread(str(image_path))
            base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            keypoint_base, destination_base = sift.detectAndCompute(base_image_gray, None)
            orb_kp1, orb_des1 = orb.detectAndCompute(base_image_gray, None)
            bf_feature_matcher = cv2.BFMatcher()
        else:
            target_image = cv2.imread(str(image_path))
            target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            keypoint_target, destination_target = sift.detectAndCompute(target_image_gray, None)
            orb_kp2, orb_des2 = orb.detectAndCompute(target_image_gray, None)
            matches = []

            if destination_target is None or orb_des2 is None:
                print('empty destiation target image....')
                predicted_values = np.random.rand(len(PREDICTION_COLS))
            else:
                try:
                    # SIFT LOGIC
                    matches = bf_feature_matcher.knnMatch(destination_base, destination_target, k=2)
                    best_matches = [m for m, _ in matches]
                    relative_points = np.float32([keypoint_base[m.queryIdx].pt for m in best_matches])
                    target_points = np.float32([keypoint_target[m.trainIdx].pt for m in best_matches])
                    H, _ = cv2.findHomography(relative_points, target_points, cv2.RANSAC, 5.0)

                    # ORB LOGIC
                            # Match keypoints between the two images
                    orb_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
                    orb_matches = orb_matcher.match(orb_des1, orb_des2, None)
                    
                    # Extract matched keypoints
                    points1 = np.float32([orb_kp1[m.queryIdx].pt for m in orb_matches]).reshape(-1, 1, 2)
                    points2 = np.float32([orb_kp2[m.trainIdx].pt for m in orb_matches]).reshape(-1, 1, 2)

                    E, _ = cv2.findEssentialMat(points1, points2, CAMERA_MATRIX)
                    _, R, t, _ = cv2.recoverPose(E, points1, points2, CAMERA_MATRIX)
                    # translation = np.append(t[:2], 0) * 2000


                    if H is not None:
                        inf_mask = np.isinf(H)
                        nan_mask = np.isnan(H)
                        if np.any(inf_mask) or np.any(nan_mask):
                            predicted_values = np.random.rand(len(PREDICTION_COLS))
                        else:
                            qw, qx, qy, qz = Rotation.from_matrix(H).as_quat()
                            x,y,z = t.ravel()
                            predicted_values = np.array([x, y, z, qw, qx, qy, qz])

                            # h,w,_ = base_image.shape
                            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                            # dst = cv2.perspectiveTransform(pts,H)
                            # print()

                            # print(f'Trans: {[x,y,z]}')
                            # print(f'Quat: {[round(x, 4) for x in [qw, qx, qy, qz]]}')
                            # print()
                            # for first (1)
                            # translation 
                            # 2.21703339,23.38332176,12.38890553
                            # quaternion
                            # 0.99002814,-0.0722533,0.07088085,-0.09797749

                    else:
                        predicted_values = np.random.rand(len(PREDICTION_COLS))
                except Exception as e:
                     predicted_values = np.random.rand(len(PREDICTION_COLS))
                     print(e)
                    
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

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()

    image_dir = data_dir / "images"
    chain_ids = submission_format_df.index.get_level_values(0).unique()
    for chain_id in chain_ids:
        logger.info(f"Processing chain: {chain_id}")
        chain_dir = image_dir / chain_id
        assert chain_dir.exists(), f"Chain directory does not exist: {chain_dir}"
        chain_df = predict_chain(chain_dir)
        submission_df.loc[chain_id] = chain_df.values

    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
