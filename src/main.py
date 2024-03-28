from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
import math
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import transforms3d.quaternions as quat
from scipy.spatial.transform import Rotation


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

def rotationMatrixToQuaternion(R):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    return qw, qx, qy, qz


def form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

def decomp_essential_mat(E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = form_transf(R, t)
            # Make the projection matrix
            P_cur = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(P, P_cur, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        print(f'Relative scale: {relative_scale}')
        t = t * relative_scale
        t

        return [R1, t]


MODEL_TYPE = "DPT_Large"
MIDAS = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIDAS.to(DEVICE)
MIDAS.eval()

def estimate_depth(image_path):
    transform = Compose([
        Resize(384),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_batch = transform(img).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        prediction = MIDAS(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output_depth = prediction.cpu().numpy()
    return output_depth


def extract_and_match_features(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2

def derive_3d_points(matched_points_2d, depth_map):
    points_3d = []
    for x, y in matched_points_2d:
        depth = depth_map[int(y), int(x)]
        points_3d.append([x, y, depth])
    return np.array(points_3d)

def estimate_pose(points_3d, points_2d, camera_matrix):
    _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, None)
    return rvec, tvec

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
    for i, image_path in path_per_idx.items():
        if i == 0:
            predicted_values = REFERENCE_VALUES
            sift = cv2.SIFT_create()
            reference_image = cv2.imread(str(image_path))
            reference_image_path = image_path
            depth_map_reference = estimate_depth(reference_image_path)
            # keypoints1, descriptors1 = sift.detectAndCompute(reference_image, None)
        else:
            try:
                target_image = cv2.imread(str(image_path))
                # Apply feature matching
                matched_points_reference, matched_points_target = extract_and_match_features(reference_image, target_image)
                # Depth Estimation
                depth_map_target = estimate_depth(reference_image_path)

                matched_points_3d_target = derive_3d_points(matched_points_target, depth_map_target)
                # matched_points_3d_reference = derive_3d_points(matched_points_reference, depth_map_reference)

                # Estimate 3D Position and Orientation
                focal_length_x = 5212.5371
                focal_length_y = 6255.0444
                # focal_length_x = 1000
                # focal_length_y = 1000
                principal_point_x = target_image.shape[1] / 2
                principal_point_y = target_image.shape[0] / 2
                K = np.array([[focal_length_x, 0, principal_point_x],
                                        [0, focal_length_y,principal_point_y],
                                        [0, 0, 1]])
                
                rvec, translation_vector = estimate_pose(matched_points_3d_target, matched_points_reference, K)

                rotation = Rotation.from_rotvec(rvec.flatten())
                quaternion = rotation.as_quat()
                qw, qx, qy, qz = quaternion
                x, y, z = 0,0,0
                # x, y, z = translation_vector
                predicted_values = np.array([x, y, z, qw, qx, qy, qz])                
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
