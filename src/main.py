from pathlib import Path

import os
import click
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from preprocessing.blur_removal import detect_blur
from preprocessing.kernel_functions import apply_kernel
from preprocessing.center_of_mass import find_cm
# from Utility.detect_object import detect, draw_rects_around_objects


INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

data_dir = r"D:\School\Deep_Neural_Nets\Final_Project\all_data"
output_path = "D:\\School\\Deep_Neural_Nets\\Final_Project\\output"
preprocessed_output_path = "D:\\School\\Deep_Neural_Nets\\Final_Project\\preprocessing_output"

ADD_CM_Text = True

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
        else:

            image = cv2.imread(str(image_path))

            # object_found = detect(image)
            if ADD_CM_Text:
                image_cm = find_cm(image)
                cv2.circle(image, (image_cm[0], image_cm[1]), 5, (0, 0, 255), thickness=-1)
                # draw_rects_around_objects(image, object_found)
    
            # Write original image
            output_filename = create_output_filepath(str(image_path), chain_id, preprocessed_output_path, "orig")
            cv2.imwrite(output_filename, image)

            kernels = ["sharpen", "outline", "emboss", "identity", "k2", "k3"]
            for kernel_name in kernels:
                kernel_image = apply_kernel(image, kernel_name)
                output_filename = create_output_filepath(str(image_path), chain_id, preprocessed_output_path, kernel_name)

                if ADD_CM_Text:
                    try:
                        image_cm = find_cm(kernel_image)
                    except:
                        print(kernel_name + " failed for image " + output_filename)

                #     object_found = detect(image)
                #     draw_rects_around_objects(kernel_image, object_found)

                cv2.imwrite(output_filename, kernel_image)

            # TODO: actually make predictions! we don't actually do anything useful here!
            predicted_values = np.random.rand(len(PREDICTION_COLS))

        chain_df.loc[i] = predicted_values

    # double check we made predictions for each image
    assert (
        chain_df.notnull().all(axis="rows").all()
    ), f"Found NaN values for chain {chain_id}"
    assert (
        np.isfinite(chain_df.values).all().all()
    ), f"Found NaN or infinite values for chain {chain_id}"

    return chain_df

def add_cm_text():
    pass

def write_modified_images(image_path, chain_id, blur_removed_im, _other_image):

    # Write blur removed image
    output_filename = create_output_filepath(image_path, chain_id, preprocessed_output_path, "modified")
    cv2.imwrite(output_filename, blur_removed_im)

    return


def create_output_filepath(image_path, chain_id, output_dir, label=""):
    output_filename = image_path.split("\\")[-1]
    output_filename = output_filename.split(".")[0]

    output_dir = os.path.join(output_dir, chain_id)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, output_filename)

    if label:
        output_file = output_file + "_" + label + ".png"
    else:
        output_file = output_file + ".png"

    return output_file

def write_preprocessed_images():
    pass


# @click.command()
# @click.argument(
#     "data_dir",
#     type=click.Path(exists=True, file_okay=False, dir_okay=True),
# )
# @click.argument(
#     "output_path",
#     type=click.Path(exists=False),
# )
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
    main(data_dir, output_path)
