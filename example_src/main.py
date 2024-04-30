from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
import keras
import tensorflow as tf

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

def predict_chain(chain_dir: Path, model, range_df):
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
        #_reference_img = cv2.imread(str(reference_img_path))
        _reference_img = tf.image.resize(tf.image.decode_png(tf.io.read_file(str(reference_img_path)), channels=3), (512,640))
        #_reference_img = keras.utils.load_img(str(reference_img_path), target_size=(512,640))
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
            #_other_image = cv2.imread(str(image_path))
            _other_image = tf.image.resize(tf.image.decode_png(tf.io.read_file(str(image_path)), channels=3), (512,640))
            #_other_image = keras.utils.load_img(str(image_path), target_size=(512,640))
            range = range_df.loc[(chain_id, i), :]
            predicted_values = model.predict([np.asarray([_reference_img]), np.asarray([_other_image]), np.asarray(range)])
            predicted_values_df = pd.DataFrame(predicted_values, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
            predicted_values_df[["qw", "qx", "qy", "qz"]] = [0,0,0,0]
            predicted_values = np.asarray(predicted_values_df)
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

    range_path = data_dir / "range.csv"
    range_df = pd.read_csv(range_path, index_col=INDEX_COLS)
    range_df = range_df.ffill()
    range_df = range_df.fillna(range_df.mean()) # ensure all nan vals are filled

    # load trained model
    model = keras.models.load_model('./test_model-logcosh-c1x2-16x3-3e-000005-larger-kernels-7500-depth.keras')

    image_dir = data_dir / "images"
    chain_ids = submission_format_df.index.get_level_values(0).unique()

    for chain_id in chain_ids:
        logger.info(f"Processing chain: {chain_id}")
        chain_dir = image_dir / chain_id
        assert chain_dir.exists(), f"Chain directory does not exist: {chain_dir}"
        chain_df = predict_chain(chain_dir, model, range_df)
        submission_df.loc[chain_id] = chain_df.values

    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()