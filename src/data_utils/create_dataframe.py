import pandas as pd

import numpy as np

def create_image_dataframe_dict(train_labels_df, range_df, image_dir):

    chain_ids = train_labels_df.index.get_level_values(0).unique()

    feature_list = []
    translation_list = []
    rotation_list = []
    for chain_id in chain_ids:
        chain_dir = image_dir / chain_id
        for image_path in chain_dir.glob("*.png"):
            feature_list, translation_list, rotation_list = append_data_list(feature_list, translation_list, rotation_list, train_labels_df, range_df, chain_id, image_path)

    feature_dataframe = pd.DataFrame(feature_list, columns=['Chain_ID', 'Image_Name', 'Image_Path', 'Base_Image_Name', 'Base_Image_Path', 'Range'])
    translation_dataframe =  pd.DataFrame(translation_list, columns=['x', 'y', 'z'])
    rotation_dataframe =  pd.DataFrame(rotation_list, columns=['qw', 'qx', 'qy', 'qz'])

    output_dataframes = {
        "features": feature_dataframe,
        "translation": translation_dataframe,
        "rotation": rotation_dataframe
    }

    return output_dataframes

def append_data_list(feature_list, translation_list, rotation_list, train_labels_df, range_df, chain_id, image_path):

    file_num = image_path.stem
    img_num = int(file_num)

    base_image_name = str(chain_id) + "_000"
    base_image_path =  image_path.parents[0] / "000.png"

    image_name = str(chain_id) + "_" + file_num

    train_label_row = train_labels_df.loc[chain_id, img_num]
    # trans_label = np.array([train_label_row['x'], train_label_row['y'], train_label_row['z']])
    # rot_label = np.array([train_label_row['qw'], train_label_row['qx'], train_label_row['qy'], train_label_row['qz']])

    range_row = range_df.loc[chain_id, img_num]
    
    feature_row = [chain_id, image_name, image_path, base_image_name, base_image_path, range_row["range"]]
    trans_label = [train_label_row['x'], train_label_row['y'], train_label_row['z']]
    rot_label = [train_label_row['qw'], train_label_row['qx'], train_label_row['qy'], train_label_row['qz']]
 
    # Append to lists
    feature_list.append(feature_row)
    translation_list.append(trans_label)
    rotation_list.append(rot_label)

    return feature_list, translation_list, rotation_list
