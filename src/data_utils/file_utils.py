
# def write_failure_log(image_path, msg):
#     output_file = os.path.join(output_path, PREPROCESSING_LOG)
#     with open(PREPROCESSING_LOG, 'a') as f:
#         f.write(str(image_path) + msg + "\n")
#     f.close()
#     return


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
