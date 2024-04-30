
## Need to put validation split here










# def validation_split(path_per_idx, labels, subset=None, validation_split=None):

#     df = pd.DataFrame()
#     filenames = []
#     labels = []
    
#     # for filepath in path_per_idx.items():
#     #     filenames.append(filepath)
#     #     labels.append(label)

#     df["filenames"] = filenames
#     df["labels"] = labels
    
#     if subset == "train":
#         split_indexes = int(len(df) * validation_split)
#         train_df = df[split_indexes:]
#         val_df = df[:split_indexes]
#         return train_df, val_df

#     return df

# train_df, val_df = data_to_df(train_dir, subset="train", validation_split=0.2)

