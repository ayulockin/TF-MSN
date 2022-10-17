import glob
import json
import os

import pandas as pd
import wandb
from tqdm import tqdm


def download_dataset(
    dataset_name: str, dataset_type: str, version: str = "latest", save_at="artifacts/"
):
    """
    Utility function to download the data saved as W&B artifacts and return a dataframe
    with path to the dataset and associated label.

    Args:
        dataset_name (str): The name of the dataset - `train`, `val`, `test`, `out-class`, and `in-class`.
        dataset_type (str): The type of the dataset - `labelled-dataset`, `unlabelled-dataset`.
        version (str): The version of the dataset to be downloaded. By default it's `latest`,
            but you can provide different version as `vX`, where, X can be 0,1,...

        Note that the following combination of dataset_name and dataset_type are valid:
            - `train`, `labelled-dataset`
            - `val`, `labelled-dataset`
            - `test`, `labelled-dataset`
            - `in-class`, `unlabelled-dataset`
            - `out-class`, `unlabelled-dataset`

    Return:
        df_data (pandas.DataFrame): Dataframe with path to images with associated labels if present.
    """
    if dataset_name == "train" and os.path.exists(save_at + "train.csv"):
        data_df = pd.read_csv(save_at + "train.csv")
    elif dataset_name == "val" and os.path.exists(save_at + "valid.csv"):
        data_df = pd.read_csv(save_at + "valid.csv")
    elif dataset_name == "test" and os.path.exists(save_at + "test.csv"):
        data_df = pd.read_csv(save_at + "test.csv")
    # TODO (ayulockin): unlabelled dataset
    else:
        data_df = None
        print("Downloading dataset...")

    if data_df is None:
        # Download the dataset.
        wandb_api = wandb.Api()
        artifact = wandb_api.artifact(
            f"ayush-thakur/ssl-study-data/{dataset_name}:{version}", type=dataset_type
        )
        artifact_dir = artifact.download()

        # Open the W&B table downloaded as a json file.
        json_file = glob.glob(artifact_dir + "/*.json")
        assert len(json_file) == 1
        with open(json_file[0]) as f:
            data = json.loads(f.read())
            assert data["_type"] == "table"
            columns = data["columns"]
            data = data["data"]

        # Create a dataframe with path and label
        df_columns = ["image_id", "image_path", "width", "height"]
        if "label" in columns:
            df_columns += ["label"]
        data_df = pd.DataFrame(columns=df_columns)

        for idx, example in tqdm(enumerate(data)):
            image_id = int(example[0])
            image_dict = example[1]
            image_path = os.path.join(artifact_dir, image_dict.get("path"))
            height = image_dict.get("height")
            width = image_dict.get("width")

            df_data = [image_id, image_path, width, height]
            if "label" in columns:
                df_data += [example[2]]
            data_df.loc[idx] = df_data

    # Shuffle the dataframe
    if dataset_name == "train":
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the dataframes as csv
    if dataset_name == "train" and not os.path.exists(save_at + "train.csv"):
        data_df.to_csv(save_at + "train.csv", index=False)

    if dataset_name == "val" and not os.path.exists(save_at + "valid.csv"):
        data_df.to_csv(save_at + "valid.csv", index=False)

    if dataset_name == "test" and not os.path.exists(save_at + "test.csv"):
        data_df.to_csv(save_at + "test.csv", index=False)
    # TODO (ayulockin): unlabelled dataset

    return data_df


def preprocess_dataframe(df):
    # TODO (ayulockin): take care of df without labels.
    # Remove unnecessary columns
    df = df.drop(["image_id", "width", "height"], axis=1)
    assert len(df.columns) == 2

    # Fix types
    df[["label"]] = df[["label"]].apply(pd.to_numeric)

    image_paths = df.image_path.values
    labels = df.label.values

    return image_paths, labels
