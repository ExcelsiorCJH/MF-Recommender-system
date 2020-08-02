import os, io
import requests
import zipfile

import numpy as np

from collections import defaultdict
from .types_ import *


def get_data(
    data_path: str = "data/ml-20m/ratings.csv", download: bool = False,
) -> List[List[Union[int, int, float, int]]]:
    """
    Arguments
    ---------
    data_path : str
        ratings.csv dataset path
    download : bool
        download dataset or not
    
    Returns
    -------
    dataset : list of list[int, int, float, int]
        [[user_id, movie_id, rating, ts], ...]
    """
    if download or not os.path.exists("data"):
        data_url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
        dir_path = "data"

        os.mkdir(dir_path)
        r = requests.get(data_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dir_path)

    dataset = []
    for idx, line in enumerate(open(f"{data_path}", "r")):
        if idx == 0:
            # remove header
            continue
        user_id, item_id, rating, ts = line.split(",")
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)
        ts = int(ts)

        dataset.append([user_id, item_id, rating, ts])
    return dataset


def train_test_split(data: np.ndarray, data_type: str = "DS2") -> Union[np.ndarray, np.ndarray]:
    """
    dataset split method
    =====================

    Arguments
    ---------
    data : np.ndarray
        dataset to split train/test set
    data_type : str
        data type = ['DS1', 'DS2']

    Returns
    -------
    trainset : np.ndarray
    testset : np.ndarray
    """
    trainset, testset = [], []
    if data_type == "DS1":
        train_ts = [1104505203, 1230735592]
        test_ts = [1230735600, 1262271552]
        for row in data:
            user_id, item_id, rating, ts = row
            if train_ts[0] <= ts <= train_ts[1]:
                trainset.append([user_id, item_id, rating, ts])
            elif test_ts[0] <= ts <= test_ts[1]:
                testset.append([user_id, item_id, rating, ts])
    else:
        for row in data:
            user_id, item_id, rating, ts = row
            if ts < 1388502017:
                trainset.append([user_id, item_id, rating, ts])
            elif ts >= 1388502017:
                testset.append([user_id, item_id, rating, ts])

    return np.array(trainset), np.array(testset)


def get_rating_dict(dataset: List[List]) -> Union[Dict]:
    user_rating_dict = defaultdict(list)
    item_rating_dict = defaultdict(list)

    for row in dataset:
        user_id, item_id, rating, ts = row
        user_rating_dict[user_id].append(rating)
        item_rating_dict[item_id].append(rating)

    for user_id, ratings in user_rating_dict.items():
        user_rating_dict[user_id] = np.mean(ratings)
    for item_id, ratings in item_rating_dict.items():
        item_rating_dict[item_id] = np.mean(ratings)

    return user_rating_dict, item_rating_dict
