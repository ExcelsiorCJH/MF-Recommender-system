import os, io
import requests
import zipfile

import numpy as np

from collections import defaultdict
from .types_ import *


def get_data(
    download: bool = False, data_path: str = "data/ml-20m/ratings.csv",
) -> List[List[Union[int, int, float, int]]]:
    """
    Arguments
    ---------
    download : bool
        download dataset or not
    data_path : str
        ratings.csv dataset path
    Returns
    -------
    dataset : list of list[int, int, float, int]
        [[user_id, movie_id, rating, ts], ...]
    """
    if download or not os.path.exsts("data"):
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
            elif ts > 1388502017:
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


# def train_test_split(
#     data: np.ndarray,
#     data_type: str = 'DS2',
#     train_ts: List[int] = [1104505203, 1230735592],
#     test_ts: List[int] = [1230735600, 1262271552],
# ) -> Union[np.ndarray, np.ndarray]:
#     """
#     dataset split method
#     =====================

#     Arguments
#     ---------
#     data : np.ndarray
#         dataset to split train/test set
#     train_ts : list of int
#         range of train timestamp
#         option = ([1104505203, 1230735592], [789652000, 1388502017])
#     test_ts : list of int
#         range of test timestamp
#         option = ([1230735600, 1262271552], [1388502017, 1427784000])

#     Returns
#     -------
#     trainset : np.ndarray
#     testset : np.ndarray

#     """
#     train_ts.sort()
#     test_ts.sort()

#     trainset, testset = [], []
#     for row in data:
#         user_id, item_id, rating, ts = row
#         if train_ts[0] <= ts <= train_ts[1]:
#             trainset.append([user_id, item_id, rating, ts])
#         elif test_ts[0] <= ts <= test_ts[1]:
#             testset.append([user_id, item_id, rating, ts])

#     return np.array(trainset), np.array(testset)


# def train_test_split(
#     data: np.ndarray,
#     train_ts: List[int] = [1104505203, 1230735592],
#     test_ts: List[int] = [1230735600, 1262271552],
#     user_rating: bool = True,
# ) -> Union[np.ndarray, np.ndarray]:
#     """
#     dataset split method
#     =====================

#     Arguments
#     ---------
#     data : np.ndarray
#         dataset to split train/test set
#     train_ts : list of int
#         range of train's timestamp
#     test_ts : list of int
#         range of test's timestamp
#     user_rating : bool
#         use user specific ratings (only trainset)

#     Returns
#     -------
#     trainset : np.ndarray
#     testset : np.ndarray

#     """
#     train_ts.sort()
#     test_ts.sort()

#     if user_rating:
#         user_rating_dic = defaultdict(list)

#     trainset, testset = [], []
#     for row in data:
#         user_id, item_id, rating, ts = row
#         if train_ts[0] <= ts <= train_ts[1]:
#             trainset.append([user_id, item_id, rating, ts])
#             if user_rating:
#                 user_rating_dic[user_id].append(rating)
#         elif test_ts[0] <= ts <= test_ts[1]:
#             testset.append([user_id, item_id, rating, ts])

#     if user_rating:
#         for u_id, ratings in user_rating_dic.items():
#             user_rating_dic[u_id] = np.mean(ratings)

#         return np.array(trainset), np.array(testset), user_rating_dic
#     else:
#         return np.array(trainset), np.array(testset)
