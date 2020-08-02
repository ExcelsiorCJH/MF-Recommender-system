import os
import time
import numpy as np

from collections import defaultdict
from utils import rmse, time_since, to_csv
from utils.types_ import *


class AddPMF(object):
    def __init__(
        self,
        n_user: int,
        n_item: int,
        user_rating_dict: Dict,
        item_rating_dict: Dict,
        n_feature: int = 100,
        batch_size: int = 1e5,
        learning_rate: float = 0.0005,
        momentum: float = 0.9,
        reg: float = 1e-2,
        alpha: float = 0.4,
        max_rating: float = 5.0,
        min_rating: float = 1.0,
        save_dir: str = "outputs/DS2",
    ):
        """
        Probabilistic Matrix Factorization(PMF) + Bias Class
        =============================================

        Arguments
        ---------
        n_user : int
            number of users
        n_item : int
            number of items
        n_feature : int
            latent vector dimension
        batch_size : int
            batch size for training
        learning_rate : float
            learning rate
        momentum : float
            SGD with momentum
        reg : float
            regularization parameter
        alpha: float = 0.4
            Weight for user/item rating
        max_rating : float
            maximum of rating
        min_rating : float
            minimum of rating
        user_rating_dict : Dict
            user specific average ratings
        item_rating_dict : Dict
            item specific average ratings
        save_dir: str
            save model/results dir
        """

        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.reg = reg

        self.max_rating = max_rating
        self.min_rating = min_rating

        self.save_dir = save_dir  # f"../{save_dir}"

        # specific user/item mean ratings
        self.user_rating_dict = user_rating_dict
        self.item_rating_dict = item_rating_dict
        self.alpha = alpha

        # initialize user/item features
        self.user_features = 0.1 * np.random.rand(n_user, n_feature)
        self.item_features = 0.1 * np.random.rand(n_item, n_feature)

        # initialize user/item bias
        self.user_bias = np.zeros(n_user)
        self.item_bias = np.zeros(n_item)

    def fit(self, ratings: np.ndarray, validset: np.ndarray = None, epochs: int = 20,) -> None:
        """
        Arguments
        ---------
        ratings : np.ndarray
            ratings matrix for training (i.e. train dataset)
        validset : np.ndarray
            validation dataset
        epochs : int
            number of iterations
        """

        # average of ratings
        self.mean_rating = np.mean(ratings[:, 2])
        # minimum of ratings
        if self.min_rating > np.min(ratings[:, 2]):
            self.min_rating = np.min(ratings[:, 2])

        best_loss = 0

        # initialize user gradient & momentum
        user_feature_grads, user_feature_mom = (
            np.zeros((self.n_user, self.n_feature)),
            np.zeros((self.n_user, self.n_feature)),
        )
        # initialize item gradient & momentum
        item_feature_grads, item_feature_mom = (
            np.zeros((self.n_item, self.n_feature)),
            np.zeros((self.n_item, self.n_feature)),
        )
        # initialize user/item bias gradient & momentum
        user_bias_grads, user_bias_mom = np.zeros(self.n_user), np.zeros(self.n_user)
        item_bias_grads, item_bias_mom = np.zeros(self.n_item), np.zeros(self.n_item)

        batch_num = int(np.ceil(ratings.shape[0] / self.batch_size))
        start_time = time.time()
        self.train_losses = []
        self.valid_losses = []
        for epoch in range(1, epochs + 1):
            # dataset shuffling
            np.random.shuffle(ratings)

            for batch_idx in range(batch_num):
                start_idx = int(batch_idx * self.batch_size)
                end_idx = int((batch_idx + 1) * self.batch_size)
                batch = ratings[start_idx:end_idx]

                # compute gradient
                user_ids = batch.take(0, axis=1).astype(int)
                item_ids = batch.take(1, axis=1).astype(int)
                u_features = self.user_features.take(user_ids, axis=0)
                i_features = self.item_features.take(item_ids, axis=0)
                u_bias = self.user_bias.take(user_ids)
                i_bias = self.item_bias.take(item_ids)

                user_ratings, item_ratings = [], []
                for u_id, i_id in zip(user_ids, item_ids):
                    user_ratings.append(self.user_rating_dict.get(u_id, self.mean_rating))
                    item_ratings.append(self.item_rating_dict.get(i_id, self.mean_rating))

                user_ratings, item_ratings = np.array(user_ratings), np.array(item_ratings)
                weight_ratings = self.alpha * user_ratings + (1 - self.alpha) * item_ratings
                # outputs = np.sum(
                #     (u_features + u_bias[:, np.newaxis]) * (i_features + i_bias[:, np.newaxis]),
                #     axis=1,
                # )
                outputs = np.sum(u_features * i_features, axis=1) + u_bias + i_bias
                errs = outputs - (batch.take(2, axis=1) - weight_ratings)
                err_mat = np.tile(2 * errs, (self.n_feature, 1)).T

                user_grads = i_features * err_mat + self.reg * u_features
                item_grads = u_features * err_mat + self.reg * i_features
                u_bias_grads = u_bias * errs + self.reg * u_bias
                i_bias_grads = i_bias * errs + self.reg * i_bias

                # clear all gradients
                user_feature_grads.fill(0.0)
                item_feature_grads.fill(0.0)
                user_bias_grads.fill(0.0)
                item_bias_grads.fill(0.0)
                for idx in range(batch.shape[0]):
                    user_id, item_id, rating = batch[idx]
                    user_id, item_id = int(user_id), int(item_id)

                    user_feature_grads[user_id, :] += user_grads[idx]
                    item_feature_grads[item_id, :] += item_grads[idx]
                    user_bias_grads[user_id] += u_bias_grads[idx]
                    item_bias_grads[item_id] += i_bias_grads[idx]

                # update momentum
                user_feature_mom = (
                    self.momentum * user_feature_mom + self.learning_rate * user_feature_grads
                )
                item_feature_mom = (
                    self.momentum * item_feature_mom + self.learning_rate * item_feature_grads
                )
                user_bias_mom = self.momentum * user_bias_mom + self.learning_rate * user_bias_grads
                item_bias_mom = self.momentum * item_bias_mom + self.learning_rate * item_bias_grads

                # update user/item matrix
                self.user_features -= user_feature_mom
                self.item_features -= item_feature_mom
                # update user/item bias
                self.user_bias -= user_bias_mom
                self.item_bias -= item_bias_mom

            # rmse train loss
            train_preds = self.predict(ratings[:, :2])
            train_loss = rmse(train_preds, ratings[:, 2])
            self.train_losses.append(train_loss)

            # save losses
            if validset is None:
                print(
                    f"ellapse: {time_since(start_time)} | epoch: {epoch:03d} | train RMSE: {train_loss:.6f}"
                )

            else:
                valid_preds = self.predict(validset[:, :2])
                valid_loss = rmse(valid_preds, validset[:, 2])
                self.valid_losses.append(valid_loss)
                print(
                    f"ellapse: {time_since(start_time)} | epoch: {epoch:03d} | train RMSE: {train_loss:.6f} | valid RMSE: {valid_loss:.6f}"
                )

                # save result & weights
                if best_loss == 0 or valid_loss < best_loss:
                    best_loss = valid_loss

                    result_dir = f"{self.save_dir}/results"
                    weight_dir = f"{self.save_dir}/weights"

                    to_csv(
                        f"{result_dir}/output_val_addpmf.csv", valid_preds, validset, header=True
                    )
                    self._save_model(weight_dir)
                    print(f"save result and model weights at {best_loss}")

        return None

    def predict(self, ratings: np.ndarray) -> np.ndarray:
        """
        Arguments
        ---------
        ratings : np.ndarray
            ratings matrix for training (i.e. test dataset)
        
        Returns
        -------
        preds : np.ndarray
            prediction of test dataset
        """
        user_ids = ratings.take(0, axis=1).astype(int)
        item_ids = ratings.take(1, axis=1).astype(int)
        u_features = self.user_features.take(user_ids, axis=0)
        i_features = self.item_features.take(item_ids, axis=0)
        u_bias = self.user_bias.take(user_ids)
        i_bias = self.item_bias.take(item_ids)

        user_ratings, item_ratings = [], []
        for u_id, i_id in zip(user_ids, item_ids):
            user_ratings.append(self.user_rating_dict.get(u_id, self.mean_rating))
            item_ratings.append(self.item_rating_dict.get(i_id, self.mean_rating))

        user_ratings, item_ratings = np.array(user_ratings), np.array(item_ratings)
        weight_ratings = self.alpha * user_ratings + (1 - self.alpha) * item_ratings
        preds = np.sum(u_features * i_features, axis=1) + u_bias + i_bias + weight_ratings
        # preds = (
        #     np.sum(
        #         (u_features + u_bias[:, np.newaxis]) * (i_features + i_bias[:, np.newaxis]), axis=1
        #     )
        #     + self.mean_rating
        # )

        # clip rating
        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating
        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def load_weights(
        self, file_name: str = "outputs/DS2/weights/best_model_addpmf.npz",
    ):
        # w_dict = np.load(f"../{file_name}")
        w_dict = np.load(file_name)
        self.user_features = w_dict["user_features"]
        self.item_features = w_dict["item_features"]
        self.user_bias = w_dict["user_bias"]
        self.item_bias = w_dict["item_bias"]
        self.mean_rating = w_dict["mean_rating"]

        print("load pre-trained model complete!")
        return None

    def _save_model(self, weight_dir: str):
        """
        save best model's weights
        """
        np.savez(
            f"{weight_dir}/best_model_addpmf.npz",
            user_features=self.user_features,
            item_features=self.item_features,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            mean_rating=self.mean_rating,
        )
        return None
