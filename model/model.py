import time
import numpy as np

from utils import rmse, time_since
from utils.types_ import *


class MF(object):
    def __init__(
        self,
        n_user: int,
        n_item: int,
        n_feature: int = 10,
        batch_size: int = 1e5,
        learning_rate: float = 50.0,
        momentum: float = 0.8,
        reg: float = 1e-2,
        converge: float = 1e-5,
        max_rating: int = 5,
        min_rating: int = 1,
        is_bias: bool = True,
    ):

        self.n_user = n_user  # num of users
        self.n_item = n_item  # num of items
        self.n_feature = n_feature

        self.batch_size = batch_size  # batch_size
        self.learning_rate = learning_rate  # learning rate
        self.momentum = momentum

        self.reg = reg  # regularization
        self.converge = converge

        self.max_rating = max_rating
        self.min_rating = min_rating

        # use_bias or not
        self.is_bias = is_bias

        # initialize user/item features
        self.user_features = 0.1 * np.random.rand(n_user, n_feature)
        self.item_features = 0.1 * np.random.rand(n_item, n_feature)

        # initialize user/item bias
        if self.is_bias:
            self.user_bias, self.item_bias = self._init_bias()

    def fit(self, ratings: np.ndarray, user_rating_dic: Dict = None, epochs: int = 30):
        self.mean_rating = np.mean(ratings[:, 2])
        prev_loss = 0

        # user/item features gradient & momentum
        user_feature_grads, user_feature_mom = (
            np.zeros((self.n_user, self.n_feature)),
            np.zeros((self.n_user, self.n_feature)),
        )
        item_feature_grads, item_feature_mom = (
            np.zeros((self.n_item, self.n_feature)),
            np.zeros((self.n_item, self.n_feature)),
        )

        # user/item bias gradient & momentum
        if self.is_bias:
            user_bias_grads, user_bias_mom = np.zeros(self.n_user), np.zeros(self.n_user)
            item_bias_grads, item_bias_mom = np.zeros(self.n_item), np.zeros(self.n_item)

        batch_num = int(ratings.shape[0] // self.batch_size)
        start_time = time.time()
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

                outputs = np.sum(u_features * i_features, axis=1)
                if user_rating_dic:
                    self.user_rating_dic = user_rating_dic
                    user_ratings = [
                        user_rating_dic.get(u_id, self.mean_rating) for u_id in user_ids
                    ]
                    errs = outputs - (batch.take(2, axis=1) - np.array(user_ratings))
                else:
                    errs = outputs - (batch.take(2, axis=1) - self.mean_rating)

                if self.is_bias:
                    u_bias = self.user_bias.take(user_ids)
                    i_bias = self.item_bias.take(item_ids)
                    errs += u_bias + i_bias
                    u_bias_grads = u_bias * errs + self.reg * u_bias
                    i_bias_grads = i_bias * errs + self.reg * i_bias
                    user_bias_grads.fill(0.0)
                    item_bias_grads.fill(0.0)

                err_mat = np.tile(2 * errs, (self.n_feature, 1)).T
                user_grads = i_features * err_mat + self.reg * u_features
                item_grads = u_features * err_mat + self.reg * i_features

                user_feature_grads.fill(0.0)
                item_feature_grads.fill(0.0)
                for idx in range(batch.shape[0]):
                    user_id, item_id, rating = batch[idx]
                    user_id, item_id = int(user_id), int(item_id)

                    user_feature_grads[user_id, :] += user_grads[idx]
                    item_feature_grads[item_id, :] += item_grads[idx]
                    if self.is_bias:
                        user_bias_grads[user_id] += u_bias_grads[idx]
                        item_bias_grads[item_id] += i_bias_grads[idx]

                # update momentum
                user_feature_mom = (self.momentum * user_feature_mom) + (
                    (self.learning_rate / batch.shape[0]) * user_feature_grads
                )
                item_feature_mom = (self.momentum * item_feature_mom) + (
                    (self.learning_rate / batch.shape[0]) * item_feature_grads
                )
                # update user/item matrix
                self.user_features -= user_feature_mom
                self.item_features -= item_feature_mom

                if self.is_bias:
                    user_bias_mom = (self.momentum * user_bias_mom) + (
                        (self.learning_rate / batch.shape[0]) * user_bias_grads
                    )
                    item_bias_mom = (self.momentum * item_bias_mom) + (
                        (self.learning_rate / batch.shape[0]) * item_bias_grads
                    )
                    # update user/item bias
                    self.user_bias -= user_bias_mom
                    self.item_bias -= item_bias_mom

            # loss
            train_preds = self.predict(ratings[:, :2])
            train_loss = rmse(train_preds, ratings[:, 2])
            print(
                f"ellapse: {time_since(start_time)} | epoch: {epoch:03d} | train RMSE: {train_loss:.6f}"
            )

            # early-stopping
            if abs(train_loss - prev_loss) < self.converge:
                print(f"train loss decreased {prev_loss} --> {train_loss}. Stop.")

            prev_loss = train_loss
        return None

    def _init_bias(self) -> Union[np.ndarray, np.ndarray]:
        user_bias = np.zeros(self.n_user)
        item_bias = np.zeros(self.n_item)
        return user_bias, item_bias

    def predict(self, ratings: np.ndarray) -> np.ndarray:
        user_ids = ratings.take(0, axis=1).astype(int)
        item_ids = ratings.take(1, axis=1).astype(int)
        u_features = self.user_features.take(user_ids, axis=0)
        i_features = self.item_features.take(item_ids, axis=0)

        if self.is_bias:
            u_bias = self.user_bias.take(user_ids)
            i_bias = self.item_bias.take(item_ids)
            if self.user_rating_dic:
                user_ratings = [
                    self.user_rating_dic.get(u_id, self.mean_rating) for u_id in user_ids
                ]
                preds = (
                    np.sum(u_features * i_features, axis=1)
                    + np.array(user_ratings)
                    + u_bias
                    + i_bias
                )
            else:
                preds = np.sum(u_features * i_features, axis=1) + self.mean_rating + u_bias + i_bias
        else:
            if self.user_rating_dic:
                user_ratings = [
                    self.user_rating_dic.get(u_id, self.mean_rating) for u_id in user_ids
                ]
                preds = np.sum(u_features * i_features, axis=1) + np.array(user_ratings)
            else:
                preds = np.sum(u_features * i_features, axis=1) + self.mean_rating

        # clip rating
        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds

    def save(self):
        pass
