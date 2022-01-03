import logging
import os
import pickle
from collections import namedtuple
from dataclasses import dataclass, fields
from datetime import timedelta

import gcsfs
import numpy as np
import pandas as pd
import yaml

logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def import_data():
    """Declaring the default Parameters"""

    interim_fs = gcsfs.GCSFileSystem(project="tiger-mle")
    params_path = (
        f"gs://pg-explore/data/magento/interim/argentina_parameters.yml"
    )
    with interim_fs.open(params_path, "r") as fp:
        params = yaml.safe_load(fp)

    """Importing best model from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/model_argentina.pkl"
    with interim_fs.open(pickle_file, "rb") as handle:
        best_model = pickle.load(handle)

    """Importing df_dataset from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_dataset.pkl"
    with interim_fs.open(pickle_file, "rb") as handle:
        pickle_obj = pickle.load(handle)
    df_dataset = pickle_obj["Argentina"]

    """Importing user_feature_matrix from gcs"""
    pickle_file = (
        f"gs://pg-explore/data/magento/interim/dict_user_feature_matrix.pkl"
    )
    with interim_fs.open(pickle_file, "rb") as handle:
        pickle_obj = pickle.load(handle)
    user_feature_matrix = pickle_obj["Argentina"]

    """Importing item_feature_matrix from gcs"""
    pickle_file = (
        f"gs://pg-explore/data/magento/interim/dict_item_feature_matrix.pkl"
    )
    with interim_fs.open(pickle_file, "rb") as handle:
        pickle_obj = pickle.load(handle)
    item_feature_matrix = pickle_obj["Argentina"]

    """Importing df_prod from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_prod.pkl"
    with interim_fs.open(pickle_file, "rb") as handle:
        pickle_obj = pickle.load(handle)
    df_prod = pickle_obj["Argentina"]

    """Importing df_joined from gcs"""
    df_joined = pd.read_csv(
        r"gs://pg-explore/data/sample_dataset/intermediate/joined/df_joined_Argentina.csv"
    )

    return (
        df_prod,
        best_model,
        df_dataset,
        user_feature_matrix,
        item_feature_matrix,
        df_joined,
        params,
    )


class Model:
    """
    Class to facilitate getting product recommendation.

    Parameters
    ----------
    dataset: lightfm.data.Dataset
        Fitted dataset.
    model: lightfm.lightfm.LightFM
        Trained model.
    user_features: csr_matrix of shape [n_users, n_users + n_user_features], optional
        Each row contains that user’s weights over features.
    item_features: csr_matrix of shape [n_items, n_items + n_item_features], optional
        Each row contains that item’s weights over features.
    """

    def __init__(self, dataset, model, user_features=None, item_features=None):
        self.model = model
        self.user_features = user_features
        self.item_features = item_features
        # Internal, external id mappings
        mappings = dataset.mapping()
        self.ex_in_user = mappings[0]
        self.ex_in_item = mappings[2]
        # User, item numbers
        self.n_users = len(self.ex_in_user)
        self.n_items = len(self.ex_in_item)
        # Recommendation scores
        self.score = None

    def __score(self) -> np.ndarray:
        # __score will affect the ranking of the SKU within the recommendation of the store
        # Predicted scores
        user_internal_ids = list(self.ex_in_user.values())
        item_internal_ids = list(self.ex_in_item.values())
        model_dict = self.model
        model = model_dict["Argentina"]
        return model.predict(
            user_ids=np.repeat(user_internal_ids, self.n_items),
            item_ids=np.tile(item_internal_ids, self.n_users),
            user_features=self.user_features,
            item_features=self.item_features,
        ).reshape(self.n_users, self.n_items)

    def recommend(self) -> pd.DataFrame:
        """
        Recommend products for users.

        Returns
        -------
        recommendations:
            Every user with a item list ordered from the most recommended item to the least.
        """
        user_external_ids_ = list(self.ex_in_user.keys())
        item_external_ids_ = list(self.ex_in_item.keys())
        user_external_ids = np.repeat(user_external_ids_, self.n_items)
        item_external_ids = np.tile(item_external_ids_, self.n_users)
        # Getting scores for every user-item pair
        score = self.score
        if score is None:
            score = self.score = self.__score()
        order = np.argsort(-score)
        score_ = np.take_along_axis(score, order, axis=1).flatten()
        rank_ = np.tile(np.arange(1, self.n_items + 1), self.n_users)
        item_external_ids = item_external_ids.reshape(
            self.n_users, self.n_items
        )
        items_ = np.take_along_axis(item_external_ids, order, axis=1).flatten()
        """ Declaring the default Parameters"""
        SDM_USER = "user"
        SDM_ITEM = "item"
        SDM_RANK = "ranking"
        df_reco = pd.DataFrame(
            np.array([user_external_ids, items_, score_, rank_]).T,
            columns=[SDM_USER, SDM_ITEM, "score", SDM_RANK],
        )
        return (
            df_reco.astype(
                {
                    SDM_USER: type(user_external_ids_[0]),
                    SDM_ITEM: type(item_external_ids_[0]),
                    "score": float,
                    SDM_RANK: int,
                }
            )
            .set_index([SDM_USER, SDM_ITEM])
            .sort_index(level=0, sort_remaining=False)
        )


def build_model(
    df_dataset, best_model, user_feature_matrix, item_feature_matrix
):
    """To do."""
    model = Model(
        df_dataset,
        best_model,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix,
    )
    return model


def filter_by_recency(df_joined, params):
    """Filter df_joined by start date, return df_recent."""
    recency = params["dataset"]["recency"]
    df_joined["sales_document_date"] = pd.to_datetime(
        df_joined["sales_document_date"]
    )
    start_date = df_joined["sales_document_date"].max() - timedelta(recency)
    df_joined = df_joined[df_joined["sales_document_date"] > start_date]
    df_recent = (
        df_joined.groupby(
            ["subsector", "category", "brand", "product_key", "prod_name"]
        )
        .size()
        .reset_index()
        .drop(columns=0)
    )
    return df_recent


def create_model_recommendations(model, df_recent, df_prod, country):
    """To do."""
    rank_df = model.recommend().reset_index()
    # left outer join to get recommended, but not purchased
    recom_df = (
        rank_df.merge(
            df_prod[["cust_id", "product_key"]],
            left_on=["user", "item"],
            right_on=["cust_id", "product_key"],
            how="outer",
            indicator=True,
        )
        .query('_merge=="left_only"')
        .drop(columns=["cust_id", "product_key", "_merge"])
    )

    recom_df = (
        recom_df.merge(df_recent, left_on=["item"], right_on=["product_key"])
        .sort_values(["user", "ranking"])
        .drop(columns=["item"])
        .rename(columns={"user": "cust_id"})
    )
    cols = [
        "cust_id",
        "subsector",
        "category",
        "brand",
        "product_key",
        "prod_name",
        "score",
        "ranking",
    ]
    recom_df = recom_df[cols]
    recom_df["country"] = country
    return recom_df


"""Model evaluation step."""


def start_inference():
    print("Inside the method start inference")
    logger.info("Inference method starts ")
    """Importing the data """
    (
        df_prod,
        best_model,
        df_dataset,
        user_feature_matrix,
        item_feature_matrix,
        df_joined,
        params,
    ) = import_data()
    """Model Inference."""
    dict_recom = {}
    country = "Argentina"
    print("Before starting the recommendation")
    model = build_model(
        df_dataset, best_model, user_feature_matrix, item_feature_matrix
    )
    df_recent = filter_by_recency(df_joined, params)
    recom_df = create_model_recommendations(model, df_recent, df_prod, country)
    print(recom_df)
    dict_recom[country] = recom_df
    print("Method start inference - Ends")
    logger.info(dict_recom)
    logger.info("Inference method ends ")
