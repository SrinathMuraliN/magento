import pickle
import gcsfs
import yaml
import logging
from datetime import datetime
from itertools import islice
from sys import stdout

from lightfm import LightFM
from lightfm.evaluation import auc_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def import_data():

    interim_fs = gcsfs.GCSFileSystem(project= "tiger-mle")
    params_path = f"gs://pg-explore/data/magento/interim/argentina_parameters.yml"
    with interim_fs.open(params_path, "r") as fp:
            params = yaml.safe_load(fp)

    """Importing dict_positive from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_positive.pkl"
    with interim_fs.open(pickle_file, 'rb') as handle:
            pickle_obj = pickle.load(handle)
    dict_positive = pickle_obj['Argentina']

    """Importing dict_dataset from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_dataset.pkl"
    with interim_fs.open(pickle_file, 'rb') as handle:
            pickle_obj = pickle.load(handle)
    dict_dataset = pickle_obj['Argentina']

    """Importing dict_user_feature_matrix from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_user_feature_matrix.pkl"
    with interim_fs.open(pickle_file, 'rb') as handle:
            pickle_obj = pickle.load(handle)
    dict_user_feature_matrix = pickle_obj['Argentina']

    """Importing dict_item_feature_matrix from gcs"""
    pickle_file = f"gs://pg-explore/data/magento/interim/dict_item_feature_matrix.pkl"
    with interim_fs.open(pickle_file, 'rb') as handle:
            pickle_obj = pickle.load(handle)
    dict_item_feature_matrix = pickle_obj['Argentina']

    return dict_positive, dict_dataset, dict_user_feature_matrix,dict_item_feature_matrix,params


def fit_model(
    interactions,
    *,
    weights=None,
    user_features=None,
    item_features=None,
    num_epochs=1,
    num_threads=1,
    loss="warp",
    **kwargs,
):
    """See https://making.lyst.com/lightfm/docs/lightfm.html#lightfm.LightFM.fit."""
    hyperparams = {**kwargs}
    hyperparams["loss"] = loss
    model = LightFM(**hyperparams)
    model.fit(
        interactions,
        sample_weight=weights,
        user_features=user_features,
        item_features=item_features,
        epochs=num_epochs,
        num_threads=num_threads,
    )
    return model

class RandomSearchCV:
    """
    Randomized search on hyper parameters.

    Parameters
    ----------
    df_positive: pandas.DataFrame
        Dataframe contains unique positive interactions of user-item pairs,
        has columns of the form ['user', 'item'] or ['user', 'item', 'weights']
    dataset: lightfm.data.Dataset
        Lightfm dataset that has been fitted with the user/item ids and feature names.
    user_features: csr_matrix of shape [n_users, n_users + n_user_features], optional
        Each row contains that user’s weights over features.
    item_features: csr_matrix of shape [n_items, n_items + n_item_features], optional
        Each row contains that item’s weights over features.
    test_size: float
        Proportion of the dataset to include in the test split.
    nfolds: int, optional
        Number of folds in k-fold cv, must be at least 2. Defaults to 5.
    """

    def __init__(
        self,
        df_positive,
        dataset,
        user_features=None,
        item_features=None,
        test_size=0.1,
        nfolds=5,
    ):
        self.df_positive = df_positive
        self.dataset = dataset
        self.user_features = user_features
        self.item_features = item_features
        self.train, self.test = train_test_split(df_positive, test_size=test_size)
        self.nfolds = nfolds

    def sample_hyperparameters(self):
        """Code from https://stackoverflow.com/a/49983651."""
        while True:
            yield {
                "no_components": np.random.randint(16, 64),
                "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
                "learning_rate": np.random.exponential(0.05),
                "item_alpha": np.random.exponential(1e-8),
                "user_alpha": np.random.exponential(1e-8),
                "max_sampled": np.random.randint(5, 15),
                "num_epochs": np.random.randint(5, 50),
            }

    def train_test_split_(self):
        """Build interaction and weight matrices for train and test sets."""
        train_interactions, train_weights = self.dataset.build_interactions(
            self.train.to_numpy()
        )
        test_interactions, _ = self.dataset.build_interactions(self.test.to_numpy())
        return train_interactions, train_weights, test_interactions

    def kfold_split(self):
        """Yield interaction and weight matrices from k-fold train and validation sets."""
        kf = KFold(n_splits=self.nfolds, shuffle=True)
        for train_index, val_index in kf.split(self.train):
            # Train_interactions will be a matrix with SKU and Store with a marker without
            # indicating the strength of the relationship (TBC),
            # train_weights actually strength of the relationship.
            train_interactions, train_weights = self.dataset.build_interactions(
                self.train.iloc[train_index, :].to_numpy()
            )
            interactions_val, _ = self.dataset.build_interactions(
                self.train.iloc[val_index, :].to_numpy()
            )
            yield train_interactions, train_weights, interactions_val

    def random_search(self, num_samples=10, num_threads=1, loss="warp"):
        """Random search hyperparameters using k-fold cross-validation."""
        for i, hyperparams in enumerate(
            islice(self.sample_hyperparameters(), num_samples), 1
        ):
            now = datetime.now().strftime("%H:%M:%S")
            stdout.write(f"{now}, evaluating the hyperparameter set NO.{i}...\n")
            scores = []
            for no_fold, (
                train_interactions,
                train_weights,
                val_interactions,
            ) in enumerate(self.kfold_split(), 1):
                now = datetime.now().strftime("%H:%M:%S")
                stdout.write(
                    f"  {now}, Training and evaluating with fold NO.{no_fold}...\n"
                )
                model = fit_model(
                    interactions=train_interactions,
                    weights=train_weights,
                    user_features=self.user_features,
                    item_features=self.item_features,
                    num_threads=num_threads,
                    loss=loss,
                    **hyperparams,
                )
                validation_score = auc_score(
                    model,
                    train_interactions=train_interactions,
                    test_interactions=val_interactions,
                    user_features=self.user_features,
                    item_features=self.item_features,
                ).mean()
                scores.append(validation_score)
            score = np.mean(scores)
            stdout.write(
                f"Validation score for hyperparams set {hyperparams} is {score}.\n\n"
            )
            yield (score, hyperparams)

    def select_hyperparams(self, num_samples=10, num_threads=1, loss="warp"):
        """Select hyperparameters using k-fold cross-validation."""
        # Select the parameters associated with the maximum AUC from random_search,
        # lambda x:x[0] specifies AUC
        stdout.write("Random search started.\n")
        _, hyperparams = max(
            self.random_search(
                num_samples=num_samples,
                num_threads=num_threads,
                loss=loss,
            ),
            key=lambda x: x[0],
        )
        # We are getting the AUC from entire dataset:
        # (train - 90% of entire dataset, test- 10% of entire dataset)
        stdout.write("Evaluating the model with the best hyperparameters...\n")
        train_interactions, train_weights, test_interactions = self.train_test_split_()
        model = fit_model(
            interactions=train_interactions,
            weights=train_weights,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=num_threads,
            loss=loss,
            **hyperparams,
        )
        test_score = auc_score(
            model,
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            user_features=self.user_features,
            item_features=self.item_features,
        ).mean()
        stdout.write(f"Test score is {test_score}.\n")
        return test_score, hyperparams

    def select_model(
        self, num_samples: int = 10, num_threads: int = 1, loss: str = "warp"
    ):
        """
        Train the model with the whole dataset and the best hyperparameters.

        Parameters
        ----------
        num_samples
            Number of combinations of hyperparameters to try.
        num_epochs
            Number of epochs to run.
        num_threads
            Number of parallel computation threads to use.
            Should not be higher than the number of physical cores.
        loss
            One of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’): the loss function.

            We suggest to use warp-kos if your rating dataframe doesn't have a
            third column, warp if it does. They perform the best and they are the
            distinguishing features of the lightfm package.

            See https://making.lyst.com/lightfm/docs/lightfm.html#lightfm.LightFM

        Returns
        -------
        test_score: float
            Test auc to report.
        model: LightFM instance
            The trained model.
        """
        test_score, hyperparams = self.select_hyperparams(
            num_samples=num_samples,
            num_threads=num_threads,
            loss=loss,
        )
        stdout.write("Fitting the whole dataset with the best hyperparameter set...\n")
        train_interactions, train_weights = self.dataset.build_interactions(
            self.df_positive.to_numpy()
        )
        model = fit_model(
            interactions=train_interactions,
            weights=train_weights,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=num_threads,
            loss=loss,
            **hyperparams,
        )
        stdout.write("Done.\n")
        return test_score, model

def perform_cross_validation(
    df_positive,
    df_dataset,
    user_feature_matrix,
    item_feature_matrix,
    params
):
    """Feed the 4 outputs got from the model input stage to this stage.

    Add loss="warp-kos" to select_model.
    If your df_positive doesn't have a 3rd rating column.
    Note that because the demo data are generated randomly.
    Any good model will only produce around 50% AUC.
    """
    test_score, best_model = RandomSearchCV(
        df_positive,
        df_dataset,
        user_features=user_feature_matrix,
        item_features=item_feature_matrix,
        test_size=params["dataset"]["test_size"],
        nfolds=params["dataset"]["nfolds"]) \
        .select_model(num_samples=params["dataset"]["num_samples"],
                      num_threads=params["dataset"]["num_threads"])
    return test_score, best_model


def start_training():

    logger.info("starting training...")

    # Importing data for model training
    dict_positive, dict_dataset, dict_user_feature_matrix,dict_item_feature_matrix,params = import_data()
    """Run model training step."""

    dict_test_score = {}
    dict_best_model = {}

    country = 'Argentina'
    logger.info ("cross validation country:", country)

    test_score, best_model = perform_cross_validation(
        dict_positive,dict_dataset,dict_user_feature_matrix,dict_item_feature_matrix, params)

    logger.info ("auc", test_score)
    logger.info ("hyperparameters", best_model.get_params())

    dict_test_score[country] = test_score
    dict_best_model[country] = best_model
    logger.info (dict_best_model)
    logger.info("Training Completed")

