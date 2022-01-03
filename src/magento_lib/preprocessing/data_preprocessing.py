import yaml
import pickle
import gcsfs
import pandas as pd
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Preprocessing logic
def aggregate_per_grouping_col(df_joined, params):
    """To do."""
    df_prod = df_joined.groupby(params["dataset"]["grouping_cols"]) \
        .agg({"order_qty": "size", "net_value": "sum"}).reset_index()
    df_prod = df_prod.rename(columns={"order_qty": "order_freq"})
    return df_prod


def get_rating_sku_storemaster_data(df_prod, params):
    """To do."""
    df_rating = df_prod[["cust_id", "product_key", "order_freq"]] \
        .rename(columns={"cust_id": "store", "product_key": "sku", "order_freq": "rating"})
    df_skumaster = df_prod[params["dataset"]["product_features_cols"]] \
        .set_index("product_key")
    df_storemaster = df_prod[params["dataset"]["customer_features_cols"]] \
        .set_index("cust_id")
    return df_rating, df_skumaster, df_storemaster


def compute_positive_ratings(df_rating):
    """To do."""
    from do_it.nbsku import ProcessRatings
    df_positive = ProcessRatings(df_rating[["store", "sku"]], rating=False) \
        .threshold_purchases().df_rating
    return df_positive


def build_inputs(df_rating, df_storemaster, df_skumaster):
    """To do."""
    from do_it.nbsku import InputsBuilder
    df_dataset, user_feature_matrix, item_feature_matrix = InputsBuilder(
        df_rating,
        df_storemaster=df_storemaster[df_storemaster.index.isin(df_rating["store"])],
        df_skumaster=df_skumaster[df_skumaster.index.isin(df_rating["sku"])]).build()
    return df_dataset, user_feature_matrix, item_feature_matrix

def register_interim_dataset(data, dataset_name):
    if isinstance(data, (dict, list)):
        interim_fs = gcsfs.GCSFileSystem(project= "tiger-mle", token="/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json")
        pickle_file = f"gs://pg-explore/data/magento/interim/{dataset_name}.pkl"
        with interim_fs.open(pickle_file, "wb") as interim_dataset_pickle:
            pickle.dump(data, interim_dataset_pickle)


def start_preprocessing():
    logger.info("Preprocessing Method Ends")
    interim_fs = gcsfs.GCSFileSystem(project= "tiger-mle")
    params_path = f"gs://pg-explore/data/magento/interim/argentina_parameters.yml"
    with interim_fs.open(params_path, "r") as fp:
            params = yaml.safe_load(fp)


    df_joined = pd.read_csv("gs://pg-explore/data/sample_dataset/intermediate/joined/df_joined_Argentina.csv", storage_options={"token":"/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json"})

    df_prod = aggregate_per_grouping_col(df_joined, params)

    df_rating, df_skumaster, df_storemaster = get_rating_sku_storemaster_data(
                    df_prod, params)


    df_rating = df_rating.drop_duplicates(subset=['store', 'sku'])

    df_positive = compute_positive_ratings(df_rating)

    df_dataset, user_feature_matrix, item_feature_matrix = build_inputs(
                    df_rating, df_storemaster, df_skumaster)

    # Saving interim dataframe for training

    dict_prod = {}
    dict_positive = {}
    dict_dataset = {}
    dict_user_feature_matrix = {}
    dict_item_feature_matrix = {}
    country = "Argentina"

    dict_prod[country] = df_prod
    dict_positive[country] = df_positive
    dict_dataset[country] = df_dataset
    dict_user_feature_matrix[country] = user_feature_matrix
    dict_item_feature_matrix[country] = item_feature_matrix


    register_interim_dataset(dict_positive, "dict_positive")
    register_interim_dataset(dict_prod, "dict_prod")
    register_interim_dataset(dict_dataset, "dict_dataset")
    register_interim_dataset(dict_user_feature_matrix, "dict_user_feature_matrix")
    register_interim_dataset(dict_item_feature_matrix, "dict_item_feature_matrix")
    logger.info("Preprocessing Method Ends")