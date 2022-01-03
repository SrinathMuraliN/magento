import gcsfs
import pandas as pd
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Transformation Logic

def transform_order_data(df_order):
    """To do."""
    # convert datatime
#     df_order["sales_document_date"] = df_order["sales_document_date"] \
#         .astype(str).apply(lambda x: x[:4]+"-"+x[4:6]+"-"+x[6:])
    df_order["sales_document_date"] = pd.to_datetime(df_order["sales_document_date"])

    # keep processing, shipped or partially shipped orders, filter on-hold and cancelled orders
    df_order = df_order[df_order["order_header_status"]
                        .isin(["Shipped", "Partially Shipped", "Processing"])]

    # consolidate duplicated orders and keep selected fields
    cols = ["sales_organization", "sales_order_number", "sales_document_date",
            "sold_to_party", "product_key"]
    df_order = df_order.groupby(cols).agg({"order_qty": "sum", "net_value":
                                           "sum", "total": "first", "order_uom": "first",
                                           "order_header_status": "first"}).reset_index()

    # rename columns and keep only columns needed
    df_order = df_order.rename(columns={"sales_organization": "sales_org",
                                        "sold_to_party": "cust_id"})

    order_cols = ["sales_org", "cust_id", "sales_order_number", "sales_document_date",
                  "product_key", "order_qty", "net_value", "total", "order_uom",
                  "order_header_status"]

    df_order = df_order[order_cols]


    return df_order


def transform_customer_overview(df_customer_overview):
    """To do."""
    customeroverview_cols = ["sales_org", "cust_id", "city", "assortment_groups",
                             "customer_classification"]
    df_customer_overview = df_customer_overview[customeroverview_cols]
    return df_customer_overview


def transform_product_overview(df_product_overview):
    """To do."""
    productoverview_cols = ["sales_org", "product_key",
                            "subsector", "subsector_id", "category", "category_id", "brand",
                            "brand_id", "item_gtin", "prod_name"]
    df_product_overview = df_product_overview[productoverview_cols]
    return df_product_overview


# Joining customer_overview, order and product_overview

def create_joined_data(df_order, df_customer_overview, df_product_overview):
    """To do."""
    df_joined = df_order.merge(df_customer_overview, on=["sales_org", "cust_id"]) \
        .merge(df_product_overview, on=["sales_org", "product_key"])
    return df_joined

def load_data():
    logger.info("Load Data Method Starts")
    customer_overview_path = "gs://pg-explore/data/magento/OneMillionDataset/Customer_Table1k.csv/part-00000-3c8424df-7f2a-4880-a7a0-fd9d5b71868d-c000.csv"
    products_overview_path = "gs://pg-explore/data/magento/OneMillionDataset/Product1k_coalesce.csv/part-00000-d60ccfcd-fce3-4747-b88c-6ac6f5bb4e19-c000.csv"
    orders_history_path = "gs://pg-explore/data/magento/OneMillionDataset/Order_coalesce1M.csv/part-00000-1906bbca-26b1-48f7-8c75-14735d4d1772-c000.csv"


    # Loading customer data
    customers_fs = gcsfs.GCSFileSystem(project= "tiger-mle", token="/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json")
    with customers_fs.open(customer_overview_path) as customers:
        df_customer_overview = pd.read_csv(customers)

    # Loading order history data
    orders_fs = gcsfs.GCSFileSystem(project= "tiger-mle", token="/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json")
    with orders_fs.open(orders_history_path) as orders:
        df_order = pd.read_csv(orders)


    # Loading products data
    products_fs = gcsfs.GCSFileSystem(project= "tiger-mle", token="/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json")
    with products_fs.open(products_overview_path) as products:
        df_product_overview = pd.read_csv(products)


    df_order = transform_order_data(df_order)
    df_customer_overview = transform_customer_overview(df_customer_overview)
    df_product_overview = transform_product_overview(df_product_overview)

    # Joined data
    df_joined = create_joined_data(df_order, df_customer_overview, df_product_overview)

    sales_org = "Argentina"
    country_df_joined = df_joined[df_joined["sales_org"] == sales_org]

    #Save joined data
    country_df_joined.to_csv(path_or_buf=f"gs://pg-explore/data/sample_dataset/intermediate/joined/df_joined_{sales_org}.csv",storage_options={"token":"/mnt/d/PnG/keys/tiger-mle-8c54fa5ce18f.json"})
    logger.info("Load Data Method Ends")