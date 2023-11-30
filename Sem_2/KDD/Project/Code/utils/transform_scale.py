import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

CATEGORICAL_COLUMNS = ["flat_type", "region"]
DROP_COLUMNS = ["block", "street_name", "eco_category", "mrt_name", "latitude", "longitude", "pri_sch", "sec_sch", "flat_model", "subzone", "planning_area", "town", "distance_to_IEBP", "distance_to_BN"]
TARGET_VARIABLE_COLUMN = "resale_price"


def transform_scale_df(X_train, X_test):
    """
    We still need to transform various columns (e.g. string -> categorical)
    and scale the numerical columns.
    """
    # Drop columns
    X_train = X_train.drop(columns=DROP_COLUMNS)
    X_test = X_test.drop(columns=DROP_COLUMNS)

    # Scale numerical columns
    scaler = StandardScaler()
    num_df_train = X_train.select_dtypes(include=np.number)
    num_df_test = X_test.select_dtypes(include=np.number)
    # We do not want to scale lease_commence_date as it a continuous variable (new data will have
    # this in increasing order)
    num_df_train = num_df_train.drop(columns=["lease_commence_date"])
    num_df_test = num_df_test.drop(columns=["lease_commence_date"])
    scaled_num_df_train = pd.DataFrame(scaler.fit_transform(num_df_train), columns=num_df_train.columns, index=num_df_train.index)
    scaled_num_df_test = pd.DataFrame(scaler.transform(num_df_test), columns=num_df_test.columns, index=num_df_test.index)

    X_train = X_train.drop(columns=num_df_train.columns)
    X_train = pd.concat([X_train, scaled_num_df_train], axis=1)
    X_test = X_test.drop(columns=num_df_test.columns)
    X_test = pd.concat([X_test, scaled_num_df_test], axis=1)

    # Transform categorical columns using one-hot encoding
    for col in CATEGORICAL_COLUMNS:
        X_train = pd.concat([X_train, pd.get_dummies(X_train[col], prefix=col)], axis=1)
        X_train = X_train.drop(columns=[col])
        X_test = pd.concat([X_test, pd.get_dummies(X_test[col], prefix=col)], axis=1)
        X_test = X_test.drop(columns=[col])

    # Transform datetime columns
    # We only want the year and month. Encode this as a single integer
    # as a year-month ordinal
    X_train["month"] = X_train["month"].dt.year * 12 + X_train["month"].dt.month - 1
    X_train["lease_commence_date"] = X_train["lease_commence_date"] * 12
    X_test["month"] = X_test["month"].dt.year * 12 + X_test["month"].dt.month - 1
    X_test["lease_commence_date"] = X_test["lease_commence_date"] * 12

    return X_train, X_test


def transform_v2_scale_df(X_train):
    """
    We still need to transform various columns (e.g. string -> categorical)
    and scale the numerical columns.
    """
    # Drop columns
    DROP_COLUMNS = ["block", "town", "street_name", "eco_category", "latitude", "longitude", \
                    "subzone", "planning_area", "std_age_f", "std_age_m", "pri_sch", "sec_sch", "mrt_name"]
    X_train = X_train.drop(columns=DROP_COLUMNS)

    # Convert categorical data to numerical
    CATEGORICAL_COLUMNS = ["flat_type", "flat_model", "region"]
    for col in CATEGORICAL_COLUMNS:
        X_train[col] = X_train[col].astype('category')

    cat_columns = X_train.select_dtypes(['category']).columns
    X_train[cat_columns] = X_train[cat_columns].apply(lambda x: x.cat.codes)

    # # Transform datetime columns
    # # We only want the year and month. Encode this as a single integer
    # # as a year-month ordinal
    X_train["month"] = X_train["month"].dt.year * 12 + X_train["month"].dt.month - 1
    X_train["lease_commence_date"] = X_train["lease_commence_date"] * 12
    
    # # Scale numerical columns
    scaler = StandardScaler()
    num_df_train = X_train.select_dtypes(include=np.number)
    scaled_num_df_train = pd.DataFrame(scaler.fit_transform(num_df_train), columns=num_df_train.columns, index=num_df_train.index)

    X_train = X_train.drop(columns=num_df_train.columns)
    X_train = pd.concat([X_train, scaled_num_df_train], axis=1)

    return X_train
