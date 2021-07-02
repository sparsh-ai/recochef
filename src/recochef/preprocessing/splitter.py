import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def process_split_ratio(ratio):
    """Generate split ratio lists.
    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).
    Returns:
        tuple:
        - bool: A boolean variable multi that indicates if the splitting is multi or single.
        - list: A list of normalized split ratios.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")

        multi = False
    elif isinstance(ratio, list):
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if math.fsum(ratio) != 1.0:
            ratio = [x / math.fsum(ratio) for x in ratio]

        multi = True
    else:
        raise TypeError("Split ratio should be either float or a list of floats.")

    return multi, ratio


def _get_column_name(name, col_user, col_item):
    if name == "user":
        return col_user
    elif name == "item":
        return col_item
    else:
        raise ValueError("name should be either 'user' or 'item'.")


def split_pandas_data_with_ratios(data, ratios, seed=42, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios
    .. note::
        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.
    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.
    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if math.fsum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits


def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user='USERID',
    col_item='ITEMID',
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.
    Args:
        data (pandas.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.
    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = _get_column_name(
        filter_by, col_user, col_item
    )

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return (
        data
        .groupby(split_by_column)
        .filter(lambda x: len(x) >= min_rating)
    )


def _do_stratification(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    is_random=True,
    seed=42,
    col_user='USERID',
    col_item='ITEMID',
    col_timestamp='TIMESTAMP',
):
    # A few preliminary checks.
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    if col_user not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")

    if col_item not in data.columns:
        raise ValueError("Schema of data not valid. Missing Item Col")

    if not is_random:
        if col_timestamp not in data.columns:
            raise ValueError("Schema of data not valid. Missing Timestamp Col")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter_pandas(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    # Split by each group and aggregate splits together.
    splits = []

    # If it is for chronological splitting, the split will be performed in a random way.
    df_grouped = (
        data.sort_values(col_timestamp).groupby(split_by_column)
        if is_random is False
        else data.groupby(split_by_column)
    )

    for _, group in df_grouped:
        group_splits = split_pandas_data_with_ratios(
            group, ratio, shuffle=is_random, seed=seed
        )

        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)

        splits.append(concat_group_splits)

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [
        splits_all[splits_all["split_index"] == x].drop("split_index", axis=1)
        for x in range(len(ratio))
    ]

    return splits_list


def random_split(data, ratio=0.8, seed=42):
    """Pandas random splitter.
    The splitter randomly splits the input data.
    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio
            of training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    multi_split, ratio = process_split_ratio(ratio)

    if multi_split:
        splits = split_pandas_data_with_ratios(data, ratio, shuffle=True, seed=seed)
        splits_new = [x.drop("split_index", axis=1) for x in splits]

        return splits_new
    else:
        return train_test_split(data, test_size=None, train_size=ratio, random_state=seed)


def python_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user='USERID',
    col_item='ITEMID',
    col_timestamp='TIMESTAMP',
):
    """Pandas chronological splitter.
    This function splits data in a chronological manner. That is, for each user / item, the
    split function takes proportions of ratings which is specified by the split ratio(s).
    The split is stratified.
    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
        col_timestamp (str): column name of timestamps.
    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=col_user,
        col_item=col_item,
        col_timestamp=col_timestamp,
        is_random=False,
    )

def python_stratified_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user='USERID',
    col_item='ITEMID',
    seed=42,
):
    """Pandas stratified splitter.
    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.
    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=col_user,
        col_item=col_item,
        is_random=True,
        seed=seed,
    )