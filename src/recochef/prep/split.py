
"""
Module with functionality for splitting and shuffling datasets.
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import murmurhash3_32

from recochef.layers.interactions import Interactions
from recochef.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_TIMESTAMP_COL,
)


def process_split_ratio(ratio):
    """Generate split ratio lists.
    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).
    Returns:
        tuple: a tuple containing
            bool: A boolean variable multi that indicates if the splitting is multi or single.
            list: A list of normalized split ratios.
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


def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.
    Args:
        data (pd.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.
    Returns:
        pd.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    split_by_column, _ = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    rating_filtered = data.groupby(split_by_column).filter(
        lambda x: len(x) >= min_rating
    )
    return rating_filtered


def min_rating_filter_spark(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.
    Args:
        data (spark.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.
    Returns:
        spark.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    split_by_column, split_with_column = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    rating_temp = (
        data.groupBy(split_by_column)
        .agg({split_with_column: "count"})
        .withColumnRenamed("count(" + split_with_column + ")", "n" + split_with_column)
        .where(col("n" + split_with_column) >= min_rating)
    )

    rating_filtered = data.join(broadcast(rating_temp), split_by_column).drop(
        "n" + split_with_column
    )
    return rating_filtered


def _check_min_rating_filter(filter_by, min_rating, col_user, col_item):
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user
    return split_by_column, split_with_column


def split_pandas_data_with_ratios(data, ratios, seed=42, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios
    .. note::
        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.
    Args:
        data (pd.DataFrame): Pandas data frame to be split.
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


def python_random_split(data, ratio=0.75, seed=42):
    """Pandas random splitter. 
    The splitter randomly splits the input data.
    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio 
            of training data set; if it is a list of float numbers, the splitter splits 
            data into several portions corresponding to the split ratios. If a list is 
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        
    Returns:
        list: Splits of the input data as pd.DataFrame.
    """
    multi_split, ratio = process_split_ratio(ratio)

    if multi_split:
        splits = split_pandas_data_with_ratios(data, ratio, shuffle=True, seed=seed)
        splits_new = [x.drop("split_index", axis=1) for x in splits]

        return splits_new
    else:
        return train_test_split(data, test_size=None, train_size=ratio, random_state=seed)


def _do_stratification(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    is_random=True,
    seed=42,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
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

    for name, group in df_grouped:
        group_splits = split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, shuffle=is_random, seed=seed
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


def python_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    """Pandas chronological splitter.
    This function splits data in a chronological manner. That is, for each user / item, the
    split function takes proportions of ratings which is specified by the split ratio(s).
    The split is stratified.
    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
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
        list: Splits of the input data as pd.DataFrame.
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
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    seed=42,
):
    """Pandas stratified splitter.
    
    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.
    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
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
        list: Splits of the input data as pd.DataFrame.
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


def numpy_stratified_split(X, ratio=0.75, seed=42):
    """Split the user/item affinity matrix (sparse matrix) into train and test set matrices while maintaining
    local (i.e. per user) ratios.
    Main points :
    1. In a typical recommender problem, different users rate a different number of items,
    and therefore the user/affinity matrix has a sparse structure with variable number
    of zeroes (unrated items) per row (user). Cutting a total amount of ratings will
    result in a non-homogeneous distribution between train and test set, i.e. some test
    users may have many ratings while other very little if none.
    2. In an unsupervised learning problem, no explicit answer is given. For this reason
    the split needs to be implemented in a different way then in supervised learningself.
    In the latter, one typically split the dataset by rows (by examples), ending up with
    the same number of features but different number of examples in the train/test setself.
    This scheme does not work in the unsupervised case, as part of the rated items needs to
    be used as a test set for fixed number of users.
    Solution:
    1. Instead of cutting a total percentage, for each user we cut a relative ratio of the rated
    items. For example, if user1 has rated 4 items and user2 10, cutting 25% will correspond to
    1 and 2.6 ratings in the test set, approximated as 1 and 3 according to the round() function.
    In this way, the 0.75 ratio is satisfied both locally and globally, preserving the original
    distribution of ratings across the train and test set.
    2. It is easy (and fast) to satisfy this requirements by creating the test via element subtraction
    from the original dataset X. We first create two copies of X; for each user we select a random
    sample of local size ratio (point 1) and erase the remaining ratings, obtaining in this way the
    train set matrix Xtst. The train set matrix is obtained in the opposite way.
    
    Args:
        X (np.array, int): a sparse matrix to be split
        ratio (float): fraction of the entire dataset to constitute the train set
        seed (int): random seed
    Returns:
        np.array, np.array: Xtr is the train set user/item affinity matrix. Xtst is the test set user/item affinity 
            matrix. 
    """

    np.random.seed(seed)  # set the random seed
    test_cut = int((1 - ratio) * 100)  # percentage of ratings to go in the test set

    # initialize train and test set matrices
    Xtr = X.copy()
    Xtst = X.copy()

    # find the number of rated movies per user
    rated = np.sum(Xtr != 0, axis=1)

    # for each user, cut down a test_size% for the test set
    tst = np.around((rated * test_cut) / 100).astype(int)

    for u in range(X.shape[0]):
        # For each user obtain the index of rated movies
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()

        # extract a random subset of size n from the set of rated movies without repetition
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))

        # change the selected rated movies to unrated in the train set
        Xtr[u, idx_tst] = 0
        # set the movies that appear already in the train set as 0
        Xtst[u, idx_train] = 0

    del idx, idx_train, idx_tst

    return Xtr, Xtst


def random_holdout(dataset, perc=0.8, seed=1234):
    """
    Split sequence dataset randomly
    :param dataset: the sequence dataset
    :param perc: the training percentange
    :param seed: the random seed
    :return: the training and test splits
    """
    dataset = dataset.sample(frac=1, random_state=seed)
    nseqs = len(dataset)
    train_size = int(nseqs * perc)
    # split data according to the shuffled index and the holdout size
    train_split = dataset[:train_size]
    test_split = dataset[train_size:]

    return train_split, test_split


def temporal_holdout(dataset, ts_threshold):
    """
    Split sequence dataset using timestamps
    :param dataset: the sequence dataset
    :param ts_threshold: the timestamp from which test sequences will start
    :return: the training and test splits
    """
    train = dataset.loc[dataset['ts'] < ts_threshold]
    test = dataset.loc[dataset['ts'] >= ts_threshold]
    train, test = clean_split(train, test)

    return train, test


def last_session_out_split(data,
                           user_key='user_id',
                           session_key='session_id',
                           time_key='ts'):
    """
    Assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
    last_session = sessions.last()
    train = data[~data.session_id.isin(last_session.values)].copy()
    test = data[data.session_id.isin(last_session.values)].copy()
    train, test = clean_split(train, test)
    return train, test


def clean_split(train, test):
    """
    Remove new items from the test set.
    :param train: The training set.
    :param test: The test set.
    :return: The cleaned training and test sets.
    """
    train_items = set()
    train['sequence'].apply(lambda seq: train_items.update(set(seq)))
    test['sequence'] = test['sequence'].apply(lambda seq: [it for it in seq if it in train_items])
    return train, test


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions,
                         random_state=None):
    """
    Shuffle interactions.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(interactions.user_ids[shuffle_indices],
                        interactions.item_ids[shuffle_indices],
                        ratings=_index_or_none(interactions.ratings,
                                               shuffle_indices),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  shuffle_indices),
                        weights=_index_or_none(interactions.weights,
                                               shuffle_indices),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions,
                                        random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test


def user_based_train_test_split(interactions,
                                test_percentage=0.2,
                                random_state=None):
    """
    Split interactions between a train and a test set based on
    user ids, so that a given user's entire interaction history
    is either in the train, or the test set.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of users to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    if random_state is None:
        random_state = np.random.RandomState()

    minint = np.iinfo(np.uint32).min
    maxint = np.iinfo(np.uint32).max

    seed = random_state.randint(minint, maxint, dtype=np.int64)

    in_test = ((murmurhash3_32(interactions.user_ids,
                               seed=seed,
                               positive=True) % 100 /
                100.0) <
               test_percentage)
    in_train = np.logical_not(in_test)

    train = Interactions(interactions.user_ids[in_train],
                         interactions.item_ids[in_train],
                         ratings=_index_or_none(interactions.ratings,
                                                in_train),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   in_train),
                         weights=_index_or_none(interactions.weights,
                                                in_train),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[in_test],
                        interactions.item_ids[in_test],
                        ratings=_index_or_none(interactions.ratings,
                                               in_test),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  in_test),
                        weights=_index_or_none(interactions.weights,
                                               in_test),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test


