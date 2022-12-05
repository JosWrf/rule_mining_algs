from math import ceil, floor
from typing import Any, Dict, Tuple
import pandas as pd
from pandas import DataFrame


def partition_intervals(num_intervals: int, attribute: str, db: DataFrame) -> pd.Series:
    """Discretizes a numerical attribute into num_intervals of equal size.

    Args:
        num_intervals (int): Number of intervals for this attribute
        attribute (str): Name of the attribute
        db (DataFrame): Database 

    Returns:
        pd.Series : Series where every ajacent intervals are encoded as consecutive integers.
        The order of the intervals is reflected in the integers.
    """
    return pd.cut(x=db[attribute], bins=num_intervals, labels=[i for i in range(num_intervals)], include_lowest=True, retbins=True)

def partition_categorical(attribute: str, db: DataFrame) -> Dict[int, Any]:
    """Maps the given categorical attribute to consecutive integers. Can also be used for 
    numerical attributes.

    Args:
        attribute (str): Name of the attribute
        db (DataFrame): Database

    Returns:
        Dict[int, Any]: Mapping from category encoded as int to its categorical value
    """
    mapping = dict(zip(db[attribute].astype("category").cat.codes, db[attribute]))
    return mapping

def discretize_values(db: DataFrame, discretization: Dict[str, int]) -> Tuple[Dict[int, Any], DataFrame]:
    """Maps the numerical and quantititative attributes to integers as described in 'Mining Quantitative Association 
    Rules in Large Relational Tables'.

    Args:
        db (DataFrame): Original Database
        discretization (Dict[str, int]): Name of the attribute (pandas column name) and the number of intervals
        for numerical attributes or 0 for categorical attributes and numerical attributes (no intervals)

    Returns:
        Tuple[Dict[int, Any], DataFrame]: Encoded database and the mapping from the consecutive integers back to 
        the interval / value for each attribute.
    """
    attribute_mappings = {}
    # Interval size of 0 indicates numerical and categorical values [no interval]
    for attribute, ival in discretization.items():
        if ival == 0:
            attribute_mappings[attribute] = partition_categorical(attribute, db)
            db[attribute].replace(to_replace=dict(zip(db[attribute], db[attribute].astype("category").cat.codes)), inplace=True)
        else:
            x,y = partition_intervals(ival, attribute, db)
            attribute_mappings[attribute] = {i: (ceil(y[i]), floor(y[i+1]))for i in range(len(y)-1)}
            db[attribute] = x.astype("int")

    return attribute_mappings, db




