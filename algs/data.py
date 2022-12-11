import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def load_store_data() -> pd.DataFrame:
    """ Loads the stored_data.csv file and binarizes the data.

    Returns:
        pd.DataFrame: One-hot encoded store data, where each column 
        corresponds to an item.
    """
    data =[]
    with open("datasets/store_data.csv") as f:
        for line in f:
            transaction = [item.strip() for item in line.strip().rstrip().split(",")]
            data.append(transaction)

    te = TransactionEncoder()
    te_ary = te.fit_transform(data)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df

def load_shroom_data() -> pd.DataFrame:
    """Loads the mushroom dataset and names each column thereby. Further the
    eleventh attribute is dropped since it's missing for roughly a quarter of all
    instances.

    Returns:
        pd.DataFrame: DataFrame storing the categorical value for each attribute, with 
        the stalk-root attribute being dropped.
    """
    names=[
        "label",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attach",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surf-ab-ring",
        "stalk-surface-be-ring",
        "stalk-color-ab-ring",
        "stalk-color-be-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "habitat",
        "population"]

    df = pd.read_csv(
        "datasets/agaricus-lepiota.data",
        names=names, index_col= False
    )

    df["id"] = [i for i in range(len(df))]
    df.set_index("id",inplace=True)

    # Drop the stalk-root attribute since it has unknown values for 2480 instances
    df.drop("stalk-root", axis=1, inplace=True)
    return df
