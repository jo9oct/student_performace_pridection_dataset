# src/data/load_data.py

import pandas as pd
import os

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw dataset from the data/raw directory.

    Parameters:
    ----------
    filename : str
        Name of the CSV file to load.

    Returns:
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """

    # Get the absolute path of the project root directory
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    # Construct full path to data/raw folder
    data_path = os.path.join(base_path, "data", "raw", filename)

    # Load CSV file
    df = pd.read_csv(data_path)

    return df
