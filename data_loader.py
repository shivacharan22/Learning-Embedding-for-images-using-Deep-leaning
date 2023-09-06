import pandas as pd
import cudf

def dataa_loader() -> pd.DataFrame:
    """ 
    Load the dataset from a CSV file.
    
    This function loads the dataset from a CSV file located at '../input/triplet-data/datas.csv'.
    
    Args:
        None

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.

    Example:
        dataset = data_loader()
    """
    dataset = pd.read_csv('../input/triplet-data/datas.csv')
    return dataset

def cu_df_loader() -> cudf.DataFrame:
    """ 
    Load the dataset from a CSV file into a cuDF DataFrame.
    
    This function loads the dataset from a CSV file located at '../input/images-for-google-comp/embdatakaggoo.csv'
    into a cuDF DataFrame.

    Args:
        None

    Returns:
        cudf.DataFrame: A cuDF DataFrame containing the loaded dataset.

    Example:
        dataset = cu_df_loader()
    """
    dataset = cudf.read_csv('../input/images-for-google-comp/embdatakaggoo.csv')
    return dataset