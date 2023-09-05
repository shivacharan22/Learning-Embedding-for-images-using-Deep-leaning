import pandas as pd
import cudf

def dataa_loader():
    dataset = pd.read_csv('../input/triplet-data/datas.csv')
    return dataset

def cu_df_loader():
    dataset = cudf.read_csv('../input/images-for-google-comp/embdatakaggoo.csv')
    return dataset