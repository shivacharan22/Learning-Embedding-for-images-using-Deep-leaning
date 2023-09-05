import pandas as pd
from data_loader import dataa_loader,cu_df_loader

def preprocess():
    dataset = dataa_loader()
    dicr = {"galary":2871,"toys":3408,"dishes":12919,"house":17563,"landmarks": 25830}

    for index,i in enumerate(dataset['inputs']):
        if index<2872:
            listw.append((i,dataset.iloc[0:2871,0].sample().iloc[0],dataset.iloc[2872:,0].sample().iloc[0]))
        elif index<3409:
            listw.append((i,dataset.iloc[2872:3408,0].sample().iloc[0],pd.concat([dataset.iloc[0:2871,0] ,dataset.iloc[3409:,0]]).sample().iloc[0]))
        elif index<12920:
            listw.append((i,dataset.iloc[3409:12919,0].sample().iloc[0],pd.concat([dataset.iloc[0:3408,0] ,dataset.iloc[12920:,0]]).sample().iloc[0]))
        elif index<17564:
            listw.append((i,dataset.iloc[12920:17563,0].sample().iloc[0],pd.concat([dataset.iloc[0:12919,0] ,dataset.iloc[17564:,0]]).sample().iloc[0]))
        elif index<=25830:
            listw.append((i,dataset.iloc[17564:25830,0].sample().iloc[0],dataset.iloc[0:17563,0].sample().iloc[0]))

    datasa = dataset.sample(frac=1).reset_index(drop=True)
    return datasa

