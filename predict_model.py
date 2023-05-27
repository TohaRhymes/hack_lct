import pandas as pd
import numpy as np

from predict_utils import (
    final_cont_features,
    final_cat_features,
    final_features,
    incident_features,
    capital_features,
    event_to_sys,
)
from utils import d2_rename, d2_cols, d1_rename, d1_cols, col_ciphers, rename_ru


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool

from tqdm import tqdm
from datetime import timedelta

from copy import deepcopy

import pickle

imputer_continuous = pickle.load(open(f"models/imputer_continuous.pickle", "rb"))
scaler_continuous = pickle.load(open(f"models/scaler_continuous.pickle", "rb"))
imputer_categorical = pickle.load(open(f"models/imputer_categorical.pickle", "rb"))
imputer_length = pickle.load(open(f"models/imputer_length.pickle", "rb"))

models = dict()

for cur_feat in tqdm(incident_features + capital_features):
    cur_model = pickle.load(open(f"models/{cur_feat}.pickle", "rb"))
    models[cur_feat] = cur_model

predict_features = incident_features + capital_features
length_features = [i.replace("between", "length") for i in predict_features]
date_features = [i.replace("between", "date") for i in predict_features]

int_cols = [
    "year",
    "floors",
    "padiks",
    "walls",
    "break",
    "elev",
    "elev_semiload",
    "elev_load",
    "roof_queue",
    "roof_material",
    "found_type",
    "status",
    "manage",
]



def predict(d1):
    d1_f = d1.copy(deep=True).rename(columns=d1_rename)[d1_cols]
    for col in int_cols:
        d1_f[col] = d1_f[col].astype(pd.Int64Dtype())

    d1_f = d1_f.replace(np.nan, pd.NA)


    en_col_ciphers = dict()
    for c in col_ciphers:
        try:
            en_col_ciphers[d1_rename[c]] = col_ciphers[c]
            en_col_ciphers[d1_rename[c]][pd.NA] = pd.NA
        except KeyError:
            continue
    for col in en_col_ciphers:
        if col in d1_f.columns:
            d1_f[col] = d1_f[col].apply(lambda x: en_col_ciphers[col][x])
    d1_f.head()


    # In[33]:


    full_data = d1_f[['unom']+final_features]
    full_data


    # In[3]:


    for feature in date_features:
        mode_value = pd.to_datetime("today")
        full_data[feature] = mode_value
    for feature in length_features:
        mode_value = imputer_length[feature]
        full_data[feature] = mode_value

    data = full_data.copy(deep=True)
    data = data[['unom']+final_features+length_features+date_features]
    data.index=data.unom
    data.drop(['unom'], axis=1, inplace=True)
    data[final_cont_features] = imputer_continuous.transform(data[final_cont_features])
    data[final_cont_features] = scaler_continuous.transform(data[final_cont_features])
    for feature in final_cat_features:
        mode_value = imputer_categorical[feature]
        data[feature].fillna(mode_value, inplace=True)
        data[feature] = data[feature].replace(np.nan, mode_value)
    data[final_cat_features] = data[final_cat_features].astype(str)
    data


    # In[4]:


    ## predict

    predicted_data = dict()
    for cur_feat in tqdm(incident_features + capital_features):
        model = models[cur_feat]
        data_pool = Pool(data[final_features], cat_features=final_cat_features)
        data[cur_feat] = model.predict(data_pool)
        delta = data[cur_feat].apply(lambda x: timedelta(days=x))
        date_col = cur_feat.replace('between', 'date')
        data[date_col] = pd.to_datetime(data[date_col]) + delta


    # In[21]:


    data['unom']=data.index

    cur_year = pd.to_datetime("today").year
    predicted_data_incidents = dict()
    i = 0
    for key, value in tqdm(data.iterrows(), total=data.shape[0]):
        cur_building = value[['unom','year', 'break', 'roof_material', 'walls', 'found_type', 'status', 'manage']]
        for cur_feat in incident_features:
            cur_row = deepcopy(cur_building)
            cur_row['event'] = cur_feat.replace('_between', '')
            cur_row['sys'] = event_to_sys[cur_row['event']]
            cur_row['date'] = value[cur_feat.replace('between', 'date')]
            cur_row['length'] = value[cur_feat.replace('between', 'length')]
            if cur_row['date'].year==cur_year:
                predicted_data_incidents[i] = cur_row
                i+=1


    # In[22]:


    cur_year = pd.to_datetime("today").year
    predicted_data_capitals = dict()
    i = 0
    for key, value in tqdm(full_data.iterrows(), total=full_data.shape[0]):
        cur_building = value[['unom','year', 'break', 'roof_material', 'walls', 'found_type', 'status', 'manage']]
        for cur_feat in capital_features:
            cur_row = deepcopy(cur_building)
            cur_row['event'] = cur_feat.replace('_between', '')
            cur_row['date'] = value[cur_feat.replace('between', 'date')]
            cur_row['length'] = value[cur_feat.replace('between', 'length')]
            if cur_row['date'].year==cur_year:
                predicted_data_capitals[i] = cur_row
                i+=1


    # In[23]:


    pred_ind = pd.DataFrame(predicted_data_incidents).T[['unom', 'event', 'sys', 'date', 'length', 'year', 'break', 'roof_material', 'walls', 'found_type', 'status', 'manage']]
    pred_cap = pd.DataFrame(predicted_data_capitals).T[['unom', 'event', 'date', 'length', 'year', 'break', 'roof_material', 'walls', 'found_type', 'status', 'manage']]
    return pred_ind, pred_cap