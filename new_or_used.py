"""
Exercise description
--------------------

In the context of Mercadolibre's Marketplace an algorithm is needed to
predict if an item listed in the markeplace is new or used.

Your task to design a machine learning model to predict if an item is new or
used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a
function to read that dataset in `build_dataset`.

For the evaluation you will have to choose an appropiate metric and also
elaborate an argument on why that metric was chosen.

The deliverables are:
    - This file including all the code needed to define and evaluate a model.
    - A text file with a short explanation on the criteria applied to choose
      the metric and the performance achieved on that metric.


"""

import json
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


def pipeline_creation()-> sklearn.compose._column_transformer.ColumnTransformer:

    num_attribs = ["price"]
    cat_attribs = ["shipping","accepts_mercadopago", "status", "listing_type_id", "bin_initial_quantity","bin_sold_quantity", "days_passed"]
    log_pipeline = make_pipeline(
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    num_pipeline = Pipeline([
        ("standardize", MinMaxScaler(feature_range=(-1, 1))),
    ])

    cat_pipeline = Pipeline([
        ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("log", log_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return preprocessing


def make_discrete_0(x:int) -> int:
    if x>0:
        return 1
    else:
        return 0

def make_discrete_1(x:int) -> int:
    if x>1:
        return 1
    else:
        return 0



# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("Data/MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def createlists_dicts(json_i:dict) -> dict:
  dict_organized = {"sold_quantity": json_i.get("sold_quantity"),
  "lat": json_i.get("geolocation").get("latitude"),
  "lon": json_i.get("geolocation").get("longitude"),
  "shipping": json_i.get("shipping").get("free_shipping"),
  "shipping": json_i.get("shipping").get("local_pick_up"),
  "last_updated": json_i.get("last_updated"),
  "official_store_id": json_i.get("official_store_id"),
  "title": json_i.get("title"),
  "initial_quantity": json_i.get("initial_quantity"),
  "sold_quantity": json_i.get("sold_quantity"),
  "differential_pricing": json_i.get("differential_pricing"),
  "accepts_mercadopago": json_i.get("accepts_mercadopago"),
  "Date_created": json_i.get("date_created"),
  "status": json_i.get("status"),
  "start_time": json_i.get("start_time"),
  "condition": json_i.get("condition"),
  "listing_type_id": json_i.get("listing_type_id"),
  "price": json_i.get("price"),
  "country": json_i.get("seller_address").get("country").get("id"),
  }
# date feature
  start_pub = datetime.strptime(json_i["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ")
  start_date = datetime.strptime(json_i["start_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
  diff =  start_pub - start_date
  dict_organized["days_passed"] = make_discrete_0(diff.days)
# bin features
  dict_organized["bin_initial_quantity"] = make_discrete_1(dict_organized["initial_quantity"])
  dict_organized["bin_sold_quantity"] = make_discrete_0(dict_organized["sold_quantity"])


  return dict_organized

def createlists_csv(set_:list, name: str) -> None:

    list_dicst = [createlists_dicts(x) for x in set_]
    df = pd.DataFrame(list_dicst)
    df.to_csv(name, index=False)

    return df
    
if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    # translate jsons into CSVs

    ## EDA and model exploration and evaluation
    list_sets = []
    for set__, name_set in [(X_train, "Data/X_train.csv"), (X_test, "Data/X_test.csv")]:
        res_df = createlists_csv(set__, name_set)
        list_sets.append(res_df)

    # preprocessing
    binary_make = lambda x: 1 if x=="used" else 0

    y_train = np.array([binary_make(x) for x in y_train])
    preprop = pipeline_creation()
    RandomForestClassifier_estimator = make_pipeline(preprop, RandomForestClassifier(criterion = 'log_loss', 
                                                                                     max_features = 18,
                                                                                     n_estimators = 31))
    estimator = RandomForestClassifier_estimator.fit(list_sets[0], y_train)

    # model training the evaluation is in the jupyter notebook
    #saving the model
    joblib.dump(estimator, "saved_model/mercadolibre_challenge.pkl")

    # ...


