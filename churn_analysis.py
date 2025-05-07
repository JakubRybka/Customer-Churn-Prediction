import pandas as pd
import numpy as np
import os
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,  StandardScaler

def load_data():
    path = "data"
    df_info = pd.read_csv(os.path.join(path, "Customer_Info.csv"))
    df_online = pd.read_csv(os.path.join(path, "Online_Services.csv"))
    df_payment = pd.read_csv(os.path.join(path, "Payment_Info.csv"))
    df_status = pd.read_csv(os.path.join(path, "Status_Analysis.csv"))
    df_options = pd.read_csv(os.path.join(path, "Service_Options.csv"))
    df_info = df_info.drop(columns=['under_30','senior_citizen','dependents','partner'])
    df_payment = df_payment.drop(columns=['paperless_billing','payment_method','monthly_ charges','avg_monthly_long_distance_charges','total_refunds','total_extra_data_charges','total_long_distance_charges'])
    df_status = df_status.drop(columns=['satisfaction_score','cltv','customer_status','churn_score','churn_value','churn_category','churn_reason'])
    data_frames = [df_info,df_online,df_payment,df_status]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['customer_id'],
                                                    how='inner'), data_frames)
    df_merged.to_csv("data.csv")
    return df_merged.drop(columns='customer_id')

def learn(df):
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns="churn_label"),df['churn_label'], test_size=0.3, random_state=42)

    numeric_features = ['total_charges', 'total_revenue', 'age','number_of_dependents']

    categorical_features = ['gender','married','phone_service','internet_service','online_security','online_backup','device_protection','premium_tech_support','streaming_tv','streaming_movies','streaming_music','internet_type',
                            'contract']
    numeric_transformer = Pipeline(
        [('scaler', StandardScaler())]
        )
    categorical_transformer = Pipeline([('onehot', OneHotEncoder())]
                                       )
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                                   ("cat", categorical_transformer, categorical_features)])
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(class_weight='balanced'))])
    clf = pipeline.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = np.where(y_test == y_pred, 1, 0).sum()/len(y_test)
    print(acc)
    res = pd.DataFrame(y_test,columns=['churn_label'])
    res['prediction'] = y_pred
    res['accuracy'] = np.where(res['prediction'] ==res['churn_label'],'Classified','Unclassified')
    res.to_csv('acc.csv')


learn(load_data())

