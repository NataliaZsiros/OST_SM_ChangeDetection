from influxdb_client import InfluxDBClient
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import joblib
import xgboost as xgb
import time

url = "http://influxdb:8086"  
token = "9FEx1XT4dRY-7H65r2ByRsz-XTlvaGlMN9itr9fMWxdw_K6TK7n7skk9p-wr55aZ3rf8sWnEZ24fSrwEd7V0qQ=="  
org = "ChangeDetection_org"      
bucket = "ChangeDetection" 
username = "admin"
password = "password" 

print("Connecting to InfluxDB...")

while True:
    try:
        client = InfluxDBClient(url=url, token=token, org=org, username=username, password=password)
        print(client.ping())  
        break
    except Exception as e:
        print(f"Retrying... {e}")
        time.sleep(5)

print("Connected!")

query = '''
    from(bucket: "ChangeDetection")
      |> range(start: -24h)
      |> filter(fn: (r) => r["_measurement"] == "Traffic")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

result = client.query_api().query(query, org=org)

data = []
for table in result:
    for record in table.records:
        data.append(record.values)

df = pd.DataFrame(data)

print("Data loaded:")
print(df.head())

if df.empty:
    print("Dataframe empty. Waiting 10 mins...")
    time.sleep(600) #wait 10 minutes to gather some data from the stream
    print("Time elapsed.")
    result = client.query_api().query(query, org=org)

    data = []
    for table in result:
        for record in table.records:
            data.append(record.values)

    df = pd.DataFrame(data)
    print("Data loaded:")
    print(df.head())

print('Columns: ', df.columns)

#Dropping these columns as they are either influxdb generated or unique values and would undermine ML training 
df.drop(columns=['table', 'result', '_time', '_measurement', '_stop', '_start', 'StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId'], inplace=True)

X = df.drop(columns=['Traffic', 'Target'])
y = df['Target']

#Scaling X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

#Unsupervised model - KMeans
print("Training KMeans")
kmeans = KMeans(n_clusters=6, random_state=42)  
labels = kmeans.fit(scaled_X)

joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans trained and saved!")

#Supervised model - XGBoost
print("XGBoost:")
xgb_model = xgb.XGBClassifier()

param_grid = {
    'n_estimators': [100, 500, 1000, 1500],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.05]
}

print("Hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  
    scoring='accuracy',
    cv=5
)

random_search.fit(scaled_X, y)

best_xgb_model = random_search.best_estimator_
print("Done: ", best_xgb_model)

cv_scores = cross_val_score(best_xgb_model, scaled_X, y, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

joblib.dump(xgb_model, 'xgb_model.pkl')

'''
    TO-DOs: it seems this files re-runs after it finishes and sometimes gets az APIException error 
    which then leads to a KeyError when dropping the columns, however I am not sure but we should 
    catch these errors and make the code wait for a bit

    Questions for consultation:
    - When are the python scripts re-ran by Docker?
'''