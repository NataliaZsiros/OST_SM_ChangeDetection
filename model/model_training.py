from influxdb_client import InfluxDBClient
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
import time

url = "http://influxdb:8086"  
token = "TxqYqsyBImk-hYtUMvFPZCCZJ8odQSAlDhEJEwRVPw0NYEpBtTrcBLwyQysVMFws50YrDi0YrIbGgnLX5xNKAw=="  
org = "ChangeDetection_org"      
bucket = "ChangeDetection" 
username = "admin"
password = "password" 

while True:
    try:
        client = InfluxDBClient(url=url, token=token, org=org, username=username, password=password)
        print(client.ping())  
        break
    except Exception as e:
        print(f"Retrying... {e}")
        time.sleep(5)


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

if df.empty:
    time.sleep(600) #wait 10 minutes to gather some data from the stream

print('Columns: ', df.columns)

#Dropping these columns as they are either influxdb generated or unique values and would undermine ML training 
df.drop(columns=['table', 'result', '_time', '_measurement', '_stop', '_start', 'StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId'], inplace=True)

X = df.drop(columns=['Traffic', 'Target'])
y = df['Target']

#Scaling X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

#Unsupervised model - KMeans
kmeans = KMeans(n_clusters=6, random_state=42)  
labels = kmeans.fit(scaled_X)

joblib.dump(kmeans, 'kmeans_model.pkl')

#Supervised model - XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01)
xgb_model.fit(X, y)

joblib.dump(xgb_model, 'xgb_model.pkl')

'''
    TO-DOs: it seems this files re-runs after it finishes and sometimes gets az APIException error 
    which then leads to a KeyError when dropping the columns, however I am not sure but we should 
    catch these errors and make the code wait for a bit

    Questions for consultation:
    - When are the python scripts re-ran by Docker?
'''