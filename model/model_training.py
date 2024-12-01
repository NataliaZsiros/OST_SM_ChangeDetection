from influxdb_client import InfluxDBClient
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, cross_val_predict
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import time
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import PCA

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

# using silhouette score for determining the number of clusters
best_score = -1
optimal_k = 1

sample_data = resample(scaled_X, n_samples=int(0.15 * len(scaled_X)), random_state=42)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(sample_data)
    score = silhouette_score(sample_data, kmeans.labels_)
    if score > best_score:
        best_score = score
        optimal_k = k
print('Number of clusters: ', k)

print("Training KMeans")
kmeans = KMeans(n_clusters=k, random_state=42)  
labels = kmeans.fit(scaled_X)

joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans trained and saved!")

#PCA training for the Page-Hinkley algorithm in the stream pipeline
print("PCA:")
pca = PCA(n_components=1)
reduced_X = pca.fit_transform(scaled_X)

joblib.dump(pca, 'pca_model.pkl')
print("PCA trained and saved!")

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
print(f"XGBoost Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

joblib.dump(xgb_model, 'xgb_model.pkl')

#Supervised model - SVM
print("SVM:")
svm_model = SVC()

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print("Hyperparameter tuning...")
svm_random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=param_grid,
    n_iter=50,  
    scoring='accuracy',
    cv=5,
    random_state=0
)

svm_random_search.fit(scaled_X, y)
best_svm_model = svm_random_search.best_estimator_
print("Done: ", best_svm_model)

cv_scores = cross_val_score(best_svm_model, scaled_X, y, cv=5, scoring='accuracy')
print(f"SVM Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

joblib.dump(svm_model, 'svm_model.pkl')
print("SVM model trained and saved!")

# Random Forest

# Find feature importances by training the model on all the features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_

# Select features based on importance
threshold = np.mean(importances)
selected_features = X.columns[importances > threshold]
X_selected = X[selected_features]

# Train with the selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_selected, y)

cv_scores = cross_val_score(rf_selected, X_selected, y, cv=5)
print(f"Random Forest Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

joblib.dump(rf_selected, 'rf_model.pkl')
print("Random Forest trained and saved!")

# Naive Bayes
print("Training Naive Bayes classifier")
nb_model = GaussianNB()

# Hyperparameter tuning for Naive Bayes
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Common hyperparameter for Naive Bayes
}

print("Naive Bayes Hyperparameter tuning...")
random_search_nb = RandomizedSearchCV(
    estimator=nb_model,
    param_distributions=param_grid_nb,
    n_iter=5,
    scoring='accuracy',
    cv=5
)

random_search_nb.fit(scaled_X, y)

best_nb_model = random_search_nb.best_estimator_
print("Best Naive Bayes model: ", best_nb_model)

cv_scores_nb = cross_val_score(best_nb_model, scaled_X, y, cv=5, scoring='accuracy')
print(f"Naive Bayes Cross-validation Accuracy: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std():.4f})")

# Save the Naive Bayes model
joblib.dump(best_nb_model, 'naive_bayes_model.pkl')
print("Naive Bayes model trained and saved!")


#Logistic regression model
print("Logistic regression model:")
logreg_model = LogisticRegression()

# Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', 'l1']
}

print("Hyperparameter tuning...")
grid_search_logreg = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search_logreg.fit(scaled_X, y)
best_logreg_model = grid_search_logreg.best_estimator_
print("Done: ", best_logreg_model)

cv_scores = cross_val_score(best_logreg_model, scaled_X, y, cv=5, scoring='accuracy')
y_pred_cv = cross_val_predict(best_logreg_model, scaled_X, y, cv=5)
print(f"Logistic regression Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Compute metrics
f1 = f1_score(y, y_pred_cv, average='weighted')
precision = precision_score(y, y_pred_cv, average='weighted')
recall = recall_score(y, y_pred_cv, average='weighted')

print(f"Logistic regression Cross-validation F1 Score: {f1:.4f}")
print(f"Logistic regression Cross-validation Precision: {precision:.4f}")
print(f"Logistic regression Cross-validation Recall: {recall:.4f}")

joblib.dump(best_logreg_model, 'logreg_model.pkl')
print("Logistic regression model trained and saved!")

'''
Logistic regression Cross-validation Accuracy: 0.9982 (+/- 0.0005)
Logistic regression Cross-validation F1 Score: 0.9982
Logistic regression Cross-validation Precision: 0.9982
Logistic regression Cross-validation Recall: 0.9982
Logistic regression model trained and saved!
'''

columns_to_keep = ['DstJitter', 'TcpRtt', 'DIntPkt', 'DstPkts', 'DstBytes', 
                   'SynAck', 'DstRate', 'DstLoad', 'sTtl', 'DstLoss', 
                   'SAppBytes', 'pLoss', 'Sport', 'dTtl', 'Dport', 'Target']

df_selected = df[columns_to_keep]

print(df_selected)

X_1 = df_selected.drop(columns=['Target'])
y_1 = df_selected['Target']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_1, y_1)

scaler = MinMaxScaler()

X_normalized = scaler.fit_transform(X_resampled)

logreg_model_1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

print("Hyperparameter tuning...")
grid_search_logreg_1 = GridSearchCV(estimator=logreg_model_1, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search_logreg_1.fit(X_normalized, y_resampled)
best_logreg_model_1 = grid_search_logreg_1.best_estimator_
print("Done: ", best_logreg_model_1)

cv_scores = cross_val_score(best_logreg_model_1, X_normalized, y_resampled, cv=5, scoring='accuracy')
y_pred_cv = cross_val_predict(best_logreg_model_1, X_normalized, y_resampled, cv=5)
print(f"Logistic regression Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Compute metrics
f1 = f1_score(y_resampled, y_pred_cv, average='weighted')
precision = precision_score(y_resampled, y_pred_cv, average='weighted')
recall = recall_score(y_resampled, y_pred_cv, average='weighted')

print(f"Logistic regression Cross-validation F1 Score: {f1:.4f}")
print(f"Logistic regression Cross-validation Precision: {precision:.4f}")
print(f"Logistic regression Cross-validation Recall: {recall:.4f}")


joblib.dump(best_logreg_model_1, 'logreg_model_1.pkl')
print("Logistic regression model trained and saved!")

'''
Logistic regression Cross-validation Accuracy: 0.9850 (+/- 0.0013)
Logistic regression Cross-validation F1 Score: 0.9850
Logistic regression Cross-validation Precision: 0.9851
Logistic regression Cross-validation Recall: 0.9850
Logistic regression model trained and saved!
'''
