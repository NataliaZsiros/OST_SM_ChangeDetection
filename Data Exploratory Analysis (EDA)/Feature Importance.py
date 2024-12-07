import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = 'data/wustl_iiot_2021.csv'
df = pd.read_csv(dataset_path)

print(df.head())

# Remove columns as per the dataset's note and Traffic - the target detail.
columns_to_remove = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Traffic']
df = df.drop(columns=columns_to_remove)

###### Check feature has Gaussian distribution

# 1. Method 1: Histogram with Density Curve
sns.histplot(df['Mean'], kde=True)
plt.show()
#If the curve resembles a bell shape, the feature may follow a Gaussian distribution.

# 2. Method 2: Descriptive Statistics
mean = df['Mean'].mean()
median = df['Mean'].median()

print(f"Mean: {mean}, Median: {median}")
if abs(mean - median) < (0.1 * mean):  # Example threshold
    print("Feature might follow a Gaussian distribution.")
else:
    print("Feature likely does not follow a Gaussian distribution.")
#If Mean ≈ Median, the data might follow a normal distribution.    

# 3. Method 3: Skewness Check
skewness = df['Mean'].skew()
print(f"Skewness: {skewness}")
if abs(skewness) < 0.5:  # Quick threshold
    print("Feature might be Gaussian.")
else:
    print("Feature likely not Gaussian.")
# Indicates symmetry, Close to 0 → Likely Gaussian.    


# Separate features and target variable
X = df.drop('Target', axis=1)
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Convert categorical values to numerical values
lencoder=LabelEncoder()
def label_encode(data):
    data['Proto']=lencoder.fit_transform(data['Proto'])
    #data["Sport"] = lencoder.fit_transform(data["Sport"])
    #data["Dport"] = lencoder.fit_transform(data["Dport"])
    return data

X_train = label_encode(X_train)
X_test = label_encode(X_test)

# Apply SMOTE to balance classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#Since most features doesn't have Gaussian distribution, used normalization.

scaler = MinMaxScaler()

X_train_normalized = scaler.fit_transform(X_train_resampled)
X_test_normalized = scaler.transform(X_test)

X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)

print(X_train_normalized.head())

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfe = RFE(estimator=rf, n_features_to_select=15)
sfs = SequentialFeatureSelector(rf, n_features_to_select=15, direction='forward')
rf.fit(X_train_normalized, y_train_resampled)

# Predict on test set
y_pred = rf.predict(X_test_normalized)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Features by Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print(feature_importance)

X_train_rfe = rfe.fit_transform(X_train_normalized, y_train_resampled)
X_test_rfe = rfe.transform(X_test_normalized)

# Get selected features
selected_features = X.columns[rfe.get_support()]
print("Selected Features:")
print(selected_features)

X_train_sfs = sfs.fit_transform(X_train_normalized, y_train_resampled)
X_test_sfs = sfs.transform(X_test_normalized)

# Get selected features
selected_features = X.columns[sfs.get_support()]
print("Selected Features:")
print(selected_features)