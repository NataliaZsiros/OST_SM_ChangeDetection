import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA

dataset_path = 'data/wustl_iiot_2021.csv'
df = pd.read_csv(dataset_path)

print(df.columns)

df.drop(columns=['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId'], inplace=True)

X = df.drop(columns=['Traffic', 'Target'])
y = df['Target']

#Scaling X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

best_score = -1
optimal_k = 1

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_X)
    score = silhouette_score(scaled_X, kmeans.labels_)
    if score > best_score:
        best_score = score
        optimal_k = k
print('Number of clusters: ', k)

inertia_values = []

for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_X)
    inertia_values.append(kmeans.inertia_) 

plt.figure(figsize=(8, 6))
plt.plot(range(1, 30), inertia_values, marker='o', color='b', linestyle='--')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(range(1, 30))  
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=best_score, random_state=42)
kmeans.fit(scaled_X)
labels = kmeans.labels_

# Reduce data dimensions for visualization (if necessary)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_X)

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title('Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.colorbar(scatter, label='Cluster Label')
plt.show()

print(df['SrcAddr'].value_counts())

print(df[df['SrcAddr'] == '192.168.0.20']['DstAddr'].value_counts())

print(df[(df['SrcAddr'] == '192.168.0.20') & (df['DstAddr'] == "224.0.0.252")]["Traffic"].value_counts())

print(df.head())

print(df.sort_values(by='StartTime').head())

print(df.info())

print(df.describe())

print(df.shape)

print(df.isnull().sum().sum())

#checking for duplicates
duplicates=df[df.duplicated()]
if duplicates.empty:
  print('No duplicates found')
else:
  print('Duplicates found')
  print(duplicates)

print(df.columns)
print(df.dtypes)

print(df.select_dtypes(include = ['object']).nunique())

#remove these columns: 'StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 
#as they are unique to the attacks and would expose the type of the attack to the model.
#But let's keep the 'StartTime', 'LastTime' to timestamp to datastream

df.drop(columns=['SrcAddr', 'DstAddr', 'sIpId', 'dIpId'], inplace = True)

#We reviewed the potential features, using Argus tool [5], and chose 41 features that are common in network flows and also change during the attack phases

df_num_cols = df.select_dtypes(include=['float64', 'int64'])

attack_matrix = df_num_cols.corr()
plt.figure(figsize=(10,10))
sns.heatmap(attack_matrix, annot=False, cmap='viridis', linewidths=0.5, cbar = True)
plt.title('Correlation Heatmap of Numeric fields')
plt.show()

attack_correlation = attack_matrix['Target'].abs().sort_values(ascending=False)
print(attack_correlation)

attack_count = (df['Target'] == 1).sum()
normal_count = (df['Target'] == 0).sum()

print("Number of attacks in network trafic:", attack_count)
print("Number of normals in network trafic:", normal_count)

#Proportion of attacks in network traffic
plt.figure(figsize = (10, 6))
is_attack_count = df['Target'].value_counts()
plt.figure(figsize = (12, 8))
plt.pie(is_attack_count, labels = ['No', 'Yes'], autopct="%0.1f%%")
plt.title('Is Attack')
plt.show()
#Conclusion: Imbalanced dataset

# Value to exclude
value_to_exclude = 'normal'

# Filter out the major value
filtered_df = df[df['Traffic'] != value_to_exclude]

category_counts = filtered_df['Traffic'].value_counts()
total_count = category_counts.sum()

legend_labels = [
    f"{category}: {count} ({count / total_count:.1%})"
    for category, count in category_counts.items()
]

# Plot the pie chart
fig, ax = plt.subplots(figsize=(6, 6))
wedges, _ = ax.pie(
    category_counts,
    labels=None,  # No labels on the chart
    startangle=90
)

# Add a legend with counts and proportions
ax.legend(
    wedges,
    legend_labels,
    title="Attack Details",
    loc="center left",
    bbox_to_anchor=(1, 0.5),  # Places legend to the right of the chart
)

# Add a title and display the plot
plt.title(f'Attack Distribution')
plt.show()