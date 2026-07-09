# -----------------------------------------------------
# DECODELABS PROJECT 3 - CUSTOMER SEGMENTATION
# -----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv(r'C:\Users\Lucky\Desktop\Decodelabs ds projects\project-3\marketing_campaign.csv',sep='\t')
print('Dataset Shape:', df.shape)
print('\\nFirst 5 Rows:')
print(df.head())
# Removing customer ID
if 'ID' in df.columns:
    df.drop('ID',axis=1,inplace=True)
elif 'Id'in df.columns:
    df.drop("Id",axis=1,inplace=True)
#filling missing values
df['Income'] = df['Income'].fillna(df['Income'].median())
# Converting date column
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
# Creating new features
current_year = pd.Timestamp.now().year
df['Age'] = current_year - df['Year_Birth']
df['Children'] = df['Kidhome'] + df['Teenhome']
df['Total_Spending'] = df['MntWines']
df = pd.get_dummies(
    df,
    columns=['Education', 'Marital_Status'],
    drop_first=True
)
# Removing datetime column
df.drop('Dt_Customer', axis=1, inplace=True)
print('\nProcessed Data:')
print(df.head())
#Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
#applying pca 
# Reducing dimensions while keeping 95% variance
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f'\nComponents needed for 95% variance: {n_components}')
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
#elbow method
inertia = []
K = range(2, 11)
for k in K:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_pca)
    inertia.append(model.inertia_)
plt.figure(figsize=(7,5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
#silhouette score
scores = []
for k in K:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores.append(score)

    print(f'k={k} -> Silhouette Score = {score:.4f}')
best_k = K[np.argmax(scores)]
print(f'\nBest K according to Silhouette Score: {best_k}')
#train the model
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(X_pca)
df['Cluster'] = clusters
#data visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.show()
#analysis
summary = df.groupby('Cluster')[
    ['Income', 'Age', 'Children', 'Total_Spending', 'Recency']
].mean().round(2)
print('\nCluster Summary:')
print(summary)
print('\n===== BUSINESS PERSONAS =====')
avg_income = summary['Income'].mean()
avg_spending = summary['Total_Spending'].mean()
for cluster in summary.index:

    income = summary.loc[cluster, 'Income']
    spending = summary.loc[cluster, 'Total_Spending']
    recency = summary.loc[cluster, 'Recency']
    if spending > avg_spending and income > avg_income:
        persona = 'High-Value Customers'
    elif income > avg_income:
        persona = 'Affluent Customers'
    elif recency < summary['Recency'].mean():
        persona = 'Recently Active Budget Customers'
    else:
        persona = 'Low-Value Customers'

    print(f'Cluster {cluster}: {persona}')
#final o/p
df.to_csv(r'C:\Users\Lucky\Desktop\Decodelabs ds projects\project-3\customer_segmentation_output.csv', index=False)
print('\\nProject Completed Successfully!')
print('Output file saved as customer_segmentation_output.csv')