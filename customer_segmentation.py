# customer_segmentation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Upload Dataset
def upload_data():
    """
    Prompt user to upload a CSV file containing the dataset.
    """
    file_path = input("Enter the full path to your dataset CSV file: ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path and try again.")
        exit()
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

# Step 2: Preprocess the Data
def preprocess_data(df):
    """
    Preprocess the data:
    - Encode categorical variables
    - Standardize numerical variables
    """
    # Encode categorical features
    label_columns = ['Gender', 'City', 'Membership Type', 'Satisfaction Level']
    encoder = LabelEncoder()
    for column in label_columns:
        if column in df.columns:
            df[column] = encoder.fit_transform(df[column])

    # Standardize numerical features
    numerical_columns = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    print("Data preprocessing complete.")
    return df, numerical_columns, label_columns

# Step 3: Determine Optimal Number of Clusters
def find_optimal_clusters(data, max_clusters=10):
    """
    Use the Elbow Method to determine the optimal number of clusters.
    """
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.grid()
    plt.show()

# Step 4: Apply K-Means Clustering
def apply_kmeans(df, features, n_clusters=3):
    """
    Apply K-means clustering and add a 'Cluster' column to the DataFrame.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])
    print(f"K-means clustering applied with {n_clusters} clusters.")
    return df

# Step 5: Visualize the Clusters
def visualize_clusters(df):
    """
    Visualize the clusters using a scatter plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Total Spend',
        y='Age',
        hue='Cluster',
        palette='viridis',
        data=df,
        s=100,
        alpha=0.7
    )
    plt.title('Customer Segments based on Total Spend and Age')
    plt.xlabel('Total Spend (Standardized)')
    plt.ylabel('Age (Standardized)')
    plt.legend(title='Cluster')
    plt.show()

# Main Function
if __name__ == "__main__":
    # Step 1: Upload the Dataset
    df = upload_data()

    # Step 2: Preprocess the data
    df, numerical_features, label_features = preprocess_data(df)

    # Step 3: Determine the optimal number of clusters
    print("Finding the optimal number of clusters...")
    find_optimal_clusters(df[numerical_features + label_features])

    # Step 4: Apply K-means clustering with the chosen number of clusters (e.g., K=3)
    optimal_clusters = 3  # Manually chosen based on the Elbow graph
    df = apply_kmeans(df, numerical_features + label_features, n_clusters=optimal_clusters)

    # Step 5: Visualize the resulting clusters
    visualize_clusters(df)

    # Step 6: Save the results
    output_path = "customer_segmentation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Clustered data saved to {output_path}.")
