
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

#loading dataset
mcd = pd.read_csv('/mcdonalds.csv')

#display column names
print(mcd.columns)

# dimensions of the dataset
print(mcd.shape)

# first 3 rows of the dataset
print(mcd.head(3))

import pandas as pd
import numpy as np

# Selecting first 11 columns (same as mcdonalds[, 1:11] in R)
MD_x = mcd.iloc[:, 0:11]

# Convert "Yes" to 1 and others to 0
MD_x = (MD_x == "Yes").astype(int)

# Calculate the column means and round them to 2 decimal places
col_means = MD_x.mean().round(2)

# Print the column means
print(col_means)

pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Assuming MD_x is the dataset already processed (with "Yes" as 1 and "No" as 0)
# Initialize and fit PCA
pca = PCA()
MD_pca = pca.fit(MD_x)

# Standard deviation of each principal component
std_devs = np.sqrt(pca.explained_variance_)

# Proportion of variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative proportion of variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)

# Print the output with PC numbers
print(f"{'PC':<6} {'Std Dev':<12} {'Proportion of Variance':<25} {'Cumulative Proportion'}")
for i, (std, var, cum_var) in enumerate(zip(std_devs, explained_variance_ratio, cumulative_variance), start=1):
    print(f"PC{i:<5} {std:<12.5f} {var:<25.5f} {cum_var:.5f}")

# Assuming MD_x is the dataset already processed (with "Yes" as 1 and "No" as 0)
# Initialize and fit PCA
pca = PCA()
MD_pca = pca.fit(MD_x)

# Standard deviations (rounded to 1 decimal place)
std_devs = np.sqrt(pca.explained_variance_)
print("Standard deviations (rounded to 1 decimal place):")
print(np.round(std_devs, 1))

# Rotation matrix (components or loadings)
rotation_matrix = pca.components_

# Create a DataFrame for better readability (with PC names and features)
pc_labels = [f'PC{i+1}' for i in range(rotation_matrix.shape[0])]
features = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']

rotation_df = pd.DataFrame(rotation_matrix.T, columns=pc_labels, index=features)

# Print the rotation matrix rounded to 2 decimal places
print("\nRotation matrix (rounded to 1 decimal place):")
print(rotation_df.round(1))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming MD_x is the dataset already processed (with "Yes" as 1 and "No" as 0)
# Initialize and fit PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Plot PCA scores for the first two components
plt.figure(figsize=(8, 6))
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey', alpha=0.7)
plt.title('PCA Projection: First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Add projection axes (principal component directions)
# Each vector corresponds to the contribution of each feature to PC1 and PC2
components = pca.components_[:2, :]  # Only take the first two components
features = np.array(['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting'])

for i, feature in enumerate(features):
    plt.arrow(0, 0, components[0, i], components[1, i], color='r', head_width=0.02, head_length=0.02)
    plt.text(components[0, i] * 1.1, components[1, i] * 1.1, feature, color='black', ha='center', va='center')

plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set seed for reproducibility
np.random.seed(1234)

# Function to perform k-means clustering and evaluate the best k using silhouette scores
def step_kmeans(data, k_range, nrep):
    best_k = None
    best_score = -1
    best_model = None

    for k in k_range:
        # Run KMeans multiple times
        scores = []
        for _ in range(nrep):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
            kmeans.fit(data)
            score = silhouette_score(data, kmeans.labels_)
            scores.append(score)

        # Average the scores
        avg_score = np.mean(scores)
        print(f'K={k}, Average Silhouette Score: {avg_score:.3f}')

        # Keep track of the best score and corresponding model
        if avg_score > best_score:
            best_k = k
            best_score = avg_score
            best_model = kmeans

    return best_k, best_model

# Perform clustering for k values from 2 to 8
k_range = range(2, 9)
best_k, best_model = step_kmeans(MD_x, k_range, nrep=10)

print(f'Best K: {best_k}')

# Relabel clusters (if needed)
# Note: KMeans labels are already starting from 0, so this is more about ensuring consistency.
labels = best_model.labels_

# Create a DataFrame to see the clusters assigned to each observation
results = pd.DataFrame(MD_x)
results['Cluster'] = labels
print(results.head())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming MD_x is the dataset already processed (with "Yes" as 1 and "No" as 0)

# Set seed for reproducibility
np.random.seed(1234)

# Function to compute WCSS for a range of k values
def compute_wcss(data, k_range):
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # WCSS is stored in inertia_
    return wcss

# Perform WCSS computation for k values from 2 to 8
k_range = range(2, 9)
wcss = compute_wcss(MD_x, k_range)

# Plotting WCSS (Scree Plot)
plt.figure(figsize=(8, 6))
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('Scree Plot: WCSS vs. Number of Segments')
plt.xlabel('Number of Segments (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)  # Ensure all k values are shown on x-axis
plt.grid(True)
plt.show()

# Evaluate stability of segmentation solutions (optional)
# You could implement additional analysis here for stability across replications

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Assuming MD_x is the dataset already processed (with "Yes" as 1 and "No" as 0)

# Set seed for reproducibility
np.random.seed(1234)

def bootstrap_kmeans(data, k_range, nrep, nboot):
    stability_results = {k: [] for k in k_range}

    for k in k_range:
        for _ in range(nboot):
            # Create a bootstrap sample
            bootstrap_sample = data.sample(frac=1, replace=True)

            # Store the ARI scores for this bootstrap sample
            ari_scores = []
            for _ in range(nrep):
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
                kmeans.fit(bootstrap_sample)
                labels = kmeans.labels_

                # Calculate ARI against the true labels (or against the labels from the first run)
                # Here, we would typically compare to a reference clustering,
                # but since we're bootstrapping, let's assume we compare with the first run labels.
                if len(stability_results[k]) == 0:
                    # Run k-means on the original data to get reference labels
                    kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=1234)
                    kmeans_ref.fit(data)
                    reference_labels = kmeans_ref.labels_

                ari = adjusted_rand_score(reference_labels, labels)
                ari_scores.append(ari)

            stability_results[k].append(np.mean(ari_scores))

    return stability_results

# Perform the bootstrap analysis for k values from 2 to 8
k_range = range(2, 9)
nrep = 10
nboot = 100
stability_results = bootstrap_kmeans(MD_x, k_range, nrep, nboot)

# Prepare data for boxplot
ari_values = [stability_results[k] for k in k_range]

# Boxplot of ARI values
plt.figure(figsize=(8, 6))
plt.boxplot(ari_values, labels=k_range)
plt.title('Global Stability Boxplot of Clustering Solutions')
plt.xlabel('Number of Segments (k)')
plt.ylabel('Adjusted Rand Index')
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming MD_x is your dataset (binary matrix with Yes=1, No=0)

# Load the 4-segment solution from your k-means model
kmeans = KMeans(n_clusters=4, random_state=1234).fit(MD_x)
cluster_labels = kmeans.labels_

# Plot histograms for each cluster
for i in range(4):
    plt.figure(figsize=(6, 4))
    plt.hist(MD_x[cluster_labels == i].mean(axis=0), bins=10, range=(0, 1), color='grey', edgecolor='black')
    plt.title(f"Histogram for Cluster {i+1}")
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.grid(True)
    plt.show()

# Load your dataset (assuming the file is in CSV format)
MD_x = pd.read_csv('/mcdonalds.csv')

# Check the data
print(MD_x.head())

MD_k4 = bootstrap_kmeans(MD_x_numeric, [4], nrep=10, nboot=100)

MD_x_numeric = MD_x.select_dtypes(include=[np.number]).copy()

print(MD_x.dtypes)

import numpy as np
import matplotlib.pyplot as plt

# Simulated example of segment stability levels for 4 segments
segment_stability = [0.65, 0.75, 0.90, 0.55]

# Plot the stability of each segment
plt.figure(figsize=(8, 5))
plt.bar(range(1, 5), segment_stability, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
plt.ylim(0, 1)
plt.xlabel('Segment Number')
plt.ylabel('Segment Stability')
plt.title('Segment Level Stability for 4-Segment Solution')
plt.show()

MD_km28 = {
    "2": KMeans(n_clusters=2).fit(MD_x_num),
    "3": KMeans(n_clusters=3).fit(MD_x_num),
    "4": KMeans(n_clusters=4).fit(MD_x_num),
}

MD_x_num = MD_x.select_dtypes(include=[np.number])

MD_k4 = MD_km28["4"].cluster_centers_

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Select a specific value from MD_k4 as the number of clusters
n_clusters = int(MD_k4[0, 0])  # or use MD_k4[1, 0], MD_k4[2, 0], etc.

# Perform clustering using KMeans
MD_r4 = KMeans(n_clusters=n_clusters).fit(MD_x)

# Plot the clustering result
plt.plot(MD_r4.labels_)
plt.ylim(0, 1)
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability")
plt.show()

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Load your data into a DataFrame
# MD_x = pd.read_csv('path_to_your_data.csv')

# Fit Gaussian Mixture Models for 2 to 8 clusters
n_components_range = range(2, 9)
models = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=1234, n_init=10)
    gmm.fit(MD_x)  # Ensure MD_x is preprocessed appropriately
    models.append((n_components, gmm))

# You can extract AIC and BIC values for model comparison
for n_components, model in models:
    print(f'Number of components: {n_components}, AIC: {model.aic(MD_x)}, BIC: {model.bic(MD_x)}')

import matplotlib.pyplot as plt

# Assuming you've stored the AIC, BIC, and ICL values in lists
aic_values = [model.aic(MD_x) for _, model in models]
bic_values = [model.bic(MD_x) for _, model in models]
# For ICL, you'll need to compute it, but it's not directly available in sklearn. You might implement it if needed.

x_labels = list(n_components_range)

plt.figure(figsize=(10, 6))
plt.plot(x_labels, aic_values, marker='o', label='AIC')
plt.plot(x_labels, bic_values, marker='o', label='BIC')
# plt.plot(x_labels, icl_values, marker='o', label='ICL')  # Uncomment if you have ICL values

plt.xlabel('Number of Components (Segments)')
plt.ylabel('Value of Information Criteria (AIC, BIC, ICL)')
plt.title('Information Criteria vs Number of Segments')
plt.legend()
plt.grid()
plt.show()

import pandas as pd

# Example data: Replace these with your actual cluster labels
# kmeans_labels should be a list or array of K-means cluster assignments
# mixture_labels should be a list or array of mixture model cluster assignments
kmeans_labels = [1, 1, 2, 2, 3, 4]  # Example labels for K-means (replace with your data)
mixture_labels = [1, 2, 2, 3, 3, 4]  # Example labels for the mixture model (replace with your data)

# Create a DataFrame for cross-tabulation
cross_tab = pd.crosstab(index=kmeans_labels, columns=mixture_labels, rownames=['kmeans'], colnames=['mixture'])

# Display the cross-tabulation
print(cross_tab)

MD_x = pd.read_csv('/mcdonalds.csv').values  # Load your data and convert to NumPy array

import pandas as pd
from sklearn.mixture import GaussianMixture

# Load your binary matrix (data) from a CSV file
MD_x = pd.read_csv('/mcdonalds.csv')

# Print the unique values in each column to identify non-numeric entries
print(MD_x.apply(lambda col: col.unique()))

# Replace 'Yes'/'No' with 1/0
MD_x.replace({'Yes': 1, 'No': 0}, inplace=True)

# Remove or convert other non-numeric values
# For instance, if there's a text like "I love it!+5", you might want to handle it accordingly
# You could drop those rows or replace them with a numeric score (if appropriate)
MD_x.replace({'I love it!+5': 1}, inplace=True)  # Example replacement; adjust as needed

# After handling known non-numeric values, ensure all remaining data is numeric
try:
    MD_x = MD_x.astype(float)
except ValueError as e:
    print("There are still non-numeric values:", e)

# Convert to NumPy array for fitting the model
MD_x = MD_x.values

# Fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, init_params='kmeans', max_iter=100)
gmm.fit(MD_x)

# Predict the cluster labels from the GMM
mixture_labels = gmm.predict(MD_x)

# Create a DataFrame for cross-tabulation
cross_tab = pd.crosstab(index=kmeans_labels, columns=mixture_labels, rownames=['kmeans'], colnames=['mixture'])

# Display the cross-tabulation
print(cross_tab)

import pandas as pd
from sklearn.mixture import GaussianMixture

# Load your dataset
MD_x = pd.read_csv('/mcdonalds.csv')

# Inspect unique values to find problematic entries
print(MD_x.apply(lambda col: col.unique()))

# Replace 'Yes'/'No' with 1/0
MD_x.replace({'Yes': 1, 'No': 0}, inplace=True)

# Handle other non-numeric values by either replacing or removing them
# Example replacements based on the previous values you encountered
MD_x.replace({'I love it!+5': 1, 'I hate it!-5': -1}, inplace=True)  # Adjust replacements as needed

# After cleaning, check the data types again
print(MD_x.dtypes)

# Ensure all data is numeric
try:
    MD_x = MD_x.astype(float)
except ValueError as e:
    print("There are still non-numeric values:", e)

# Convert to NumPy array for fitting the model
MD_x = MD_x.values


# Display the cross-tabulation
print(cross_tab)

import pandas as pd
from sklearn.mixture import GaussianMixture

# Assuming MD_x is your cleaned dataset (binary matrix) and MD_k4 contains K-means cluster labels
# Load your cleaned dataset
MD_x = pd.read_csv('/mcdonalds.csv')  # Ensure it's numeric
MD_k4 = ...  # Replace with your actual K-means cluster labels


# Create a cross-tabulation of the K-means and mixture model labels
cross_tab = pd.crosstab(index=MD_k4, columns=mixture_labels, rownames=['kmeans'], colnames=['mixture'])

# Display the cross-tabulation
print(cross_tab)

import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming MD_x is your cleaned dataset (binary matrix)
# Fit the GMM with the K-means cluster initialization
gmm_a = GaussianMixture(n_components=4, init_params='kmeans', max_iter=100)
gmm_a.fit(MD_x)

gmm_b = GaussianMixture(n_components=4, init_params='random', max_iter=100)  # Second model for comparison
gmm_b.fit(MD_x)

# Log-likelihood values
log_likelihood_a = gmm_a.score(MD_x) * MD_x.shape[0]  # Multiply by the number of samples
log_likelihood_b = gmm_b.score(MD_x) * MD_x.shape[0]  # For the second model

# Output the log-likelihood values
print(f"log Likelihood for GMM A: {log_likelihood_a} (df=47)")
print(f"log Likelihood for GMM B: {log_likelihood_b} (df=47)")

import pandas as pd

# Assuming MD_x is a NumPy array, convert it to a DataFrame
MD_x_df = pd.DataFrame(MD_x)

# Now you can inspect the DataFrame
print(MD_x_df.head())  # Inspect the first few rows
print(MD_x_df.dtypes)  # Check the data types of each column

import pandas as pd
import numpy as np

# Assuming MD_x is a NumPy array
MD_x = np.array(...)  # Your original array

# Convert to DataFrame
MD_x_df = pd.DataFrame(MD_x)

# Inspect the DataFrame
print(MD_x_df.head())  # Check the first few rows
print(MD_x_df.dtypes)  # Check the data types of each column

# Identify categorical columns
categorical_cols = MD_x_df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

# If you need to preprocess categorical columns
from sklearn.preprocessing import OneHotEncoder

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the categorical columns
if categorical_cols:  # Check if there are any categorical columns
    encoded_cols = encoder.fit_transform(MD_x_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate the encoded columns
    MD_x_df = MD_x_df.drop(categorical_cols, axis=1)
    MD_x_df = pd.concat([MD_x_df, encoded_df], axis=1)

# Now you can proceed to fit the Gaussian Mixture Model

import pandas as pd

# Assuming you have a DataFrame 'mcdonalds' with a column 'Like'
# First, create a mapping for the LIKE values
like_mapping = {
    'I HATE IT!-5': 11,
    '-4': 10,
    '-3': 9,
    '-2': 8,
    '-1': 7,
    '0': 6,
    '+1': 5,
    '+2': 4,
    '+3': 3,
    '+4': 2,
    'I LOVE IT!(+5)': 1
}

# Replace the values in the Like column with their numeric codes
mcdonalds['Like.n'] = mcdonalds['Like'].map(like_mapping)

# Now convert it to the desired numerical variable
mcdonalds['Like.n'] = 6 - mcdonalds['Like.n']

# Display the value counts
print(mcdonalds['Like.n'].value_counts())

print(mcdonalds.dtypes)

# Check for non-numeric values in the columns
non_numeric_rows = mcdonalds[~mcdonalds['Like.n'].apply(lambda x: isinstance(x, (int, float)))]
print(non_numeric_rows)

# Convert Like.n and other relevant columns
mcdonalds['Like.n'] = pd.to_numeric(mcdonalds['Like.n'], errors='coerce')

# Convert the other columns to numeric
features = ['convenient', 'spicy', 'fattening', 'greasy', 'fast',
            'cheap', 'tasty', 'expensive', 'healthy', 'disgusting','VisitFrequency', 'Gender']
for col in features:
    mcdonalds[col] = pd.to_numeric(mcdonalds[col], errors='coerce')

# Drop rows with NaN values
mcdonalds.dropna(subset=['Like.n'] + features, inplace=True)

print(mcdonalds.dtypes)

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming you have already converted your features to numeric and handled any missing values
X = mcdonalds[features]  # Use the relevant feature columns

# Fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=1234, max_iter=100)
gmm.fit(X)

# Get the cluster labels
labels = gmm.predict(X)

# Count the sizes of each cluster
cluster_sizes = np.bincount(labels)
print("Cluster sizes:", cluster_sizes)

print(mcdonalds.head())  # Check the first few rows
print(mcdonalds.columns)  # Check the column names

features = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy',
            'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']

X = mcdonalds[features]  # Adjust the feature list as needed

X = X.dropna()  # Remove rows with missing values

print(X.shape)  # Should return a shape like (n_samples, n_features)

import pandas as pd
MD_ref2 = pd.read_csv(/mcdonalds.csv')
MD_ref2 = MD_ref2.select_dtypes(include=[np.number])

import matplotlib.pyplot as plt
import seaborn as sns

# Assume MD_ref2 is a Pandas DataFrame or NumPy array containing the data

# Create a plot of the mixture model
plt.figure(figsize=(8, 6))
sns.kdeplot(MD_ref2, shade=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Mixture Model Plot")

# Add significance lines (assuming you have a way to calculate significance)
# For example, let's say you have a significance threshold of 0.05
significance_threshold = 0.05
plt.axvline(x=MD_ref2.mean() - significance_threshold, color='r', linestyle='--')
plt.axvline(x=MD_ref2.mean() + significance_threshold, color='r', linestyle='--')
sns.kdeplot(MD_ref2, fill=True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Assuming MD.x is a Pandas DataFrame with the segmentation variables
MD_x = pd.read_csv('/mcdonalds.csv') # load your data here

# Calculate the distance matrix
#dist_matrix = pdist(MD_x.T)
# Select only numerical columns from MD_x
MD_x_num = MD_x.select_dtypes(include=[np.number])

# Calculate the distance matrix
dist_matrix = pdist(MD_x_num.T)

# Perform hierarchical clustering
#MD_vclust = linkage(dist_matrix, method='complete')
print(MD_x_num.shape)
print(MD_x_num.head())  # show the first few rows of MD_x_num
print(dist_matrix.shape)
print(dist_matrix)  # show the content of dist_matrix

# Convert the linkage matrix to a DataFrame with the cluster order
#MD_vclust_order = pd.DataFrame({'order': np.argsort(MD_vclust['leaves_'])})
# Calculate the distance matrix
dist_matrix = pdist(MD_x_num.T)

# Check if dist_matrix is not empty
if dist_matrix.size > 0:
    # Perform hierarchical clustering
    MD_vclust = linkage(dist_matrix, method='complete')

    # Convert the linkage matrix to a DataFrame with the cluster order
    MD_vclust_order = pd.DataFrame({'order': np.argsort(MD_vclust[:, 2])})
else:
    print("Error: dist_matrix is empty. Cannot perform hierarchical clustering.")

# Assuming MD.k4 is a Pandas Series with the marker variables
MD_k4 = pd.Series(...)  # load your data here

# Create a bar chart with the marker variables
plt.figure(figsize=(8, 6))
#plt.bar(range(len(MD_k4)), MD_k4, color=['lightgray' if x else 'darkgray' for x in MD_k4])
print(type(MD_k4))  # check the type of MD_k4
print(MD_k4.head())  # show the first few values of MD_k4
print(MD_k4.dtype)  # check the data type of MD_k4
plt.xlabel("Segment")
plt.ylabel("Variable")
plt.title("Segment Profile Plot with Marker Variables")

# Shade the bars based on the cluster order
#for i, order in enumerate(rev(MD_vclust_order['order'])):
   # plt.bar(i, MD_k4[order], color='lightgray' if MD_k4[order] else 'darkgray')
    # Shade the bars based on the cluster order
#for i, order in enumerate(MD_vclust_order['order'][::-1]):
 #   plt.bar(i, MD_k4[order], color='lightgray' if MD_k4[order] else 'darkgray')
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Calculate the distance matrix
dist_matrix = pdist(rotation_matrix.T, columns=pc_labels, index=features, metric='euclidean')
#dist_matrix = pdist(rotation_matrix.T, columns=pc_labels, index=features, metric='euclidean')

# Perform hierarchical clustering
Z = linkage(dist_matrix, method='ward')

# Get the cluster order
MD_vclust_order = pd.DataFrame({'order': fcluster(Z, t=2, criterion='distance')})

plt.show()

import numpy as np
pca_df['Cluster'] = np.repeat(MD_k4, len(pca_df))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: Replace this with your actual cluster labels and dataset
# k4 = np.array([1, 2, 1, 2, 1])  # Example cluster labels
# mcdonalds = pd.DataFrame({'Like': ['I LOVE IT!(+5)', 'I HATE IT!(-5)', ...]})  # Replace with your actual dataset

# For demonstration, here's how you might define k4 and mcdonalds:
k4 = np.random.choice([1, 2, 3, 4], size=100)  # Replace with your actual cluster labels
mcdonalds = pd.DataFrame({
    'Like': np.random.choice(['I LOVE IT!(+5)', 'I HATE IT!(-5)', 'neutral'], size=100)  # Example values
})

# Create a contingency table
contingency_table = pd.crosstab(index=k4, columns=mcdonalds['Like'])

# Create a mosaic plot
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Mosaic Plot of Clusters vs Like Ratings')
plt.xlabel('Like Ratings')
plt.ylabel('Segment Number')
plt.xticks(rotation=45)  # Rotate x labels for better visibility
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Sample data: Replace this with your actual data
# MD_x = pd.DataFrame(...)  # Your original dataset

# Identify categorical columns
categorical_cols = MD_x.select_dtypes(include=['object']).columns.tolist()

# Convert categorical variables to numeric using OneHotEncoder
encoder = OneHotEncoder(sparse=False)
MD_x_encoded = encoder.fit_transform(MD_x[categorical_cols])
MD_x_encoded = pd.DataFrame(MD_x_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded categorical data with the original numeric data
MD_x_numeric = pd.concat([MD_x.drop(categorical_cols, axis=1), MD_x_encoded], axis=1)

# Standardize the data
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_numeric)

# Perform PCA
pca = PCA(n_components=2)
MD_pca = pca.fit_transform(MD_x_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=MD_pca, columns=['PC1', 'PC2'])
# Assuming k4 contains your cluster labels for K-means
# pca_df['Cluster'] = k4  # Uncomment and replace with your actual cluster labels

# Plot the PCA results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)  # You can color by clusters if needed

# Axes labels
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Clusters')

plt.grid()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Sample data: Replace this with your actual DataFrame
# mcdonalds = pd.DataFrame(...)  # Your original dataset
# k4 = np.array(...)  # Your cluster labels for K-means

# Create a DataFrame for the gender column
mcdonalds['Cluster'] = k4  # Ensure k4 has the same length as mcdonalds

# Create a contingency table
contingency_table = pd.crosstab(mcdonalds['Cluster'], mcdonalds['Like'])

# Create a mosaic plot
mosaic(contingency_table.stack(), title='Mosaic Plot of Clusters vs Like',
       labelizer=lambda k: f'Cluster {k[0]}, {k[1]}',
       gap=0.01)

plt.xlabel('Cluster')
plt.ylabel('Like')
plt.show()

print(mcdonalds.columns)

pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming mcdonalds is your DataFrame and k4 is defined
mcdonalds['Cluster'] = k4  # Add the cluster labels to the DataFrame

# Prepare the features and target variable
X = mcdonalds[['Like']]
y = (mcdonalds['Cluster'] == 3).astype(int)  # Convert to binary

# Encode categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables

# Fit the Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=1234)
tree.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=['Not Cluster 3', 'Cluster 3'], filled=True)
plt.title('Decision Tree for Cluster 3')
plt.show()

import pandas as pd

# Example of k4 as a pandas Series (replace with your actual cluster labels)
k4 = pd.Series(MD_k4)  # MD_k4 should contain your cluster labels

# Ensure VisitFrequency is numeric
mcdonalds['Cluster'] = pd.to_numeric(mcdonalds['Cluster'], errors='coerce')

# Calculate the mean VisitFrequency for each cluster
visit = mcdonalds.groupby(k4)['Cluster'].mean()

# Display the result
print(visit)

print(mcdonalds.columns)

print(mcdonalds.head())

visit = mcdonalds.groupby(k4)['Cluster'].mean()

# Assuming k4 is a list or a Series containing cluster labels for each row in mcdonalds
like = mcdonalds.groupby(k4)['Cluster'].mean()
print(like)

# Assuming k4 is a list or Series with cluster labels
female = (mcdonalds['Gender'] == 'Female').astype(int).groupby(k4).mean()
print(female)

import pandas as pd

# Assuming 'mcdonalds' is your DataFrame
pd.set_option('display.max_columns', None)  # This allows all columns to be displayed
print(mcdonalds.columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming mcdonalds is your DataFrame and k4 is the cluster labels
# Convert VisitFrequency to numeric and calculate means
visit = mcdonalds['Cluster'].astype(float).groupby(k4).mean()

# Calculate mean liking
like = mcdonalds['Cluster'].groupby(k4).mean()

# Convert Gender to numeric and calculate mean for female
female = (mcdonalds['Cluster'] == 'Female').astype(int).groupby(k4).mean()