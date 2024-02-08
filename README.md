K-Means Clustering with PCA

This repository contains a Python script for performing K-Means clustering on a dataset using PCA (Principal Component Analysis) for dimensionality reduction. The script utilizes popular machine learning libraries such as pandas, seaborn, scikit-learn, and matplotlib.
Getting Started
Prerequisites

Make sure you have the following libraries installed:

    pandas
    seaborn
    scikit-learn
    matplotlib

You can install them using the following:

pip install pandas seaborn scikit-learn matplotlib

Usage

    Clone the repository:

    git clone https://github.com/your-username/your-repo.git
cd your-repo

Modify the file_path variable in the script to point to your dataset.

file_path = 'path/to/your/dataset.csv'

Run the script:

python kmeans_clustering.py

Adjust the script parameters (e.g., number of clusters) based on your requirements.
Description

The script performs the following steps:

    Reads a CSV dataset using pandas.
    Handles NaN values in a specific column using SimpleImputer.
    Encodes categorical columns using Label Encoding.
    Selects relevant columns for clustering.
    Standardizes the data using StandardScaler.
    Applies K-Means clustering with a specified number of clusters.
    Performs PCA for dimensionality reduction.
    Visualizes the clustered data in a 2D scatter plot.

Feel free to customize the script for your specific use case.

Author

[Salim Aouamri]
