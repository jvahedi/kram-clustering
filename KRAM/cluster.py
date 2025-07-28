import hdbscan
import numpy as np 
import pandas as pd
import plotly.express as px

def map_dict(**args):
    """
    Converts keyword arguments into a dictionary.

    Args:
        **args: Keyword arguments.

    Returns:
        dict: Dictionary containing the keyword arguments and their values.
    """
    # Create and return dictionary from provided arguments
    dct = {key: value for key, value in args.items()}
    return dct

# Parameters for clustering using HDBSCAN
params_clu = {
    2: map_dict(
        min_cluster_size=9,
        cluster_selection_epsilon=0.150,
        min_samples=5,
        cluster_selection_method='eom',
        prediction_data=True
    ),
    3: map_dict(
        min_cluster_size=6,
        cluster_selection_epsilon=0.06,
        cluster_selection_method='eom',
        prediction_data=True
    ),
}

def cluster_hdbscanND(X_umapND, N_c=3, custom={}, return_obj=False, print_prop=False):
    """
    Performs HDBSCAN clustering on reduced data.

    Args:
        X_umapND (numpy.ndarray): Reduced data for clustering.
        N_c (int, optional): Number of clusters. Defaults to 3.
        custom (dict, optional): Custom parameters for clustering. Defaults to {}.
        return_obj (bool, optional): Whether to return the HDBSCAN object. Defaults to False.
        print_prop (bool, optional): Whether to print properties of the clusters. Defaults to False.

    Returns:
        tuple or hdbscan.HDBSCAN: If return_obj is False, returns (labels, probabilities); otherwise, returns HDBSCAN object.
    """
    Z = X_umapND  # Dataset for clustering

    # Initialize HDBSCAN clusterer with default or custom parameters
    db = hdbscan.HDBSCAN(**params_clu[N_c] if len(custom) == 0 else custom).fit(Z)
    
    labels = db.labels_  # Extract cluster labels
    prob = db.probabilities_  # Extract cluster probabilities
    
    if print_prop:  # Optionally print cluster properties
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
    # Return labels and probabilities or the cluster object itself
    return (labels, prob) if not return_obj else db

def build_id(OBJC_desc):
    """
    Creates an ID list based on object descriptions.

    Args:
        OBJC_desc (list): List of object descriptions.

    Returns:
        list: List of IDs corresponding to descriptions.
    """
    # Generate sequential IDs based on description length
    ID = [str(i) for i in range(len(OBJC_desc))]
    return ID

def run_attach(DF, lbl, Desc):
    """
    Attaches several metadata columns to a dataframe based on cluster labels and descriptions.

    Args:
        DF (pandas.DataFrame): DataFrame to update.
        lbl (numpy.ndarray): Array of cluster labels.
        Desc (list): Descriptions for the data points.

    Returns:
        pandas.DataFrame: Updated DataFrame with metadata columns.
    """
    DF['cluster'] = lbl.astype(str)  # Convert labels to string and attach
    DF['id'] = build_id(Desc)  # Generate and attach sequential IDs
    DF['item'] = Desc  # Attach descriptions
    DF['size'] = 0.25 * np.ones(len(DF['id']))  # Attach default size to each point
    return DF

def print_cluster(X_reduceND, labels, Desc, N_c=3):
    """
    Prints scatter plots of clustered data for 2D or 3D views using Plotly.

    Args:
        X_reduceND (numpy.ndarray): Reduced data array.
        labels (numpy.ndarray): Cluster labels for the data points.
        Desc (list): Descriptions for the data points.
        N_c (int, optional): Number of components (2D or 3D) for plotting. Defaults to 3.
    
    Displays:
        Plotly figure: Interactive scatter plot visualization of clusters.
    """
    # Determine plot dimensionality and generate plot
    if N_c == 2:  # 2D plotting
        DF = pd.DataFrame(X_reduceND, columns=['x', 'y'])  # Create DataFrame with 2D coordinates
        DF = run_attach(DF, labels, Desc)  # Attach metadata columns
        fig = px.scatter(DF, x='x', y='y', opacity=0.5, color='cluster', hover_data={'id': ': .2f', 'item': True})
        fig.update_layout(width=1100, height=600)
        fig.update_traces(marker_size=4)
        fig.show(renderer='colab')  # Display 2D scatter plot
    elif N_c == 3:  # 3D plotting
        DF = pd.DataFrame(X_reduceND, columns=['x', 'y', 'z'])  # Create DataFrame with 3D coordinates
        DF = run_attach(DF, labels, Desc)  # Attach metadata columns
        fig = px.scatter_3d(DF, x='x', y='y', z='z', opacity=1.0, color='cluster', hover_data={'id': ': .2f', 'item': True})
        fig.update_traces(marker_size=2)
        fig.show(renderer='colab')  # Display 3D scatter plot
