import umap.umap_ as umap
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

# Parameters for dimension reduction using UMAP
params_red = {
    2: map_dict(n_components=2, min_dist=0.050, n_neighbors=20, random_state=42, n_jobs=1),
    3: map_dict(n_components=3, min_dist=0.05, n_neighbors=15, random_state=42, n_jobs=1),
}

custom = px.colors.qualitative.Alphabet  # Custom color sequence for plotly plots

def reduce_umapND(X_full, N_r=3, custom={}):
    """
    Reduces high-dimensional data to 2D or 3D using UMAP.

    Args:
        X_full (numpy.ndarray): The input data to be reduced.
        N_r (int, optional): Number of components for reduction (2 or 3). Defaults to 3.
        custom (dict, optional): Custom parameters for UMAP. Defaults to {}.

    Returns:
        numpy.ndarray: The reduced dataset.
    """
    # Initialize UMAP reducer with default or custom parameters
    reducerND = umap.UMAP(**params_red[N_r]) if len(custom) == 0 else umap.UMAP(**custom[N_r])
    X_umapND = reducerND.fit_transform(X_full)  # Perform reduction
    return X_umapND

def build_id(OBJC_desc):
    """
    Creates an ID list based on object descriptions.

    Args:
        OBJC_desc (list): List of object descriptions.

    Returns:
        list: List of IDs corresponding to descriptions.
    """
    # Generate sequential IDs based on length of description list
    ID = [str(i) for i in range(len(OBJC_desc))]
    return ID
    
def run_attach_simple(DF, Desc):
    """
    Attaches a simple ID column to a dataframe based on descriptions.

    Args:
        DF (pandas.DataFrame): DataFrame to update.
        Desc (list): List of descriptions.

    Returns:
        pandas.DataFrame: Updated DataFrame with a new ID column.
    """
    # Add an ID column based on object descriptions
    DF['id'] = build_id(Desc)
    return DF

def print_reduce(X_umapND, Desc, N_r=3):
    """
    Prints scatter plots of reduced data for 2D or 3D views using Plotly.

    Args:
        X_umapND (numpy.ndarray): Reduced data array.
        Desc (list): Descriptions for the data points.
        N_r (int, optional): Number of components (2D or 3D) for plotting. Defaults to 3.
    
    Displays:
        Plotly figure: Interactive scatter plot visualization.
    """
    # Determine plot dimensionality and generate plot
    if N_r == 2:
        DF = pd.DataFrame(X_umapND, columns=['x', 'y'])  # Create DataFrame with 2D coordinates
        DF = run_attach_simple(DF, Desc)  # Attach description IDs to DataFrame
        fig = px.scatter(DF, x='x', y='y', opacity=0.4, color_discrete_sequence=custom, hover_data={'id': ': .2f'})
        fig.update_traces(marker_size=5)
        fig.update_layout(width=1100, height=600)
        fig.show(renderer='colab')  # Display 2D scatter plot
    elif N_r == 3:
        DF = pd.DataFrame(X_umapND, columns=['x', 'y', 'z'])  # Create DataFrame with 3D coordinates
        DF = run_attach_simple(DF, Desc)  # Attach description IDs to DataFrame
        fig = px.scatter_3d(DF, x='x', y='y', z='z', opacity=0.5, color_discrete_sequence=custom, hover_data={'id': ': .2f'})
        fig.update_traces(marker_size=1.5)
        fig.show(renderer='colab')  # Display 3D scatter plot
