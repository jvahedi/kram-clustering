# kram-clustering


# Data Processing Package

A versatile toolkit for reducing high-dimensional data, clustering, and text summarization using advanced machine learning models.

## Features
- Data Reduction with UMAP for 2D and 3D.
- Clustering using HDBSCAN.
- Text Summarization and Labeling using language models.

## Installation

To get started, clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
```

## Basic Usage

### Reduce Data

```python
from reduce import reduce_umapND

# Reduce data to 3D
reduced_data = reduce_umapND(X_full, N_r=3)
```

### Cluster Data

```python
from cluster import cluster_hdbscanND

# Cluster reduced data
labels, probabilities = cluster_hdbscanND(reduced_data)
```

### Summarize Text

```python
from interpret import process_topics

# Process topics based on clustering
topics_df = process_topics(labels, data, probabilities)
```

## License

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more information.

## Contact

Feel free to reach out at [Vahedi.john@columbia.edu](mailto:Vahedi.john@columbia.edu) for queries or support.

