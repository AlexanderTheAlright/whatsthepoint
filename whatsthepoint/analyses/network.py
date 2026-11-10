"""
Network Data Preparation Module
================================

Functions for preparing and creating network data structures:
- Edge list creation
- Adjacency matrix creation
- Co-occurrence networks
- Bipartite networks
- Similarity-based networks
- Network from various data formats

Author: whatsthepoint Package
"""

from typing import List, Tuple, Union, Optional, Dict
import warnings

import numpy as np
import pandas as pd
from collections import defaultdict, Counter


# ============================================================================
# Edge List Creation
# ============================================================================

def create_edgelist(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: Optional[str] = None,
    directed: bool = True,
    aggregate_weights: str = 'sum'
) -> pd.DataFrame:
    """
    Create an edge list from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    source_col : str
        Column name for source nodes
    target_col : str
        Column name for target nodes
    weight_col : str, optional
        Column name for edge weights
    directed : bool, default=True
        Whether the network is directed
    aggregate_weights : str, default='sum'
        How to aggregate weights for duplicate edges: 'sum', 'mean', 'max', 'min', 'count'

    Returns
    -------
    pd.DataFrame
        Edge list with columns: source, target, weight (if weight_col provided)
    """
    # Create base edge list
    edgelist = df[[source_col, target_col]].copy()
    edgelist.columns = ['source', 'target']

    # Add weights if provided
    if weight_col is not None:
        edgelist['weight'] = df[weight_col]
    else:
        edgelist['weight'] = 1

    # For undirected networks, sort source/target to avoid duplicates
    if not directed:
        edgelist[['source', 'target']] = pd.DataFrame(
            np.sort(edgelist[['source', 'target']].values, axis=1),
            index=edgelist.index
        )

    # Aggregate duplicate edges
    group_cols = ['source', 'target']

    if aggregate_weights == 'sum':
        edgelist = edgelist.groupby(group_cols, as_index=False)['weight'].sum()
    elif aggregate_weights == 'mean':
        edgelist = edgelist.groupby(group_cols, as_index=False)['weight'].mean()
    elif aggregate_weights == 'max':
        edgelist = edgelist.groupby(group_cols, as_index=False)['weight'].max()
    elif aggregate_weights == 'min':
        edgelist = edgelist.groupby(group_cols, as_index=False)['weight'].min()
    elif aggregate_weights == 'count':
        edgelist = edgelist.groupby(group_cols, as_index=False)['weight'].count()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate_weights}")

    return edgelist


def edgelist_from_pairs(
    pairs: List[Tuple],
    weights: Optional[List[float]] = None,
    directed: bool = True
) -> pd.DataFrame:
    """
    Create edge list from list of node pairs.

    Parameters
    ----------
    pairs : list of tuples
        List of (source, target) tuples
    weights : list of float, optional
        Edge weights
    directed : bool, default=True
        Whether network is directed

    Returns
    -------
    pd.DataFrame
        Edge list
    """
    if weights is None:
        weights = [1] * len(pairs)

    df = pd.DataFrame(pairs, columns=['source', 'target'])
    df['weight'] = weights

    if not directed:
        # Sort pairs for undirected
        df[['source', 'target']] = pd.DataFrame(
            np.sort(df[['source', 'target']].values, axis=1),
            index=df.index
        )
        # Aggregate
        df = df.groupby(['source', 'target'], as_index=False)['weight'].sum()

    return df


# ============================================================================
# Adjacency Matrix
# ============================================================================

def create_adjacency_matrix(
    edgelist: pd.DataFrame,
    node_list: Optional[List] = None,
    weighted: bool = True
) -> pd.DataFrame:
    """
    Create adjacency matrix from edge list.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge list with columns: source, target, weight (optional)
    node_list : list, optional
        List of all nodes (to include isolated nodes)
    weighted : bool, default=True
        If True, use edge weights; if False, binary adjacency

    Returns
    -------
    pd.DataFrame
        Adjacency matrix (nodes x nodes)
    """
    # Get all nodes
    if node_list is None:
        nodes = sorted(set(edgelist['source'].unique()) | set(edgelist['target'].unique()))
    else:
        nodes = sorted(node_list)

    # Initialize matrix
    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype=float)

    # Fill in edges
    for _, row in edgelist.iterrows():
        source = row['source']
        target = row['target']

        if weighted and 'weight' in edgelist.columns:
            weight = row['weight']
        else:
            weight = 1

        adj_matrix.loc[source, target] = weight

    return adj_matrix


def adjacency_to_edgelist(
    adj_matrix: pd.DataFrame,
    threshold: float = 0,
    directed: bool = True
) -> pd.DataFrame:
    """
    Convert adjacency matrix to edge list.

    Parameters
    ----------
    adj_matrix : pd.DataFrame
        Adjacency matrix
    threshold : float, default=0
        Minimum weight threshold for edges
    directed : bool, default=True
        Whether to treat as directed network

    Returns
    -------
    pd.DataFrame
        Edge list
    """
    edges = []

    for i, source in enumerate(adj_matrix.index):
        if directed:
            targets = adj_matrix.columns
        else:
            # For undirected, only take upper triangle
            targets = adj_matrix.columns[i:]

        for target in targets:
            weight = adj_matrix.loc[source, target]

            if weight > threshold:
                if directed or source != target:  # Avoid self-loops in undirected
                    edges.append({
                        'source': source,
                        'target': target,
                        'weight': weight
                    })

    return pd.DataFrame(edges)


# ============================================================================
# Co-occurrence Networks
# ============================================================================

def cooccurrence_network(
    df: pd.DataFrame,
    item_col: str,
    group_col: str,
    min_cooccurrence: int = 1,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Create co-occurrence network from grouped data.

    For example, create a network of words that co-occur in documents,
    or actors that appear in the same movies.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    item_col : str
        Column containing items (e.g., words, actors)
    group_col : str
        Column containing groups (e.g., documents, movies)
    min_cooccurrence : int, default=1
        Minimum co-occurrence threshold
    normalize : bool, default=False
        If True, normalize by number of groups

    Returns
    -------
    pd.DataFrame
        Edge list of co-occurrences
    """
    # Group items by group
    groups = df.groupby(group_col)[item_col].apply(list).values

    # Count co-occurrences
    cooccur = defaultdict(int)

    for items in groups:
        # Get unique items in this group
        unique_items = set(items)

        # Count all pairs
        for item1 in unique_items:
            for item2 in unique_items:
                if item1 < item2:  # Avoid duplicates and self-loops
                    cooccur[(item1, item2)] += 1

    # Create edge list
    edges = []
    for (item1, item2), count in cooccur.items():
        if count >= min_cooccurrence:
            weight = count / len(groups) if normalize else count
            edges.append({
                'source': item1,
                'target': item2,
                'weight': weight,
                'cooccurrence_count': count
            })

    return pd.DataFrame(edges)


def word_cooccurrence_network(
    texts: List[str],
    window_size: int = 5,
    min_cooccurrence: int = 2,
    remove_stopwords: bool = True
) -> pd.DataFrame:
    """
    Create word co-occurrence network from texts.

    Words that appear within a window are connected.

    Parameters
    ----------
    texts : list of str
        List of text documents
    window_size : int, default=5
        Size of the co-occurrence window
    min_cooccurrence : int, default=2
        Minimum co-occurrence threshold
    remove_stopwords : bool, default=True
        Whether to remove stopwords

    Returns
    -------
    pd.DataFrame
        Edge list of word co-occurrences
    """
    from .text_analysis import clean_text, tokenize, remove_stopwords as rm_stopwords

    cooccur = defaultdict(int)

    for text in texts:
        # Clean and tokenize
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)

        if remove_stopwords:
            tokens = rm_stopwords(tokens)

        # Sliding window
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + window_size, len(tokens))):
                word1 = tokens[i]
                word2 = tokens[j]

                # Sort to avoid duplicates
                if word1 < word2:
                    cooccur[(word1, word2)] += 1
                else:
                    cooccur[(word2, word1)] += 1

    # Create edge list
    edges = []
    for (word1, word2), count in cooccur.items():
        if count >= min_cooccurrence:
            edges.append({
                'source': word1,
                'target': word2,
                'weight': count
            })

    return pd.DataFrame(edges)


# ============================================================================
# Bipartite Networks
# ============================================================================

def create_bipartite_edgelist(
    df: pd.DataFrame,
    node_type1_col: str,
    node_type2_col: str,
    weight_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create bipartite network edge list.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    node_type1_col : str
        Column for first node type
    node_type2_col : str
        Column for second node type
    weight_col : str, optional
        Column for edge weights

    Returns
    -------
    pd.DataFrame
        Bipartite edge list
    """
    edgelist = df[[node_type1_col, node_type2_col]].copy()
    edgelist.columns = ['type1_node', 'type2_node']

    if weight_col is not None:
        edgelist['weight'] = df[weight_col]
    else:
        edgelist['weight'] = 1

    # Aggregate
    edgelist = edgelist.groupby(['type1_node', 'type2_node'], as_index=False)['weight'].sum()

    return edgelist


def project_bipartite(
    bipartite_edgelist: pd.DataFrame,
    project_on: str = 'type1',
    weight_method: str = 'simple'
) -> pd.DataFrame:
    """
    Project bipartite network onto one node type.

    Parameters
    ----------
    bipartite_edgelist : pd.DataFrame
        Bipartite edge list with columns: type1_node, type2_node, weight
    project_on : str, default='type1'
        Which node type to project on: 'type1' or 'type2'
    weight_method : str, default='simple'
        Method for calculating projected weights:
        - 'simple': count of shared neighbors
        - 'jaccard': Jaccard coefficient
        - 'weighted': sum of product of weights

    Returns
    -------
    pd.DataFrame
        Projected edge list
    """
    if project_on == 'type1':
        node_col = 'type1_node'
        neighbor_col = 'type2_node'
    elif project_on == 'type2':
        node_col = 'type2_node'
        neighbor_col = 'type1_node'
    else:
        raise ValueError("project_on must be 'type1' or 'type2'")

    # Group neighbors by node
    node_neighbors = bipartite_edgelist.groupby(node_col).apply(
        lambda x: set(x[neighbor_col])
    ).to_dict()

    node_weights = bipartite_edgelist.groupby(node_col).apply(
        lambda x: dict(zip(x[neighbor_col], x['weight']))
    ).to_dict()

    # Calculate projected edges
    edges = []
    nodes = list(node_neighbors.keys())

    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            neighbors1 = node_neighbors[node1]
            neighbors2 = node_neighbors[node2]

            shared = neighbors1.intersection(neighbors2)

            if len(shared) > 0:
                if weight_method == 'simple':
                    weight = len(shared)
                elif weight_method == 'jaccard':
                    union = neighbors1.union(neighbors2)
                    weight = len(shared) / len(union)
                elif weight_method == 'weighted':
                    weight = sum(
                        node_weights[node1].get(n, 0) * node_weights[node2].get(n, 0)
                        for n in shared
                    )
                else:
                    raise ValueError(f"Unknown weight_method: {weight_method}")

                edges.append({
                    'source': node1,
                    'target': node2,
                    'weight': weight,
                    'shared_neighbors': len(shared)
                })

    return pd.DataFrame(edges)


# ============================================================================
# Similarity-based Networks
# ============================================================================

def similarity_network(
    df: pd.DataFrame,
    features: List[str],
    node_id_col: str,
    similarity_metric: str = 'cosine',
    threshold: float = 0.5,
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Create network based on feature similarity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features
    features : list of str
        Column names of features to use for similarity
    node_id_col : str
        Column name for node IDs
    similarity_metric : str, default='cosine'
        Similarity metric: 'cosine', 'euclidean', 'correlation'
    threshold : float, default=0.5
        Minimum similarity threshold for edges
    top_k : int, optional
        For each node, keep only top K most similar neighbors

    Returns
    -------
    pd.DataFrame
        Edge list based on similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from scipy.spatial.distance import pdist, squareform

    # Get feature matrix
    X = df[features].values
    node_ids = df[node_id_col].values

    # Calculate similarity
    if similarity_metric == 'cosine':
        sim_matrix = cosine_similarity(X)
    elif similarity_metric == 'euclidean':
        # Convert distance to similarity
        dist_matrix = euclidean_distances(X)
        max_dist = dist_matrix.max()
        sim_matrix = 1 - (dist_matrix / max_dist) if max_dist > 0 else np.ones_like(dist_matrix)
    elif similarity_metric == 'correlation':
        sim_matrix = np.corrcoef(X)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    # Create edge list
    edges = []
    n_nodes = len(node_ids)

    for i in range(n_nodes):
        # Get similarities for this node
        sims = sim_matrix[i, :]

        if top_k is not None:
            # Get top K (excluding self)
            indices = np.argsort(sims)[::-1]
            indices = [idx for idx in indices if idx != i][:top_k]
        else:
            # Use threshold (excluding self)
            indices = np.where((sims >= threshold) & (np.arange(n_nodes) != i))[0]

        for j in indices:
            if i < j:  # Avoid duplicates
                edges.append({
                    'source': node_ids[i],
                    'target': node_ids[j],
                    'weight': sim_matrix[i, j]
                })

    return pd.DataFrame(edges)


# ============================================================================
# Network from Sequences
# ============================================================================

def transition_network(
    sequences: List[List],
    directed: bool = True,
    weighted: bool = True
) -> pd.DataFrame:
    """
    Create transition network from sequences.

    Edges represent transitions from one state to another.

    Parameters
    ----------
    sequences : list of lists
        List of sequences (e.g., [[A, B, C], [A, C, B]])
    directed : bool, default=True
        Whether network is directed
    weighted : bool, default=True
        If True, weight by transition frequency

    Returns
    -------
    pd.DataFrame
        Edge list of transitions
    """
    transitions = []

    for sequence in sequences:
        for i in range(len(sequence) - 1):
            source = sequence[i]
            target = sequence[i + 1]
            transitions.append((source, target))

    if weighted:
        # Count transitions
        transition_counts = Counter(transitions)
        edges = [
            {'source': s, 'target': t, 'weight': count}
            for (s, t), count in transition_counts.items()
        ]
    else:
        # Unique transitions
        unique_transitions = set(transitions)
        edges = [
            {'source': s, 'target': t, 'weight': 1}
            for s, t in unique_transitions
        ]

    df = pd.DataFrame(edges)

    if not directed:
        # Sort and aggregate for undirected
        df[['source', 'target']] = pd.DataFrame(
            np.sort(df[['source', 'target']].values, axis=1),
            index=df.index
        )
        df = df.groupby(['source', 'target'], as_index=False)['weight'].sum()

    return df


# ============================================================================
# Correlation Networks
# ============================================================================

def correlation_network(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    method: str = 'pearson',
    threshold: float = 0.5,
    absolute: bool = False
) -> pd.DataFrame:
    """
    Create network from correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of str, optional
        Variables to include (if None, uses all numeric columns)
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    threshold : float, default=0.5
        Minimum correlation threshold for edges
    absolute : bool, default=False
        If True, use absolute value of correlation

    Returns
    -------
    pd.DataFrame
        Edge list of correlations
    """
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlations
    corr_matrix = df[variables].corr(method=method)

    # Create edge list
    edges = []

    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            corr = corr_matrix.loc[var1, var2]

            # Apply threshold
            if absolute:
                if abs(corr) >= threshold:
                    edges.append({
                        'source': var1,
                        'target': var2,
                        'weight': abs(corr),
                        'correlation': corr
                    })
            else:
                if corr >= threshold:
                    edges.append({
                        'source': var1,
                        'target': var2,
                        'weight': corr,
                        'correlation': corr
                    })

    return pd.DataFrame(edges)


# ============================================================================
# Network Statistics
# ============================================================================

def network_summary(edgelist: pd.DataFrame, directed: bool = True) -> Dict:
    """
    Calculate basic network statistics from edge list.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge list with source and target columns
    directed : bool, default=True
        Whether network is directed

    Returns
    -------
    dict
        Dictionary of network statistics
    """
    # Get nodes
    nodes = set(edgelist['source'].unique()) | set(edgelist['target'].unique())
    n_nodes = len(nodes)
    n_edges = len(edgelist)

    # Degree distribution
    degree_dist = defaultdict(int)

    for _, row in edgelist.iterrows():
        degree_dist[row['source']] += 1
        if not directed:
            degree_dist[row['target']] += 1
        elif row['source'] != row['target']:
            degree_dist[row['target']] += 1

    degrees = list(degree_dist.values())

    # Calculate statistics
    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
        'avg_degree': np.mean(degrees) if degrees else 0,
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0
    }

    if 'weight' in edgelist.columns:
        stats['total_weight'] = edgelist['weight'].sum()
        stats['avg_weight'] = edgelist['weight'].mean()

    return stats
