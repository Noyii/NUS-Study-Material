""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
"""

import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """

""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    # Loop over all the given factors only once so that none of them are reused
    for f in factors:
        for i in range(len(jt_cliques)):
            # if the scope of a factor potential is a subset of the clique,
            # assign the factor to the clique
            if set(f.var) & set(jt_cliques[i]) == set(f.var):
                clique_factors[i] = factor_product(clique_factors[i], f)
                
                # Break so that the same factor is not assigned to another clique
                break
    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    
    Useful functions: nx.Graph(), nx.find_cliques(), tree.maximum_spanning_edges(algorithm="kruskal").

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    # Create the input graph which is already reconstituted
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        G.add_edge(e[0], e[1])

    # Returns a list of all maximal cliques in the graph
    jt_cliques = list(nx.find_cliques(G))

    # Form the cluster graph
    G_cluster = nx.Graph()

    # Find all possible sepsets and assign weight to each edge
    for i in range(len(jt_cliques)):
        cluster_a = jt_cliques[i] # C_i
        for j in range(i+1, len(jt_cliques)):
            cluster_b = jt_cliques[j] # C_j
            
            # S_ij = C_i intersection C_j
            sepset = set(cluster_a) & set(cluster_b)
            # No. of variables in sepset form the cardinality
            cardinality = len(sepset)
            if cardinality > 0:
                # Assign weight of the edge_ij as cardinality of sepset S_ij
                G_cluster.add_edge(i, j, weight=cardinality)

    # Junction tree is the maximum spanning tree with assigned edge weights
    mst_edges = tree.maximum_spanning_edges(G_cluster, data=False)
    jt_edges = []
    for edge in mst_edges:
        # Each edge of the MST is an edge of the Junction Tree
        jt_edges.append(list(edge))
    
    jt_edges = np.array(jt_edges)
    """ END YOUR CODE HERE """

    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
