""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
"""

import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """
import matplotlib.pyplot as plt

def visualize_graph(graph):
    nx.draw_networkx(graph, with_labels=True, font_weight='bold',
                     node_size=1000, arrowsize=20)
    plt.axis('off')
    plt.show()


def _sum_product(graph, cliques):
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # Will populate all messages of the kind messages[child][parent]
    for e in graph.neighbors(root):
        collect_messages(root, e, graph, messages, cliques)

    # Will populate all messages of the kind messages[parent][child]    
    for e in graph.neighbors(root):
        distribute_messages(root, e, graph, messages, cliques)
    
    marginals = []
    # Perform inference on each variable
    for inference_var in range(num_nodes):
        marginal = compute_marginal(graph, messages, inference_var, cliques)
        marginals.append(marginal)

    return marginals


def collect_messages(target, source, graph, messages, jt_cliques):
    # Collect messages recursively from child to parent
    nodes = neighbors_except_origin(graph, source, target)
    for k in nodes:
        collect_messages(source, k, graph, messages, jt_cliques)
    
    send_message(source, target, graph, messages, jt_cliques)


def distribute_messages(source, target, graph, messages, jt_cliques):
    # Distribute messages recursively from parent to child
    send_message(source, target, graph, messages, jt_cliques)
    
    nodes = neighbors_except_origin(graph, target, source)
    for k in nodes:
        distribute_messages(target, k, graph, messages, jt_cliques)


def compute_joint_distribution(factors):
    joint = Factor()
    for i in range(len(factors)):
        joint = factor_product(joint, factors[i])

    return joint

        
def send_message(source, target, graph, messages, jt_cliques):
    joint_factors = []
    
    nodes = neighbors_except_origin(graph, source, target)
    for n in nodes:
        # If a message already exists, use that
        if isinstance(messages[n][source], Factor):
            joint_factors.append(messages[n][source])

    # If unary potential exists, use that        
    if len(graph.nodes[source]) != 0:
        joint_factors.append(graph.nodes[source]['factor'])
    
    joint = compute_joint_distribution(joint_factors) 
    
    # Compute inference on target node
    marginalized_vars = np.array(list(set(joint.var) - set(jt_cliques[target])))
    messages[source][target] = factor_marginalize(joint, marginalized_vars)
    

def compute_marginal(graph, messages, source, cliques):
    joint_factors = []
    
    nodes = graph.neighbors(source)
    for n in nodes:
        # Use all messages incoming into the source
        joint_factors.append(messages[n][source])

    # If unary potential exists, use that     
    if len(graph.nodes[source]) != 0:
        joint_factors.append(graph.nodes[source]['factor'])    

    joint = compute_joint_distribution(joint_factors)

    # Compute inference on target node
    marginalized_vars = np.array(list(set(joint.var) - set(cliques[source])))
    if len(marginalized_vars) > 0:
        return factor_marginalize(joint, marginalized_vars)
    
    return joint


def neighbors_except_origin(graph, source, origin):
    nodes = list(graph.neighbors(source))
    nodes.remove(origin)
    return nodes

""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    updated_factors = []
    for factor in factors:
        factor = factor_evidence(factor, evidence)
        if not factor.is_empty():
            updated_factors.append(factor)

    evidence_vars = list(evidence.keys())
    idx = np.where(query_nodes == evidence)
    query_nodes = np.delete(query_nodes, idx)
    
    updated_edges = []
    for e in edges:
        if e[0] not in evidence_vars and e[1] not in evidence_vars:
            updated_edges.append(e)
    
    updated_edges = np.array(updated_edges)
    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    G = nx.Graph()
    for i in range(len(jt_cliques)):
        G.add_node(i, factor=jt_clique_factors[i])
    for edge in jt_edges:
        G.add_edge(edge[0], edge[1])

    clique_potentials = _sum_product(G, jt_cliques)
    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    import copy 
    from cmath import inf

    query_marginal_probabilities = [Factor() for _ in query_nodes]

    for i, var in enumerate(query_nodes):
        min_card = inf
        clique_potential = Factor()

        for j in range(len(cliques)):
            if (var in cliques[j]) & (clique_potentials[j].card < min_card).any():
                min_card = clique_potentials[j].card
                clique_potential = clique_potentials[j]

        p = copy.deepcopy(clique_potential)
        p.val = p.val / np.sum(p.val) # normalize

        marginalize_vars = list(set(p.var) - set([var]))
        query_marginal_probabilities[i] = factor_marginalize(p, marginalize_vars)
    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
