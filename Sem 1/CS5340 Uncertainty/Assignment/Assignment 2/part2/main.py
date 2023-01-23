""" CS5340 Lab 2 Part 2: Parameter Learning
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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """

""" ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I is the number of inputs. Students
    are encouraged to use numpy.linalg.solve() to get the weights. Learns weights for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    num_inputs = 0 if inputs is None else inputs.shape[1]
    weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    # Insert 1 as a bias in each row
    modified_inputs = np.insert(inputs, 0, 1, axis=1) # N * (I+1)

    A = np.matmul(modified_inputs.T, modified_inputs) # (I+1) * (I+1)
    B = np.matmul(outputs, modified_inputs) # 1 * (I+1)

    # Solve Ax = B where weights = x
    weights = np.linalg.solve(A, B.T)
    weights = weights.T # 1 * (I+1)
    """ END YOUR CODE HERE """

    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only. Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.

    """ YOUR CODE HERE """
    N = outputs.size

    # Insert 1 as a bias in each row
    modified_inputs = np.insert(inputs, 0, 1, axis=1) # N * (I+1)
    delta = np.matmul(weights, modified_inputs.T) # 1 * N

    # Calculate variance of N node observations
    var = np.sum(np.square(outputs - delta)) / N
    """ END YOUR CODE HERE """

    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. The nodes are given in
    ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the parent of j
        observations: dictionary of node: observations pair where observations[1] returns a list of
                    observations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    parameters = {}

    """ YOUR CODE HERE """
    # Construct a directed graph 
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        G.add_edge(e[0], e[1])

    for node in nodes:
        outputs = np.array(observations[node]).reshape(1, -1) # 1 * N

        parents = list(G.predecessors(node))
        # Get observation for each parent
        inputs = np.array([observations[parent] for parent in parents]).T # N * I
        
        # If no parent, I = 0
        if inputs.size == 0:
            N = len(outputs.T)
            inputs = inputs.reshape((N, 0))
        
        # Learn the weights for each node
        weights = _learn_node_parameter_w(outputs, inputs)
        # Get the variance for each node
        variance = _learn_node_parameter_var(outputs, weights, inputs)

        # Construct parameters dictionary
        weights = weights.reshape((weights.T.size, ))
        parameter = {
            'variance': variance,
            'bias': weights[0]
        }
        for i, parent in enumerate(parents):
            parameter[parent] = weights[i+1]

        parameters[node] = parameter
    """ END YOUR CODE HERE """

    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
         observation_config = json.load(f)

    nodes = observation_config['nodes']
    edges = observation_config['edges']
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
