""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
"""

import os
import json
import numpy as np
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """
from collections import defaultdict

def index_from_sample(factor, evidence, sample):
    assignment = np.zeros(len(factor.var))

    for i in range(len(factor.var)):
        var = factor.var[i]

        if var in evidence.keys():
            assignment[i] = evidence[var]
        else:
            for s in sample:
                if s[0] == var:
                    assignment[i] = s[1]

    return assignment_to_index(assignment, factor.card)


def combine_factor(A, B):
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    out = Factor()
    # Set the variables of the output
    out.var = np.union1d(A.var, B.var)

    # Set the cardinality of the output
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # Initialize the factor values to zero
    out.val = np.zeros(np.prod(out.card))
    return out


def joint_factor(factors):
    result = Factor()
    for factor in factors:
        result = combine_factor(result, factor)

    return result
""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    for node in nodes:
        sample_distribution = proposal_factors[node]
        factor = factor_evidence(sample_distribution, samples)
        samples[node] = np.random.choice(factor.card[0], 1, p=factor.val)[0]
    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    # Update given factors with evidence
    updated_proposal_factors = {}
    for node, factor in proposal_factors.items():
        observed_factor = factor_evidence(factor, evidence)
        if node in observed_factor.var:
            updated_proposal_factors[node] = observed_factor

    nodes = np.array(list(updated_proposal_factors.keys()))
    samples_count = defaultdict(int)
    for _ in range(num_iterations):
        sample = tuple(_sample_step(nodes, updated_proposal_factors).items())
        # Count how many times each sample occurred
        samples_count[sample] += 1

    weights = []
    samples = list(samples_count.keys())
    for sample in samples:
        p = [factor.val[index_from_sample(factor, evidence, sample)] for factor in target_factors.values()]
        q = [factor.val[index_from_sample(factor, evidence, sample)] for factor in updated_proposal_factors.values()]
        weights.append(np.prod(p) / np.prod(q))
    
    weights = np.array(weights)
    # Normalize weights
    weights = weights / np.sum(weights)

    factors = [factor for factor in updated_proposal_factors.values()]
    out = joint_factor(factors)
    for i, sample in enumerate(samples):
        # Multiply sample probability to the no. of times it occurred
        out.val[index_from_sample(out, evidence, sample)] = weights[i] * samples_count[sample]

    # Normalize probability
    out.val /= np.sum(out.val)
    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
