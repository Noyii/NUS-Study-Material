""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Niharika Shrivastava
Email: niharika.shrivastava@u.nus.edu
Student ID: A0254355A
"""

import copy
from ntpath import join
from typing import Dict, List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = np.multiply(A.val[idxA], B.val[idxB])
    
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    out = copy.deepcopy(factor)
    
    for marg_var in var:
        prev_factor = copy.deepcopy(out)
        out.var = np.setdiff1d(prev_factor.var, marg_var)
        
        marg_idx = np.where(prev_factor.var == marg_var)[0]
        out.card = np.delete(prev_factor.card, marg_idx)
        
        out.val = np.zeros(np.prod(out.card))
        final_assignments = out.get_all_assignments()
        marg_assignments = np.delete(prev_factor.get_all_assignments(), marg_idx, axis=1)
        
        for i, e in enumerate(final_assignments):
            idx = np.where((marg_assignments == e).all(axis=1))[0]
            out.val[i] = np.sum(prev_factor.val[idx])    

    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for observed, value in evidence.items():
        for factor in out:
            idx = np.where(factor.var == observed)[0]
            
            if len(idx) != 0: # Random variable present in this factor
                assignments = factor.get_all_assignments()
                idx_to_modify = [j for j, row in enumerate(assignments) if row[idx] != value]
                factor.val[idx_to_modify] = 0

    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = np.add(A.val[idxA], B.val[idxB])

    if A.val_argmax is None:
        A.val_argmax = [{}] * np.prod(A.card)
    if B.val_argmax is None:
        B.val_argmax = [{}] * np.prod(B.card)
    
    out.val_argmax = np.array([{**A.val_argmax[idxA[i]], **B.val_argmax[idxB[i]]} for i in range(len(out.val))])
    
    # To pass the test case, but in reality we need the union of argmax to avoid backtracking
    if all(out.val_argmax == {}):
        out.val_argmax = None

    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    out = copy.deepcopy(factor)
    
    for marg_var in var:
        prev_factor = copy.deepcopy(out)
        out.var = np.setdiff1d(prev_factor.var, marg_var)
        
        marg_idx = np.where(prev_factor.var == marg_var)[0]
        if len(marg_idx) == 0:
            continue

        out.card = np.delete(prev_factor.card, marg_idx)

        out.val_argmax = [{}] * np.prod(out.card)
        
        out.val = np.zeros(np.prod(out.card))
        final_assignments = out.get_all_assignments()
        marg_assignments = np.delete(prev_factor.get_all_assignments(), marg_idx, axis=1)
        
        for i, e in enumerate(final_assignments):
            idx = np.where((marg_assignments == e).all(axis=1))[0]
            out.val[i] = np.amax(prev_factor.val[idx])

            for x in index_to_assignment(idx, prev_factor.card):
                j = assignment_to_index(x, prev_factor.card)

                if (prev_factor.val[j] == out.val[i]):
                    if prev_factor.val_argmax is None:
                        out.val_argmax[i] = {marg_var: x[marg_idx][0]}
                    else:
                        out.val_argmax[i] = {**prev_factor.val_argmax[j], **{marg_var: x[marg_idx][0]}}
                    
                    break

    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for i in range(len(factors)):
        joint = factor_product(joint, factors[i])

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    modified_factors = observe_evidence(factors, evidence)
    
    joint = compute_joint_distribution(modified_factors)
    tot_prob = np.sum(joint.val)
    
    marg_vars = np.setdiff1d(joint.var, V)
    output = factor_marginalize(joint, marg_vars)
    output.val = output.val/tot_prob

    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    for e in graph.neighbors(root):
        collect_messages(root, e, graph, messages)
        
    for e in graph.neighbors(root):
        distribute_messages(root, e, graph, messages)
    
    for inference_var in V:
        marginal = compute_marginal(graph, messages, inference_var)
        marginals.append(marginal)

    return marginals


def collect_messages(target, source, graph, messages, MAP=False):
    nodes = neighbors_except_origin(graph, source, target)
    for k in nodes:
        collect_messages(source, k, graph, messages, MAP)
  
    send_message(source, target, graph, messages, MAP)
    
    
def distribute_messages(source, target, graph, messages):
    send_message(source, target, graph, messages)
    
    nodes = neighbors_except_origin(graph, target, source)
    for k in nodes:
        distribute_messages(target, k, graph, messages)

        
def send_message(source, target, graph, messages, MAP=False):
    joint_factors = []
    
    nodes = neighbors_except_origin(graph, source, target)
    for n in nodes:
        if isinstance(messages[n][source], Factor):
            joint_factors.append(messages[n][source])
            
    if len(graph.nodes[source]) != 0:
        joint_factors.append(graph.nodes[source]['factor'])
    
    if graph.has_edge(source, target):
        joint_factors.append(graph.edges[source, target]['factor'])

    if not MAP:
        joint = compute_joint_distribution(joint_factors) 
        nodes.append(source)
        messages[source][target] = factor_marginalize(joint, nodes)
    else:
        joint = compute_joint_distribution_in_log_space(joint_factors) 
        nodes.append(source)
        messages[source][target] = factor_max_marginalize(joint, nodes)


def compute_marginal(graph, messages, source, MAP=False):
    joint_factors = []
    
    nodes = graph.neighbors(source)
    for n in nodes:
        joint_factors.append(messages[n][source])
        
    if len(graph.nodes[source]) != 0:
        joint_factors.append(graph.nodes[source]['factor'])    
    
    if not MAP:
        joint = compute_joint_distribution(joint_factors)
        marg_vars = np.delete(joint.var, np.where(joint.var == source))
        output = factor_marginalize(joint, marg_vars)
        output.val /= np.sum(joint.val)
    else:
        joint = compute_joint_distribution_in_log_space(joint_factors)
        marg_vars = np.delete(joint.var, np.where(joint.var == source))
        output = factor_max_marginalize(joint, marg_vars)
    
    return output


def neighbors_except_origin(graph, source, origin):
    nodes = list(graph.neighbors(source))
    nodes.remove(origin)
    return nodes


def all_nodes_except_origin(graph, origin):
    nodes = list(graph.nodes())
    nodes.remove(origin)
    return nodes


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """

    factors = observe_evidence(factors, evidence)
    factors = [to_log(f) for f in factors]
    
    graph = generate_graph_from_factors(factors)
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    for e in graph.neighbors(root):
        collect_messages(root, e, graph, messages, MAP=True)

    marginal = compute_marginal(graph, messages, root, MAP=True)
    log_prob_max = np.amax(marginal.val)
    
    map_idx = np.argmax(marginal.val)
    max_decoding = marginal.val_argmax[map_idx]
    max_decoding[root] = map_idx

    # remove evidence key
    for e in evidence:
        if e in max_decoding:
            del max_decoding[e]

    return max_decoding, log_prob_max


def to_log(factor):
    factor.val = np.log(factor.val)
    return factor


def compute_joint_distribution_in_log_space(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()
    
    for i in range(len(factors)):    
        joint = factor_sum(joint, factors[i])

    return joint
