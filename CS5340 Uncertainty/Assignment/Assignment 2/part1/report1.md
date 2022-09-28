### CS5340 Lab 2 Part 1: Junction Tree Algorithm
```
Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
```

1. `_get_clique_factors()`

- Given a set of factors Ψ ∈ {f1, ... , f𝑘} from an UGM, assign each f𝑘 to a cluster 𝐶_i(𝑘) s.t. Scope(f𝑘) ⊆ 𝐶_i(𝑘).

- Cluster potential is the factor product of all its assigned potentials. Also, factor product of all given factors equals factor product of all cluster potentials


2. `_get_jt_clique_and_edges()`

- Form a graph `G` from the given nodes and edges. This graph is already reconstituted. 

- Find the maximal cliques of `G`. These form the nodes of the junction tree.

- Loop over all the maximal cliques pairwise and find all possible sepsets S_ij s.t. `S_ij = C_i intersection C_j` where `C_i`, `C_j` are separate cliques. Find cardinality of each sepset. A non-zero cardinality means an edge can be created from `C_i` to `C_j` with edge weight as the corresponding sepset cardinality.
```
C_i = {X1}, C_j = {X1, X2, X3}, S_ij = {X1}
Cardinality(S_ij) = 1
```

- This process forms a cluster graph `G_cluster`. Find the maximum spanning tree for `G_cluster` with given edge weights. This gives the desired junction tree.

3. `_update_mrf_w_evidence()`

- Update each factor with the evidence. If factor is empty after evidence, discard it.
- Remove evidence variables from the query nodes
- Remove all edges between nodes that join to evidence nodes.

4. `_get_clique_potentials()`

- Create a junction tree from the given edges and nodes. Its possible that after observing evidence its a junction forest.

- Peform sum-product algorithm on each junction tree. This outputs the clique potentials of all cliques present in that junction tree. The sum-product algo is taken from my Lab1 code.

5. `_get_node_marginal_probabilities()`

- A query node can be present in more than 1 clique. Inference on any of these cliques provides the desired output. However, cliques are of varying sizes and marginalization of large cliques is computationally expensive. 

- Hence, find the smallest clique with the query node for each query and perform inference on them. 
