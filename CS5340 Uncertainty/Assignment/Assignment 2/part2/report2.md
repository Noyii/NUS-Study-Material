### CS5340 Lab 2 Part 2: Parameter Learning
```
Name: Niharika Shrivastava
Email: e0954756@u.nus.edu
Student ID: A0254355A
```

1. `_learn_node_parameter_w()`

- Taking a derivative of L wrt to all weights and equating to 0 gives us I+1 equations. We can write this in the form of matrix multiplications to solve it efficiently. Derivation provided in the end.

- Linear equation to be solved is:
```
Ax = B, where

A: Coefficient matrix of (I+1)*(I+1) -> observation of I parents with 1 bias in a square matrix
B: Output matrix of 1 * (I+1) -> observation of node multiplied by observation of each parent
x: Output weight parameters of i * (I+1)
```

2. `_learn_node_parameter_var()`

- Derivation provided in the end.

3. `_get_learned_parameters()`

- Construct a DGM using given nodes and edges.
- For each node, find its parents observations along with the node's observations and learn the weights.
- Learn the variance given the weights and all observations.
- Construct the required dictionary. `weight[0]` --> bias
