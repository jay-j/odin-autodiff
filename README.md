# Odin AutoDiff
A basic library for automatic differentiation in Odin. 

This implements [Forward Accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation) by storing a list of operations (including derivatives) that get re-executed to calculate the effect of varying one (or more) inputs on the system. 

The basic building block is a `Var`, which is an arbitray-length vector (including 1) representing constants, inputs, intermediate values, and outputs. `Var` are related by `Op` (operations), such as `add()`. The lists of `Var`s, `Op`s, and backing memory are bundled in a `Graph`

# Usage
```odin
graph : ^Graph = graph_init()
x : Var = var_copy(graph, []f64{3}, "x")
y : Var = var_copy(graph, []f64{4}, "y")

z : Var = mult(x, y, "z")

uncertainty_collect(graph, targets = []Var{z}, knobs = []Var{x,y})
```

There are more examples in the code!
