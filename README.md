<h1 align="center">Sparse Automatic Differentiation – <tt>SpAutoDiff.jl</tt></h1>

<p align="center">
    <img src="media/sparse_matrix.svg" style="max-width:300px; width:100%">
</p>

$$
\begin{aligned}
h(x) &= g(f(x)) \\
& \text{then} \\
\mathtt{D} h(c) &= \left( \mathtt{D} g(b) \right) \left( \mathtt{D} f(c) \right)
\end{aligned}
$$

# Overview

A simple Julia package for automatic differentiation (AD) with focus on sparse
derivatives.

1. What does this package do?
> This package allows computing reverse-mode automatic derivatives via dynamic
computational graph building; Similarly to eager execution PyTorch. As
computations are performed, a small memory footprint graph is built. When the
user requests derivatives.
> This package, when tracing the reverse graph, uses full derivative composition
rules (without sensitivity vectors) and while this is normally not efficient, if
the graph is sparse, the composition can be computed very fast in practice.

2. Why Julia?
> Julia has an excellent sparse linear algebra support directly in main library.
> Julia also provides just-in-time compilation, so even dynamic graph building
gets compiled to efficient machine code.

3. Why sparsity in automatic differentiation or in derivatives?
> For first-order optimization (vast majority of machine learning), the gradient
is dense and its computation via reverse-mode auto differentiation is
computationally on the order of forward computation. However, for (a)
optimization problems in other domains and (b) second-order optimization (e.g.
Newton's method), the Hessian is often sparse and its computation is much more
expensive than forward computation. This package aims to provide a way to
compute sparse derivatives.
> There are alternatives to direct jacobian compositions, as used here, but they
are often rather complicated -- see e.g. *What color is your Jacobian? Graph
coloring for computing derivatives*.

4. Why another AD package?
> Sparsity in the derivatives.

5. Are tensors of dimension > 2 supported?
> Yes, but the package reshapes the tensor to a 2D representation both
internally and in its output.

### Basic Example

In order to take derivatives with respect to a variable, you need to wrap an
input in the `SparseAutoDiff.Tensor` type. Then, you can take derivatives with
respect to that variable using the `SparseAutoDiff.compute_jacobian` function.

```julia
using SparseAutoDiff, LinearAlgebra
const SAD = SparseAutoDiff

n = 100
A = SAD.Tensor(sprandn(n, n, 0.01))
b = SAD.Tensor(randn(n))
c = A * b
# then
J1 = SAD.compute_jacobian(c, b) # compute_jacobian(what, wrt to what)
```
which gives a natively sparse array!
```
100×100 SparseMatrixCSC{Float64, Int64} with 115 stored entries:
⎡⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠐⡀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠂⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠁⠀⠀⠀⠀⠀⠀⠀⠀⠠⎥
⎢⠀⠀⠀⠄⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⎥
⎢⡀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠂⠃⠀⠀⠀⠀⢀⠄⠀⠈⠁⠀⠀⠀⠀⠀⠑⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠐⠀⎥
⎢⠀⠄⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠐⡀⠀⎥
⎢⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠄⠀⠀⠀⠀⠀⠌⠀⠀⠀⠂⠀⠀⠀⠠⠀⠀⠀⢈⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠁⠄⠀⠀⠠⠁⠀⠀⠀⠐⠀⠀⠂⠀⠀⠁⠀⠀⠀⠀⠀⠀⠠⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠄⠐⠐⠀⠀⢀⠐⠀⠀⠀⠀⠀⠂⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠐⠂⠀⠀⠀⠀⠀⠂⢀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠐⠐⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⎥
⎢⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠁⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⡀⠀⠀⠀⠀⠐⠀⡀⠀⠁⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠂⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⎥
⎢⠂⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠠⠀⠀⠀⠀⠀⎥
⎢⠠⠀⠀⠀⠀⠁⠀⠀⠀⠀⠠⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⠀⡀⎥
⎣⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠠⠀⠀⠀⎦
```


# Installation

You can install the package by navigating to the root and issuing
```bash
$ julia
julia> ]
(@v1.x) pkg> dev .
```
or
```bash
$ julia
julia> using Pkg
julia> Pkg.dev(".")
```

# Roadmap

- [x] publish first version
- [x] create basic tests against other auto-diff packages
- [x] realize the Hessian composition graph is a very hard problem (repeated variables)
- [x] create graph for the backward pass (for higher order derivatives): Hessian === Jacobian(Jacobian)
- [x] document the package
- [ ] benchmark (time) against established AutoDiff packages: Zygote, Enzyme, etc.
- [ ] document the package

# References


```
@book{magnus2019matrix,
  title={Matrix differential calculus with applications in statistics and econometrics},
  author={Magnus, Jan R and Neudecker, Heinz},
  year={2019},
  publisher={John Wiley \& Sons}
}

@article{gebremedhin2005color,
  title={What color is your Jacobian? Graph coloring for computing derivatives},
  author={Gebremedhin, Assefaw Hadish and Manne, Fredrik and Pothen, Alex},
  journal={SIAM review},
  volume={47},
  number={4},
  pages={629--705},
  year={2005},
  publisher={SIAM}
}
```

- [PyTorch](https://pytorch.org/)
- [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)
- [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/)
