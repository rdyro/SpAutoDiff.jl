module SpAutoDiff

using LinearAlgebra, SparseArrays, Printf, Statistics
using BenchmarkTools

include("types.jl")

include("dense_diff.jl")
include("softmaxish.jl")
include("lsemaxish.jl")

include("rules.jl")
include("graph.jl")

export Tensor

end
