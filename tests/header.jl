#using Revise
#try
#  using SpAutoDiff
#catch
#  using Pkg
#  Pkg.develop(path=joinpath(@__DIR__, ".."))
#  using SpAutoDiff
#end
include(joinpath(@__DIR__, "../src/SpAutoDiff.jl"))
using LinearAlgebra, SparseArrays, Printf, Statistics
using BenchmarkTools#, Debugger, ReverseDiff
