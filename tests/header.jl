#using Revise
#try
#  using SpAutoDiff
#catch
#  using Pkg
#  Pkg.develop(path=joinpath(@__DIR__, ".."))
#  using SpAutoDiff
#end

if !isdefined(Main, :SpAutoDiff)
  include(joinpath(@__DIR__, "../src/SpAutoDiff.jl"))
end
using LinearAlgebra, SparseArrays, Printf, Statistics
using BenchmarkTools#, Debugger, ReverseDiff
SAD = SpAutoDiff
