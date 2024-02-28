#using Revise
using Test
using LinearAlgebra, SparseArrays, Printf, Statistics
using BenchmarkTools, Debugger

using SpAutoDiff
SAD = SpAutoDiff

#using Revise
#try
#  using SpAutoDiff
#catch
#  using Pkg
#  Pkg.develop(path=joinpath(@__DIR__, ".."))
#  using SpAutoDiff
#end

#if !isdefined(Main, :SpAutoDiff)
#  include(joinpath(@__DIR__, "../src/SpAutoDiff.jl"))
#end
#SAD = SpAutoDiff

#if !isdefined(Main, :SpAutoDiff)
#  include(joinpath(@__DIR__, "../src/naked_SpAutoDiff.jl"))
#  SpAutoDiff = nothing
#end
