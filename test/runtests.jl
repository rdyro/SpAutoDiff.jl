using Revise
using SpAutoDiff
using LinearAlgebra, SparseArrays
using Test
SAD = SpAutoDiff

# params #####################################################
N_GLOBAL = 5

# dense derivatives ##########################################
dd_modules = [
  SAD.DenseDiffReverseDiff,
  SAD.DenseDiffZygote,
  #SAD.DenseDiffFiniteDiff
]
dd_names = [
  "DenseDiffReverseDiff",
  "DenseDiffZygote",
  #"DenseDiffFiniteDiff"
]
for (DD, name) in zip(dd_modules, dd_names)
  @testset "$name" begin
    n = N_GLOBAL
    @test DD.jacobian_gen(x -> x)(randn(n)) ≈ Matrix(1.0 * I, n, n)
    @test DD.jacobian_gen(x -> x)(randn(n, n)) ≈ Matrix(1.0 * I, n * n, n * n)
    @test DD.hessian_gen(x -> sum(x .^ 2))(randn(n)) ≈ Matrix(2.0 * I, n, n)

    Z = zeros(2 * n, n)
    Z[1, 1] = 2.0
    Z[n + 2, 2] = 2.0
    @test DD.hessian_gen(x -> x[1:2] .^ 2)(randn(n)) ≈ Z
  end
end


# derivatives ################################################
DD = SAD.DenseDiffReverseDiff

unitary_fns = [
  sum,
  x -> sum(x .^ 2),
  x -> cos.(x),
  x -> x - x,
  x -> 2 * x,
  x -> x[1:div(length(x), 2)],
  x -> x[1:div(length(x), 2)] .^ 3,
]

@testset "SpAutoDiff.jl - Base - Jacobian" begin
  n = N_GLOBAL
  x = SAD.Tensor(randn(n))

  for fn in unitary_fns
    @test SAD.compute_jacobian(fn(x), x) ≈ DD.jacobian_gen(fn)(x.value)
  end
end

@testset "SpAutoDiff.jl - Base - Hessian" begin
  n = N_GLOBAL
  x = SAD.Tensor(randn(n))

  for fn in unitary_fns
    @test (val = SAD.compute_hessian(fn(x), x)[2];
    val == nothing || isa(val, SparseMatrixCSC))
  end

  for fn in unitary_fns
    @test (val = SAD.compute_hessian(fn(x), x)[2];
    val == nothing ? zeros(length(fn(x)) * length(x), length(x)) : val) ≈
          DD.hessian_gen(fn)(x.value)
  end
end

# STL ########################################################
stl_fns = [
           x -> SAD.always(x .- 0.0), 
           x -> SAD.eventually(x .- 0.0), 
           x -> SAD.always(x .- 0.0; scale=4.32123), 
           x -> SAD.eventually(x .- 0.0; scale=4.32123), 
          ]

@testset "SpAutoDiff.jl - STL" begin
  n = N_GLOBAL
  x = SAD.Tensor(randn(n))

  for fn in stl_fns
    @test SAD.compute_jacobian(fn(x), x) ≈ DD.jacobian_gen(fn)(x.value)
  end

  for fn in stl_fns
    @test SAD.compute_hessian(fn(x), x)[2] ≈ DD.hessian_gen(fn)(x.value)
  end
end

return
