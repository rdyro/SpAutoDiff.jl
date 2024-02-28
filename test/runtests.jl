#using Revise
using SpAutoDiff
using LinearAlgebra, SparseArrays
using Test
SAD = SpAutoDiff

# params #####################################################
N_GLOBAL = 27

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

r = randn(5)
unitary_fns = [
  sum,
  x -> sum(x .^ 2),
  x -> cos.(x),
  x -> x - x,
  x -> 2 * x,
  x -> x[1:div(length(x), 2)],
  x -> x[1:div(length(x), 2)] .^ 3,
  x -> kron(sparse(1.123 * I, 4, 4), reshape(x, 3, 9)),
  x -> kron(reshape(x, 3, 9), sparse(1.123 * I, 4, 4)),
  x -> kron(spdiagm(r), reshape(x, 3, 9)),
  x -> kron(reshape(x, 3, 9), spdiagm(r)),
]

@testset "SpAutoDiff.jl - Jacobian" begin
  n = N_GLOBAL
  x = SAD.Tensor(randn(n))

  for fn in unitary_fns
    @test SAD.compute_jacobian(fn(x), x) ≈ DD.jacobian_gen(fn)(x.value)
  end
end

@testset "SpAutoDiff.jl - Hessian" begin
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
  x -> SAD.always(x .- 0.0; scale = 4.32123),
  x -> SAD.eventually(x .- 0.0; scale = 4.32123),
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

# implicit differentiation ###################################
EPS = 1e-7
k_fn(A, b, x) = A' * (A * x - b) + EPS * x
F_fn(A, b) = (A' * A + EPS * I) \ (A' * b)

@testset "SpAutoDiff.jl - Implict Differentiation" begin
  m, n = 3, 1
  A, b = randn(m, n), randn(m)

  x = F_fn(A, b)
  @assert norm(k_fn(A, b, x)) < 1e-5
  @test SAD.implicit_1st((x, A) -> k_fn(A, b, x), x, A) ≈
        DD.jacobian_gen(A -> F_fn(A, b))(A)
  return
end

return
