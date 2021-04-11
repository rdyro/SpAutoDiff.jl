using SpAutoDiff
using LinearAlgebra, SparseArrays
using Test
SAD = SpAutoDiff

dd_modules =
  [SAD.DenseDiffZygote, SAD.DenseDiffReverseDiff, SAD.DenseDiffFiniteDiff]
dd_names = ["DenseDiffZygote", "DenseDiffReverseDiff", "DenseDiffFiniteDiff"]
for (DD, name) in zip(dd_modules, dd_names)
  @testset "$name" begin
    n = 3
    @test DD.jacobian_gen(x -> x)(randn(n)) ≈ Matrix(1.0 * I, n, n)
    @test DD.jacobian_gen(x -> x)(randn(n, n)) ≈ Matrix(1.0 * I, n * n, n * n)
    @test DD.hessian_gen(x -> sum(x .^ 2))(randn(n)) ≈ Matrix(2.0 * I, n, n)
  end
end
