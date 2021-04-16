include("header.jl")

EPS = 1e-7
k_fn(A, b, x) = A' * (A * x - b) + EPS * x
F_fn(A, b) = (A' * A + EPS * I) \ (A' * b)

#DD = SAD.DenseDiffFiniteDiff
DD = SAD.DenseDiffZygote

@testset "Implict Differentiation" begin
  m, n = 3, 1
  global A, b = randn(m, n), randn(m)

  global x = F_fn(A, b)
  @assert norm(k_fn(A, b, x)) < 1e-5
  @test SAD.implicit_1st((x, A) -> k_fn(A, b, x), x, A) ≈
        DD.jacobian_gen(A -> F_fn(A, b))(A) atol = 1e-3

  #global Dppk = DD.hessian_gen(A -> k_fn(A, b, x))(A)
  #global Dzpk = DD.jacobian_gen(x -> DD.jacobian_gen(A -> k_fn(A, b, x))(A))(x)
  #global Dpzk = DD.jacobian_gen(A -> DD.jacobian_gen(x -> k_fn(A, b, x))(x))(A)

  global A_, x_ = SAD.Tensor(A), SAD.Tensor(x)
  global Dppk = SAD.compute_jacobian(
    SAD.compute_jacobian(k_fn(A_, b, x), A_; create_graph = true),
    A_,
  )
  global Dzpk = SAD.compute_jacobian(
    SAD.compute_jacobian(k_fn(A_, b, x_), A_; create_graph = true),
    x_,
  )
  global Dpzk = SAD.compute_jacobian(
    SAD.compute_jacobian(k_fn(A_, b, x_), x_; create_graph = true),
    A_,
  )

  global Dzzk = DD.hessian_gen(x -> k_fn(A, b, x))(x)
  global Dzk = DD.jacobian_gen(x -> k_fn(A, b, x))(x)

  global Dpz = SAD.implicit_1st((x, A) -> k_fn(A, b, x), x, A)
  global Dpz2 = DD.jacobian_gen(A -> F_fn(A, b))(A)
  @test Dpz ≈ Dpz2 atol = 1e-3

  global Dppz = DD.hessian_gen(A -> F_fn(A, b))(A)

  global u, v, w =
    Dppk, Dzpk * Dpz, kron(Dpz', Matrix(I, length(x), length(x))) * Dpzk
  global lhs =
    Dppk +
    Dzpk * Dpz +
    kron(Dpz', Matrix(I, length(x), length(x))) * Dpzk +
    kron(Dpz', Matrix(I, length(x), length(x))) * Dzzk * Dpz

  global Dppz2 = reduce(
    vcat,
    [
      -(Dzk \ lhs[(1 + (i - 1) * length(x)):(i * length(x)), :])
      for i in 1:length(A)
    ],
  )
  #global Dppz3 = zeros(size(Dppz)...)
  #for i in 1:length(A)
  #  lhs_ = [lhs[i, :]'; lhs[length(A) + i, :]']
  #  sol = -(Dzk \ lhs_)
  #  Dppz3[i, :] = sol[1, :]
  #  Dppz3[length(A) + i, :] = sol[2, :]
  #end

  global Dppz4 = -(kron(Dzk', Matrix(I, length(A), length(A))) \ lhs)

  return
end
