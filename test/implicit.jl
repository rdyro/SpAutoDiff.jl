using Debugger

include("header.jl")

EPS = 1e-7

function loss(a, b, x)
  A = reshape(a, length(b), :)
  return sum((A * x - b) .^ 2) + EPS * sum(x .^ 2)
end

function k_fn(a, b, x)
  A = reshape(a, :, length(x))
  return A' * (A  * x - b) + EPS * x
end

function F(a, b)
  A = reshape(a, length(b), :)
  return (A' * A + EPS * I) \ (A' * b)
end

function main()
  m, n = 3, 1
  a = randn(m * n)
  b = randn(m)

  
  x = F(a, b)
  k = k_fn(a, b, x)
  @assert norm(k) < 1e-5
  fn(x, a) = k_fn(a, b, x)

  global x_, a_ = SAD.Tensor(x), SAD.Tensor(a)
  global f_ = fn(x_, a_)
  global Ja_ = SAD.compute_jacobian(f_, a_)

  global ret = SAD.implicit_1st(fn, x, a)
  global Ja = SAD.FiniteDiff.finite_difference_jacobian(a -> F(a, b), a)

  display(k)
  display(ret)

  return
end
main()
