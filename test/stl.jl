include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff
using ReverseDiff

using BenchmarkTools

function test()
  t = range(-5; stop = 5, length = 8)
  a = SAD.Tensor(sin.(t))
  b = SAD.Tensor(sin.(t .+ pi / 2))
  f(a, b) = SAD.until(a, b; scale=Inf)
  f(a, b) = reshape(a, :)
  c = f(a, b)
  J = SAD.compute_jacobian(c, a)
  display(collect(SAD.compute_jacobian(c, a)))
  #display(collect(SAD.compute_jacobian(c, b)))
  println(repeat("#", 80))
  display(ReverseDiff.gradient(x -> f(x, b.value), a.value)')
  #display(ReverseDiff.gradient(x -> f(a.value, x), b.value)')

  return
end
test()
