include(joinpath(@__DIR__, "header.jl"))
using ReverseDiff, Debugger


function test()
  a = SAD.Tensor(randn(3, 4, 5))
  f(a) = sum(a; dims = (1, 3))
  c = f(a)
  J = SAD.compute_jacobian(c, a)
  display(collect(J))
  f_ = f(a.value)
  J2 = _reduce(
    vcat,
    map(
      i -> reshape(ReverseDiff.gradient(x -> f(x)[i], a.value), 1, :),
      1:length(f_),
    ),
  )
  display(J2)
  display(norm(J2 - J))

  return
end
test()
