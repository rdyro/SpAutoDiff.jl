include(joinpath(@__DIR__, "../src/naked_SpAutoDiff.jl"))
using Debugger, BenchmarkTools

function test()
  # test matrix multiplication
  a, b = Tensor(randn(2, 5)), Tensor(randn(5, 3))
  @assert isapprox(
    compute_jacobian((a * b), a),
    reshape(jacobian_gen(a -> a * b.value)(a.value), :, length(a.value)),
  )
  @assert isapprox(
    compute_jacobian((a * b), b),
    reshape(jacobian_gen(b -> a.value * b)(b.value), :, length(b.value)),
  )

  A = Tensor(randn(3, 3))
  x = Tensor(randn(3))
  v = A * x
  c = sum(v .^ 2)

  global J, H = compute_hessian(c, x)
  display(J)
  display(isa(H, AbstractArray) ? collect(H) : H)

  return
end
test()
