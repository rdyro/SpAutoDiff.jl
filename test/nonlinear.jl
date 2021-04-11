include(joinpath(@__DIR__, "../src/naked_SpAutoDiff.jl"))
using Debugger, BenchmarkTools, ReverseDiff

function test()
  #a = Tensor(randn(5, 3))
  #J = compute_jacobian(a', a)
  #display(collect(J))
  #f = a'
  #display(a)
  #display(f)
  #J2 = reduce(
  #  vcat,
  #  map(i -> vec(ReverseDiff.gradient(a -> vec(a')[i], a.value)), 1:length(f))',
  #)
  #display(J2)
  #println(norm(vec(J - J2)))
  #return

  # test matrix multiplication
  global a = Tensor(randn(5))
  A = randn(3, 5)
  #f(a) = lsemaxish(vcat(a, a); scale=1.0)
  #f(a) = 3.0 * lsemaxish(vcat(a, A * a); scale=5.32)
  f(a) = dot(a, a)
  global c = f(a)
  display(f(a))
  global val = compute_jacobian(f(a), a)
  display(val)
  display(compute_jacobian(val, a))
  @assert isapprox(
    compute_jacobian(f(a), a),
    reshape(jacobian_gen(f)(a.value), :, length(a.value)),
  )
  return

  global J, H = compute_hessian(f(a), a)
  display(J)
  println(repeat("#", 80))
  display(isa(H, AbstractArray) ? collect(H) : H)

  display(ReverseDiff.hessian(f, a.value))

  return
end
test()
