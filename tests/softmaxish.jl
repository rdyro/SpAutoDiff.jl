##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff
using ReverseDiff
hessian, jacobian = ReverseDiff.hessian, ReverseDiff.jacobian 
gradient = ReverseDiff.gradient
##$#############################################################################
##^# tests #####################################################################
function test2()
  x, n, scale = randn(5), 4, 4.312

  fn = SAD.softmaxish
  grad_fn = SAD.softmaxish_jacobian
  hess_fn = SAD.softmaxish_hessian

  g = gradient(x -> fn(x, n; scale = scale), x)'
  display(g)
  g2 = grad_fn(x, n; scale = scale)
  display(collect(g2))
  t = SAD.Tensor(x)
  display(collect(SAD.compute_jacobian(fn(t, n; scale=scale), t)))
  println(repeat("#", 80))
  h = hessian(x -> SAD.softminish(x, n; scale = scale), x)
  display(h)
  h2 = -hess_fn(-x, n; scale = scale)
  display(collect(h2))
end
test2()

function test3()
  x, n, scale = randn(n), 30, 5.23
  s = softmax(x; scale = scale)
  #Ds = -s * s' + diagm(0 => s)
  #
  #J1 = jacobian(x -> maxish(x; scale=scale), x)
  J1 = ReverseDiff.gradient(x -> sum(x .* softmax(x; scale = scale)), x)'
  #J2 = scale * x .* Ds + diagm(0 => s)
  J2 = reshape(s + scale * ((x .* s) - dot(x, s) * s), 1, :)
  display(J1)
  display(J2)
  println(norm((J1 - J2)[:]))

  #J1 = jacobian(
  #  x -> (x' * softmax(x; scale = scale)) .* softmax(x; scale = scale),
  #  x,
  #)
  #J2 = s * (s + scale * Ds * x)' + (x' * s) * scale * Ds
  #display(J1)
  #display(J2)
  #println(norm((J1 - J2)[:]))

  J1 = jacobian(x -> collect(maxish(x, 3; scale = scale)), x)
  J2 = maxish_jacobian(x, 3; scale = scale)
  @views println(norm((J1 - J2)[:]))
end
##$#############################################################################
