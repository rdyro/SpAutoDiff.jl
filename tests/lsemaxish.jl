##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff
using ReverseDiff
hessian, jacobian = ReverseDiff.hessian, ReverseDiff.jacobian
##$#############################################################################
##^# tests #####################################################################
function test()
  n = 3
  x = randn(n)
  s = softmax(x)
  Ds = -s * s' + diagm(0 => s)

  scale = 4.3234
  f_fn(x) = lsemaxish(x; scale = scale)
  f = f_fn(x)
  display(f)
  println()
  #g = gradient(f_fn, x)[1]'
  #display(g)
  #g2 = lsemaxish_jacobian(x; scale=scale)
  #display(g2)
  println()
  #@btime h = hessian($f_fn, $x)
  h = hessian(f_fn, x)
  display(h)
  #@btime h2 = lsemaxish_hessian($x)
  h2 = lsemaxish_hessian(x; scale = scale)
  display(h2)

  return
end

function test2()
  x, n, scale = randn(5), 2, 3.324
  h = hessian(x -> SAD.lseminish(x, n; scale = scale), x)
  display(h)
  h2 = -SAD.lsemaxish_hessian(-x, n; scale = scale)
  display(collect(h2))
end
test2()

function test3()
  n, scale = 30, 5.23
  x = randn(n)
  s = SAD.softmax(x; scale = scale)
  #Ds = -s * s' + diagm(0 => s)

  #J1 = jacobian(x -> maxish(x; scale=scale), x)
  J1 = gradient(x -> sum(x .* SAD.softmax(x; scale = scale)), x)'
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
