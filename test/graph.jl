##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff
##$#############################################################################
##^# graph tests ###############################################################
function test()
  n = 10
  a = Tensor(rand(n))
  #a = Tensor(rand(n, n))
  b = Tensor(rand(n, n))

  #@btime c = $b - $a
  #c = b - a + a + a + a
  #global c = add(add(a, b), a)
  #global c = 3.0 * a + a - b * 4
  c = b * SAD.softminish(a)

  #@btime J = compute_jacobian($c, $a)
  @btime J = SAD.compute_jacobian($c, $a)
  J = SAD.compute_jacobian(c, a)
  display(collect(J))
  @btime J_ = SAD.jacobian_gen(a -> $b.value * SAD.softminish(a))($a.value)
  J_ = SAD.jacobian_gen(a -> b.value * SAD.softminish(a))(a.value)
  display(reshape(J_, :, length(a)))

  #@add_rule function add(cache, a, b)
  #  return a + b
  #end [(cache, a, b) -> (a + b), (cache, a, b) -> (a + b)]

  return
  #return J
  #return a, b, c
end
##$#############################################################################
