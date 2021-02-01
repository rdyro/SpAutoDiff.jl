##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff
##$#############################################################################
##^# new tests #################################################################
function test()
  r = randn(5)
  fn = tan
  fn_derv = x -> -sin(x)
  a = SAD.Tensor(r)
  c = fn.(a)
  display(c)
  display(fn.(r))
  println(repeat("#", 80))

  display(collect(SAD.compute_jacobian(c, a)))
  display(diagm(0 => fn_derv.(r)))
  return
end
test()
##$#############################################################################
