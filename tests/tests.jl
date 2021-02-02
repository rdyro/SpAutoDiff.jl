##^# imports ###################################################################
include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff

using BenchmarkTools
##$#############################################################################
##^# new tests #################################################################
function test()
  r = rand(5)
  a = SAD.Tensor(r)
  b = SAD.Tensor(r)
  #c = sum((a ./ 2) .^ 2)
  #c = (a .^ 2) ./ 2
  #c = reduce(vcat, [a, b, b])
  c = a[1]
  #c = a .^ SAD.Tensor(2.0)
  #c = sum(a)
  println(repeat("#", 80))

  display(collect(SAD.compute_jacobian(c, a)))
  display(a)
  return
end
test()
##$#############################################################################
