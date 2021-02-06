include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff

using BenchmarkTools

function test()
  r = rand(3, 2)
  a = SAD.Tensor(r)
  b = SAD.Tensor(r)
  #c = sum((a ./ 2) .^ 2)
  #c = (a .^ 2) ./ 2
  #c = reduce(vcat, [a, b, b])
  c = a[2:3, 2]
  #c = a .^ SAD.Tensor(2.0)
  #c = sum(a)
  display(c)
  println(repeat("#", 80))

  display(collect(SAD.compute_jacobian(c, a)))
  display(a)
  return
end
test()
