include(joinpath(@__DIR__, "header.jl"))
SAD = SpAutoDiff

using BenchmarkTools

function test()
  t = range(-5; stop = 5, length = 20)
  a = SAD.Tensor(sin.(t))
  b = SAD.Tensor(sin.(t .+ pi / 2))
  #c = SAD.lseminish(reduce(vcat, [a, b]); scale=1e1)
  #c = SAD.and(SAD.eventually(a), SAD.always(b); scale=1e0)
  @btime begin
    c = $SAD.until($a, $b; scale=Inf, use_outer_max=false)
    J = SAD.compute_jacobian(c, $a)
  end
  @btime begin
    c = $SAD.until($a, $b; scale=Inf, use_outer_max=true)
    J = SAD.compute_jacobian(c, $a)
  end
  c = SAD.until(a, b; scale=Inf)
  #list = [a[1:i] for i = 1:length(a)]
  #for c in list
  #  display(c)
  #end
  #c = reduce(vcat, list)
  println(repeat("#", 80))
  #display(c)

  J = collect(SAD.compute_jacobian(c, a))
  display(J)
  #display(a)
  return J
end
ret = test()
