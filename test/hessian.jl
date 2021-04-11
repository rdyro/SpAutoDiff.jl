#using Debugger
include("header.jl")

function test()
  r = randn(50)
  a = SAD.Tensor(r)
  c = 0.5 * sum(abs.(a .^ 2))
  J = SAD.compute_jacobian(c, a)
  display(J)
  J, H = SAD.compute_hessian(c, a)
  #@btime J = SAD.compute_jacobian($c, $a)
  #@btime J, H = SAD.compute_hessian($c, $a)
  display(H)

  return
end
test()

return
