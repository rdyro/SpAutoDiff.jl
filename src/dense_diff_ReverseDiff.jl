module DenseDiffReverseDiff
using ReverseDiff
gradient, hessian = ReverseDiff.jacobian, ReverseDiff.hessian

import .._reduce

function jacobian_gen(fn; argnums = 1)
  return function (args...; kwargs...)
    # differentiate wrt to the first argument only
    @assert argnums == 1 ||
            argnums == (1,) ||
            (argnums == () && length(args) == 1)
    gs = gradient(
      arg ->
        (f = fn(arg, args[2:end]...; kwargs...); size(f) == () ? [f] : f),
      args[1],
    )
    return reshape(gs, :, length(args[1]))
  end
end

function hessian_gen(fn; argnums = ())
  return function (args...; kwargs...)
    # differentiate wrt to the first argument only
    @assert argnums == 1 ||
            argnums == (1,) ||
            (argnums == () && length(args) == 1)
    fn_ = arg1 -> fn(reshape(arg1, size(args[1])...), args[2:end]...; kwargs...)
    f = fn_(args[1])
    if size(f) == ()
      return reshape(hessian(fn_, args[1]), length(args[1]), length(args[1]))
    else
      return _reduce(
        vcat,
        [hessian(x -> fn_(x)[i], reshape(args[1], :)) for i in 1:length(f)],
      )
    end
  end
end
end
