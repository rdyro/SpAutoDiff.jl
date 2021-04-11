module DenseDiffFiniteDiff
using FiniteDiff
gradient = FiniteDiff.finite_difference_jacobian
hessian = FiniteDiff.finite_difference_hessian

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

function hessian_gen(fn; argnums = 1)
  return function (args...; kwargs...)
    # differentiate wrt to the first argument only
    @assert argnums == 1 ||
            argnums == (1,) ||
            (argnums == () && length(args) == 1)
    hs = hessian(
      arg1 -> fn(reshape(arg1, size(args[1])...), args[2:end]...; kwargs...),
      reshape(args[1], :),
    )
    if length(hs) == length(args[1])^2
      return reshape(hs, length(args[1]), length(args[1]))
    else
      return reduce(
        vcat,
        eachslice(reshape(hs, :, length(args[1]), length(args[1])); dims = 1),
      )
    end
  end
end
end
