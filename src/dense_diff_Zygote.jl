module DenseDiffZygote
using Zygote

import .._reduce

include(joinpath(@__DIR__, "utils.jl"))

#gradient, hessian = Zygote.gradient, Zygote.hessian
#function jacobian_gen(fn; argnums = (), bdims = 0)
#  f_fn, f_fn_, fi_fn = nothing, nothing, nothing
#  return function (args...)
#    (f_fn == nothing) && (f_fn = fn)
#    f = f_fn(args...)
#    batch_dims = Tuple(ndims(f):-1:(ndims(f) - bdims + 1))
#    if bdims > 0
#      f = reduce_sum(f; dims = batch_dims)
#    end
#    @assert typeof(f) <: AbstractArray || typeof(f) <: Number
#    if size(f) == () && bdims == 0
#      gs = gradient(f_fn, args...)
#      gs = [reshape(g, 1, :) for g in gs]
#    elseif size(f) == ()
#      if f_fn_ == nothing
#        f_fn_ = function (args...)
#          return reduce_sum(f_fn(args...); dims = batch_dims)
#        end
#      end
#      gs = gradient(f_fn_, args...)
#      gs = [reshape(g, 1, :) for g in gs]
#    else
#      if fi_fn == nothing
#        fi_fn = function (i, args...)
#          return reshape(reduce_sum(f_fn(args...); dims = batch_dims), :)[i]
#        end
#      end
#      gs_list = [gradient(fi_fn, i, args...)[2:end] for i in 1:length(f)]
#      gs = [stack(g) for g in zip(gs_list...)]
#      #gs = [reshape(g, size(f)..., size(g)[2:end]...) for g in gs]
#      gs = [reshape(g, length(f), :) for g in gs]
#    end
#    if !(typeof(argnums) <: Tuple) && size(argnums) == ()
#      argnums = [argnums]
#    end
#    if length(argnums) != 0
#      gs = Tuple(g for (i, g) in enumerate(gs) if i in argnums)
#    end
#    return length(gs) == 0 ? nothing : (length(gs) == 1 ? gs[1] : gs)
#  end
#end

gradient, hessian = Zygote.gradient, Zygote.hessian

function jacobian_gen(fn; argnums = ())
  return function (args...; kwargs...)
  #return function (arg)
    # differentiate wrt to the first argument only
    @assert argnums == 1 ||
            argnums == (1,) ||
            (argnums == () && length(args) == 1)
    fn_ = arg1 -> fn(reshape(arg1, size(args[1])...), args[2:end]...; kwargs...)
    f = fn_(args[1])
    if size(f) == ()
      return reshape(Zygote.gradient(fn_, args[1])[1], 1, length(args[1]))
    else
      return _reduce(
        vcat,
        [
          reshape(gradient(x -> fn_(x)[i], reshape(args[1], :))[1], 1, :)
          for i in 1:length(f)
        ],
      )
    end
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
