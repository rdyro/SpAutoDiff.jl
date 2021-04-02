##^# imports ###################################################################
if !@isdefined DifferentiationModule
  #using Zygote
  #gradient, hessian = Zygote.gradient, Zygote.hessian
  #using ReverseDiff
  #gradient, hessian = ReverseDiff.gradient, ReverseDiff.hessian
  using FiniteDiff
  gradient = FiniteDiff.finite_difference_jacobian
  hessian = FiniteDiff.finite_difference_hessian
end
##$#############################################################################
##^# derivatives ###############################################################
function jacobian_gen(fn; argnums = (), bdims = 0)
  return function (args...)
    return gradient(
      arg -> (f = fn(arg, args[2:end]...); size(f) == () ? [f] : f),
      args[1],
    )
  end
end

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
#    elseif size(f) == ()
#      if f_fn_ == nothing
#        f_fn_ = function (args...)
#          return reduce_sum(f_fn(args...); dims = batch_dims)
#        end
#      end
#      gs = gradient(f_fn_, args...)
#    else
#      if fi_fn == nothing
#        fi_fn = function (i, args...)
#          return reshape(reduce_sum(f_fn(args...); dims = batch_dims), :)[i]
#        end
#      end
#      gs_list = [gradient(fi_fn, i, args...)[2:end] for i in 1:length(f)]
#      gs = [stack(g) for g in zip(gs_list...)]
#      gs = [reshape(g, size(f)..., size(g)[2:end]...) for g in gs]
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

function hessian_gen(fn; argnums = ())
  return function (args...)
    if length(args) == 1
      hs = hessian(fn, args[1])
    else
      @assert argnums == 1
      hs = hessian(
        arg1 -> fn(reshape(arg1, size(args[1])...), args[2:end]...),
        reshape(args[1], :),
      )
    end
    return hs
  end
end
##$#############################################################################
