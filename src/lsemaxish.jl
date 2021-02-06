##^# functions definitions #####################################################
function lse(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  xmax = maximum(x)
  (isinf(scale)) && (return xmax)
  return log(sum(exp.(scale * (x .- xmax)))) / scale + xmax
end

function lsemaxish(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  return lse(x; scale = scale)
end

function lsemaxish(x::AbstractArray{T,1}, nb::Real; scale::Real = 1) where {T}
  nb = Int(nb)
  @assert 1 <= nb <= length(x)
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]
  return lse(x_; scale = scale)
end

function lseminish(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  return -lsemaxish(-x; scale = scale)
end

function lseminish(x::AbstractArray{T,1}, nb::Real; scale::Real = 1) where {T}
  return -lsemaxish(-x, nb; scale = scale)
end
##$#############################################################################
##^# gradients definitions #####################################################
function lsemaxish_jacobian(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  return softmax(x; scale = scale)'
  #s = softmax(x; scale = scale)
  #return reshape(s + scale * ((x .* s) - dot(x, s) * s), 1, :)
end

function lsemaxish_hessian(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  s = softmax(x; scale = scale)
  Ds = -s * s'
  Ds[diagind(Ds)] += s
  Ds *= scale
  #return (scale * x .+ 1) .* Ds + diagm(0 => scale * s) -
  #       scale * (s * (s + Ds * x)' + dot(x, s) * Ds)
  return Ds
end

function lsemaxish_jacobian(
  x::AbstractArray{T,1},
  nb::Real;
  scale::Real = 1,
) where {T}
  nb = Int(nb)
  @assert 1 <= nb <= length(x)
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]

  J_ = lsemaxish_jacobian(x_; scale = scale)
  return sparse(
    ones(Int, nb),
    pidx[(end - nb + 1):end],
    reshape(J_, :),
    1,
    length(x),
  )
end

function lsemaxish_hessian(
  x::AbstractArray{T,1},
  nb::Real;
  scale::Real = 1,
) where {T}
  nb = Int(nb)
  @assert (1 <= nb <= length(x))
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]

  H_ = lsemaxish_hessian(x_; scale = scale)
  return sparse(
    repeat(pidx_; outer = nb),
    repeat(pidx_; inner = nb),
    reshape(H_, :),
    length(x),
    length(x),
  )
end
##$#############################################################################
