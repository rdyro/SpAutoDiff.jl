##^# functions definitions #####################################################
function softmax(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  xmax, xidx = findmax(x)
  if isinf(scale)
    z = ones(T, length(x))
    z[xidx] = T(1)
    return z
  end
  x_ = x .- xmax
  z = exp.(scale * x_)
  z = z ./ sum(z)
  return z
end

function softmaxish(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  return sum(x .* softmax(x; scale = scale))
end

function softmaxish(x::AbstractArray{T,1}, nb::Real; scale::Real = 1) where {T}
  nb = Int(nb)
  @assert 1 <= nb <= length(x)
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]
  return softmaxish(x_; scale = scale)
end

function softminish(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  return -softmaxish(-x; scale = scale)
end

function softminish(x::AbstractArray{T,1}, nb::Real; scale::Real = 1) where {T}
  nb = Int(nb)
  return -softmaxish(-x, nb; scale = scale)
end
##$#############################################################################
##^# gradients definitions #####################################################
function softmaxish_jacobian(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  s = softmax(x; scale = scale)
  return reshape(s + scale * ((x .* s) - dot(x, s) * s), 1, :)
end

function softmaxish_hessian(x::AbstractArray{T,1}; scale::Real = 1) where {T}
  s = softmax(x; scale = scale)
  Ds = -s * s'
  Ds[diagind(Ds)] += s
  Ds *= scale
  return (scale * x .+ 1) .* Ds + diagm(0 => scale * s) -
         scale * (s * (s + Ds * x)' + dot(x, s) * Ds)
end

function softmaxish_jacobian(
  x::AbstractArray{T,1},
  nb::Real;
  scale::Real = 1,
) where {T}
  nb = Int(nb)
  @assert 1 <= nb <= length(x)
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]

  J_ = softmaxish_jacobian(x_; scale = scale)
  return sparse(
    ones(Int, nb),
    pidx[(end - nb + 1):end],
    reshape(J_, :),
    1,
    length(x),
  )
end

function softmaxish_hessian(
  x::AbstractArray{T,1},
  nb::Real;
  scale::Real = 1,
) where {T}
  nb = Int(nb)
  @assert (1 <= nb <= length(x))
  pidx = sortperm(x)
  @views pidx_ = pidx[(end - nb + 1):end]
  @views x_ = x[pidx_]

  H_ = softmaxish_hessian(x_; scale = scale)
  return sparse(
    repeat(pidx_; outer = nb),
    repeat(pidx_; inner = nb),
    reshape(H_, :),
    length(x),
    length(x),
  )
end
##$#############################################################################
