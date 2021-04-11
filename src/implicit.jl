function implicit_1st(
  k_fn::Function,
  y::Union{Tensor{T},AbstractArray{T},T},
  params...;
  reg::T = 1e-9,
  ) where {T}
  #::Vector{Union{AbstractArray{T}, T}} where {T}
  y = isa(y, Tensor) ? y : Tensor{T}(y)
  params = [isa(param, Tensor) ? param : Tensor{T}(param) for param in params]
  k = k_fn(y, params...)
  Jy = reshape(compute_jacobian(k, y), length(y), length(y))
  Js = [compute_jacobian(k, param) for param in params]
  F = lu(Jy + reg * I)
  ret = [
    -reshape(
      F \ reshape(J, length(y), length(param)),
      size(y)...,
      size(param)...,
    ) for (J, param) in zip(Js, params)
  ]
  return ret
end
