mutable struct Tensor{T}
  value::Union{T,AbstractArray{T},UniformScaling{T}}
  requires_grad::Bool
  #parents::Vector{Union{Tensor{T}, Union{Real, AbstractArray{Real}}}}
  parents::Vector{Union{Tensor{T},Any}}
  parameters::Union{Dict{Symbol,Any},Nothing}
  #cache::Dict{Union{Symbol, String}, Union{<: Real, AbstractArray{<: Real}}}
  cache::Nothing
  jacobian_fns::Union{Vector{Function},Function}
  hessian_fns::Union{Vector{Function},Function,Nothing}
end

Tensor{T}(
  value::Union{<:Real,AbstractArray{<:Real},UniformScaling{<:Real}},
  requires_grad = true,
) where {T} = Tensor{T}(
  #T.(value),
  value,
  requires_grad,
  Tensor{T}[],
  nothing,
  #Dict{Union{Symbol, String}, Union{<: Real, AbstractArray{<: Real}}}(),
  nothing,
  Function[],
  Function[],
)
Tensor(value::Union{T,AbstractArray{T}}, requires_grad = true) where {T} =
  Tensor{T}(value, requires_grad)

import Base.display
function display(t::Tensor)
  #println(t)
  #println("with value:")
  println("Tensor with value:")
  display(t.value)
  return
end

import Base: length, size, lastindex, ndims, collect
length(a::Tensor) = length(a.value)
size(a::Tensor) = size(a.value)
size(a::Tensor, idx::Int) = size(a.value, idx)
lastindex(a::Tensor) = length(a.value)
lastindex(a::Tensor, idx::Int) = size(a.value, idx)
ndims(a::Tensor) = ndims(a.value)
collect(a::Tensor) = collect(a.value)

import Base: convert
convert(::Type{T}, a::Tensor{T}) where {T <: Real} = T(a.value)
