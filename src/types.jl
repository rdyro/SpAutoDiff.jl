mutable struct Tensor{T}
  value::Union{T, AbstractArray{T}}
  requires_grad::Bool
  #parents::Vector{Union{Tensor{T}, Union{Real, AbstractArray{Real}}}}
  parents::Vector{Union{Tensor{T}, Any}}
  parameters::Union{Dict{Symbol, Any}, Nothing}
  cache::Dict{Union{Symbol, String}, Union{<: Real, AbstractArray{<: Real}}}
  jacobian_fns::Union{Vector{Function}, Function}
  hessian_fns::Vector{Function}
end

Tensor{T}(value::Union{<: Real,AbstractArray{<: Real}}, requires_grad = true) where {T} =
  Tensor{T}(
    T.(value),
    requires_grad,
    Tensor{T}[],
    nothing,
    Dict{Union{Symbol, String}, Union{<: Real, AbstractArray{<: Real}}}(),
    Function[],
    Function[],
  )
Tensor(value::Union{T,AbstractArray{T}}, requires_grad = true) where {T} =
  Tensor{T}(value, requires_grad)

import Base.display
function display(t::Tensor)
  println(t)
  println("with value:")
  display(t.value)
  return
end
