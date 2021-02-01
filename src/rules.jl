##^# macro tools ###############################################################
global ALIAS_IDX = 1
macro add_rule(fn_, jacobian_fns_)
  global fn = fn_
  global jacobian_fns = jacobian_fns_

  if fn.args[1].head == :where
    error("Parametric functions not currently supported")
  else
    fn_name = fn.args[1].args[1]
    has_params =
      isa(fn.args[1].args[2], Expr) && fn.args[1].args[2].head == :parameters
    fn_parameters = has_params ? fn.args[1].args[2] : nothing
    fn_args = has_params ? fn.args[1].args[4:end] : fn.args[1].args[3:end]
    fn_args_names = map(arg -> isa(arg, Symbol) ? arg : arg.args[1], fn_args)
    fn_args_types =
      map(arg -> isa(arg, Symbol) ? :(Tensor{T}) : arg.args[2], fn_args)
    fn_args_list = map(
      i -> Expr(:(::), fn_args_names[i], fn_args_types[i]),
      1:length(fn_args_names),
    )
    fn_head = Expr(:where, Expr(:call, fn_name, fn_args_list...), :T)
    if has_params
      insert!(fn_head.args[1].args, 2, Expr(:parameters, :(parameters...)))
    end
    fn_args_is_tensor = map(
      arg ->
        isa(arg, Symbol) ? arg == :Tensor :
        arg.head == :curly && arg.args[1] == :Tensor,
      fn_args_types,
    )
  end
  @assert any(fn_args_is_tensor)

  global ALIAS_IDX
  fn_alias = Symbol(@sprintf("rule_alias_%09d!", ALIAS_IDX))
  ALIAS_IDX += 1

  value_call = Expr(
    :call,
    fn_alias,
    :cache,
    map(
      i ->
        fn_args_is_tensor[i] ? Expr(:., fn_args_names[i], :(:value)) :
        fn_args_names[i],
      1:length(fn_args),
    )...,
  )
  if has_params
    insert!(value_call.args, 2, Expr(:parameters, :(parameters...)))
  end

  # define the function on types ####################################
  fn_body = quote
    cache = Dict{Union{Symbol,String},Union{<:Real,AbstractArray{<:Real}}}()
    value = $value_call
    ret = Tensor{T}(value)
    ret.cache = cache
    ret.parents = $(Expr(:vect, fn_args_names...))
    ret.parameters = $(has_params ? :(parameters) : :(nothing))
    ret.requires_grad = any(
      $(Expr(
        :vect,
        map(arg -> Expr(:., arg, :(:requires_grad)), fn_args_names)[fn_args_is_tensor]...,
      )),
    )
    ret.jacobian_fns = $jacobian_fns
    return ret
  end
  fn_def = Expr(:function, fn_head, fn_body)

  # define the alias ################################################
  cache_arg = Expr(
    :(::),
    :cache,
    :(Dict{Union{Symbol,String},Union{<:Real,AbstractArray{<:Real}}}),
  )
  alias_body = fn.args[2]
  alias = Expr(
    :function,
    Expr(
      :where,
      Expr(
        :call,
        fn_alias,
        cache_arg,
        map(
          i -> Expr(
            :(::),
            fn_args_names[i],
            fn_args_is_tensor[i] ? :(Union{<:Real,AbstractArray{<:Real}}) :
            fn_args_types[i],
          ),
          1:length(fn_args_names),
        )...,
      ),
      :T,
    ),
    alias_body,
  )
  if has_params
    insert!(alias.args[1].args[1].args, 2, fn_parameters)
  end

  #display(fn_def)
  #display(alias)

  #return
  return esc(quote
    $alias
    $fn_def
  end)
end
##$#############################################################################
##^# utility functions definitions #############################################
import Base: length, size
length(a::Tensor) = length(a.value)
size(a::Tensor) = size(a.value)
##$#############################################################################
import Base: map, broadcast
import .Broadcast: broadcasted, materialize

const fn_derv_map = Dict{Function,Function}(sin => cos, cos => x -> -sin(x))

@add_rule function map(cache, f::Function, a)
  return map(f, a)
end [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    return haskey(fn_derv_map, f) ? spdiagm(0 => fn_derv_map[f].(a)) : nothing
  end,
]

@add_rule function broadcasted(cache, f::Function, a)
  return f.(a)
end [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    return haskey(fn_derv_map, f) ? spdiagm(0 => fn_derv_map[f].(a)) : nothing
  end,
]
##$#############################################################################
##^# operators definitions #####################################################
import Base: +, -, *, hcat, vcat

@add_rule function +(cache, a, b)
  return a + b
end [
  (cache, a, b) -> sparse(T(1) * I, length(a), length(a)),
  (cache, a, b) -> sparse(T(1) * I, length(b), length(b)),
]
@add_rule function -(cache, a, b)
  return a - b
end [
  (cache, a, b) -> sparse(T(1) * I, length(a), length(a)),
  (cache, a, b) -> sparse(T(-1) * I, length(b), length(b)),
]

@add_rule function *(cache, a, b)
  return a * b
end [
  function (cache, a, b)
    if length(b) == 1
      return sparse(b * I, length(a), length(a))
    elseif length(a) == 1
      return reshape(b, :, 1)
    else
      return kron(b', sparse(I, size(a, 2), size(a, 2)))
    end
  end,
  function (cache, a, b)
    if length(a) == 1
      return sparse(a * I, length(b), length(b))
    elseif length(b) == 1
      return reshape(a, :, 1)
    else
      return kron(sparse(I, size(b, 2), size(b, 2)), a)
    end
  end,
]
*(a::Real, b::Tensor{T}) where {T} = Tensor{T}(a) * b
*(a::Tensor{T}, b::Real) where {T} = a * Tensor{T}(b)

@add_rule function hcat(cache, a, b)
  return hcat(a, b)
end [
  function (cache, a, b)
    n1, n2 = length(a), length(b)
    return [sparse(I, n1, n1); spzeros(n2, n1)]
  end,
  function (cache, a, b)
    n1, n2 = length(a), length(b)
    return [spzeros(n1, n2); sparse(I, n2, n2)]
  end,
]

@add_rule function vcat(cache, a, b)
  return vcat(a, b)
end [
  function (cache, a, b)
    n1, n2 = size(a, 1), size(b, 1)
    J = collect(1:length(a))
    I = zeros(Int, length(a))
    for i = 1:size(a, 2)
      I[((i - 1) * n1 + 1):(i * n1)] = ((i - 1) * (n1 + n2)) .+ (1:n1)
    end
    V = ones(T, length(a))
    return sparse(I, J, V, length(a) + length(b), length(a))
  end,
  function (cache, a, b)
    n1, n2 = size(a, 1), size(b, 1)
    J = collect(1:length(b))
    I = zeros(Int, length(b))
    for i = 1:size(b, 2)
      I[((i - 1) * n2 + 1):(i * n2)] = ((i - 1) * (n1 + n2) + n1) .+ (1:n2)
    end
    V = ones(T, length(b))
    return sparse(I, J, V, length(a) + length(b), length(b))
  end,
]
##$#############################################################################
##^# maxish rules ##############################################################
@add_rule function softmaxish(cache, x; scale = 1)
  return softmaxish(x; scale = scale)
end [(cache, x; scale = 1) -> softmaxish_jacobian(x; scale = scale)]

@add_rule function softminish(cache, x; scale = 1)
  return softminish(x; scale = scale)
end [(cache, x; scale = 1) -> softmaxish_jacobian(-x; scale = scale)]

@add_rule function softmaxish(cache, x, n; scale = 1)
  return softmaxish(x, n; scale = scale)
end [
  (cache, x, n; scale = 1) -> softmaxish_jacobian(x, n; scale = scale)
  (cache, x, n; scale = 1) -> nothing
]

@add_rule function softminish(cache, x, n; scale = 1)
  return softminish(x, n; scale = scale)
end [
  (cache, x, n; scale = 1) -> softmaxish_jacobian(-x, n; scale = scale),
  (cache, x, n; scale = 1) -> nothing,
]

function softmaxish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
  return softmaxish(x, Tensor{T}(n); scale = scale)
end
function softminish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
  return softminish(x, Tensor{T}(n); scale = scale)
end



@add_rule function lsemaxish(cache, x; scale = 1)
  return lsemaxish(x; scale = scale)
end [(cache, x; scale = 1) -> lsemaxish_jacobian(x; scale = scale)]

@add_rule function lseminish(cache, x; scale = 1)
  return lseminish(x; scale = scale)
end [(cache, x; scale = 1) -> lsemaxish_jacobian(-x; scale = scale)]
##$#############################################################################
