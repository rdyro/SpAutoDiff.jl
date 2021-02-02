##^# macro tools ###############################################################
global ALIAS_IDX = 1
macro add_rule(fn_, jacobian_fns_)
  #global fn = fn_
  #global jacobian_fns = jacobian_fns_
  fn = fn_
  jacobian_fns = jacobian_fns_

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
      arg -> isa(arg, Symbol) ? arg == :Tensor :
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
      i -> fn_args_is_tensor[i] ? Expr(:., fn_args_names[i], :(:value)) :
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
##^# operators definitions #####################################################
import Base: +, -, *, /, ^, hcat, vcat, sum

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

@add_rule function sum(cache, a)
  return sum(a)
end Function[(cache, a) -> ones(1, length(a))]
##$#############################################################################
##^# reduce operator ###########################################################
import Base: reduce

function reduce_hcat_jacobian(
  i::Int,
  cache,
  arg_list::Vector{Union{Tensor{T},AbstractArray{T},T,Real}},
) where {T}
  csizesum, clengthsum = cache["csizesum"], cache["clengthsum"]
  idxshift, n = (i > 1 ? clengthsum[i - 1] : 0), length(arg_list[i])
  I = collect(1:n) .+ idxshift
  J = collect(1:n)
  V = ones(T, n)
  return sparse(I, J, V, clengthsum[end], n)
end

function reduce_vcat_jacobian(
  i::Int,
  cache,
  arg_list::Vector{Union{Tensor{T},AbstractArray{T},T,Real}},
) where {T}
  csizesum, clengthsum = cache["csizesum"], cache["clengthsum"]
  idxshift, n = (i > 1 ? csizesum[i - 1] : 0), length(arg_list[i])
  arg = arg_list[i]
  m, n = size(arg, 1), size(arg, 2)
  I =
    reduce(vcat, [(1:m) .+ idxshift .+ (j - 1) * csizesum[1][end] for j = 1:n])
  J = collect(1:n)
  V = ones(T, n)
  return sparse(I, J, V, clengthsum[end], n)
end

const reduce_df_map = Dict{Function,Function}(
  vcat => reduce_vcat_jacobian,
  hcat => reduce_hcat_jacobian,
)

function reduce(
  f::Function,
  arg_list::Vector{Union{Tensor{T},AbstractArray{T},T,Real}},
) where {T}
  @assert haskey(reduce_df_map, f)
  global ALIAS_IDX
  fn_alias = Symbol(@sprintf("rule_alias_%09d!", ALIAS_IDX))
  ALIAS_IDX += 1

  cache = Dict{Union{Symbol,String},Union{<:Real,AbstractArray{<:Real}}}()
  value = reduce(f, [isa(arg, Tensor) ? arg.value : arg for arg in arg_list])
  cache["csizesum"] = (
    cumsum(size(arg, 1) for arg in arg_list),
    cumsum(size(arg, 2) for arg in arg_list),
  )
  cache["clengthsum"] = cumsum(length(arg) for arg in arg_list)
  ret = Tensor{T}(value)
  ret.cache = cache
  ret.parents = arg_list
  ret.parameters = nothing
  ret.requires_grad =
    any(isa(arg, Tensor) && arg.requires_grad for arg in arg_list)
  ret.jacobian_fns = reduce_df_map[f]

  return ret
end
##$#############################################################################
##^# getindex operator #########################################################
import Base: getindex

@add_rule function getindex(cache, a, idxs::Union{Int,AbstractArray{Int,1}}) 
  return a[idxs]
end [
  function (cache, a, idxs)
    J = collect(1:length(idxs))
    I = collect(size(idxs) == () ? [idxs] : idxs)
    V = ones(T, length(idxs))
    return sparse(I, J, V, length(a), length(idxs))
  end,
  (cache, a, idxs) -> nothing,
]
##$#############################################################################
##^# elementwise operators #####################################################
import Base: map, broadcast
import .Broadcast: broadcasted, materialize

const unitary_df_map = Dict{Function,Function}(sin => cos, cos => x -> -sin(x))
const binary_df_map = Dict{Function,Tuple{Function,Function}}(
  Base.:- => ((x, y) -> 1.0, (x, y) -> -1.0),
  Base.:^ => ((x, y) -> y * x^(y - 1.0), (x, y) -> (x^y) * log(x)),
  Base.:/ => ((x, y) -> 1 / y, (x, y) -> -x / y^2),
)

@add_rule function map(cache, f::Function, a)
  return map(f, a)
end [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    if haskey(unitary_df_map, f)
      val = unitary_df_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
]

@add_rule function broadcasted(cache, f::Function, a)
  return f.(a)
end [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    if haskey(unitary_df_map, f)
      val = unitary_df_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
]

@add_rule function broadcasted(cache, f::Function, a, b)
  return f.(a, b)
end [
  (cache, f, a, b) -> nothing,
  function (cache, f, a, b)
    if haskey(binary_df_map, f)
      val = binary_df_map[f][1].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
  function (cache, f, a, b)
    if haskey(binary_df_map, f)
      val = binary_df_map[f][2].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
]
function broadcasted(f::Function, a::Tensor{T}, b::Real) where {T}
  return broadcasted(f, a, Tensor{T}(b))
end
function broadcasted(f::Function, a::Real, b::Tensor{T}) where {T}
  return broadcasted(f, Tensor{T}(a), b)
end
# this is to capture the literal exponentiation (like squaring)
# this trick might break something else, but YOLO
function broadcasted(
  literal_pow::Function,
  f::Function,
  a::Tensor{T},
  b::Base.Val{V},
) where {T,V}
  @assert literal_pow == Base.literal_pow
  return broadcasted(f, a, T(V))
end

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
