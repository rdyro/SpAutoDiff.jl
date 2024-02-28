##^# macro tools ###############################################################
global ALIAS_IDX = 1
macro add_rule(fn_, jacobian_fns_, hessian_fns_ = nothing)
  fn, jacobian_fns, hessian_fns = fn_, jacobian_fns_, hessian_fns_

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
  jacobian_alias = Symbol(@sprintf("jacobian_fns_alias_%09d", ALIAS_IDX))
  hessian_alias = Symbol(@sprintf("hessian_fns_alias_%09d", ALIAS_IDX))
  #jacobian_assign = Expr(:const, Expr(:(=), jacobian_alias, jacobian_fns))
  #hessian_assign = Expr(:const, Expr(:(=), hessian_alias, hessian_fns))
  jacobian_assign = Expr(:(=), jacobian_alias, jacobian_fns)
  hessian_assign = Expr(:(=), hessian_alias, hessian_fns)
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
    #cache = Dict{Union{Symbol,String},Union{<:Real,AbstractArray{<:Real}}}()
    cache = nothing
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
    #ret.jacobian_fns = $jacobian_fns
    ret.jacobian_fns = $jacobian_alias
    #ret.hessian_fns = $hessian_fns
    ret.hessian_fns = $hessian_alias
    return ret
  end
  fn_def = Expr(:function, fn_head, fn_body)

  # define the alias ################################################
  cache_arg = Expr(
    :(::),
    :cache,
    #:(Dict{Union{Symbol,String},Union{<:Real,AbstractArray{<:Real}}}),
    :(Nothing),
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
          #i -> Expr(
          #  :(::),
          #  fn_args_names[i],
          #  fn_args_is_tensor[i] ? :(Union{<:Real,AbstractArray{<:Real}}) :
          #      fn_args_types[i],
          #),
          i -> fn_args_names[i],
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
    $jacobian_assign
    $hessian_assign
    $alias
    $fn_def
  end)
end
##$#############################################################################
##^# linalg operators ##########################################################
import Base: +, -, /, ^, hcat, vcat, sum, dropdims, adjoint

@add_rule function +(cache, a, b)
  return a + b
end [
  #(cache, a, b) -> sparse(1.0 * I, length(a), length(a)),
  #(cache, a, b) -> sparse(1.0 * I, length(b), length(b)),
  (cache, a, b) -> 1.0 * I,
  (cache, a, b) -> 1.0 * I,
]

@add_rule function -(cache, a, b)
  return a - b
end [(cache, a, b) -> 1.0 * I, (cache, a, b) -> -1.0 * I]

@add_rule function -(cache, a)
  return -a
end Function[(cache, a) -> -T(1) * I]

+(a::Union{Real,AbstractArray}, b::Tensor{T}) where {T} = Tensor{T}(a) + b
+(a::Tensor{T}, b::Union{AbstractArray,Real}) where {T} = a + Tensor{T}(b)

-(a::Union{Real,AbstractArray}, b::Tensor{T}) where {T} = Tensor{T}(a) - b
-(a::Tensor{T}, b::Union{AbstractArray,Real}) where {T} = a - Tensor{T}(b)

import LinearAlgebra: dot, adjoint

@add_rule function dot(cache, a, b)
  return dot(a, b)
end [
  function (cache, a, b)
    if ndims(a) == 1 && ndims(b) == 1
      return b'
    else
      error("That version of [dot] is not yet supported")
      return nothing
    end
  end,
  function (cache, a, b)
    if ndims(a) == 1 && ndims(b) == 1
      return a'
    else
      error("That version of [dot] is not yet supported")
      return nothing
    end
  end,
]

dot(a::Union{Real,AbstractArray}, b::Tensor{T}) where {T} = dot(Tensor{T}(a), b)
dot(a::Tensor{T}, b::Union{AbstractArray,Real}) where {T} = dot(a, Tensor{T}(b))

@add_rule function adjoint(cache, a)
  return a'
end Function[function (cache, a)
  if ndims(a) == 1 || ndims(a) == 0
    return 1.0 * I
  elseif ndims(a) == 2
    m, n = size(a)
    Ja = collect(1:(m * n))
    Ia = reduce(vcat, map(i -> (1:n:length(a)) .+ (i - 1), 1:n))
    Va = ones(length(a))
    return sparse(Ia, Ja, Va, length(a), length(a))
  else
    error(@sprintf("No such adjoint for ndims(a) = %d", ndims(a)))
    return nothing
  end
end]


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
    for i in 1:size(a, 2)
      I[((i - 1) * n1 + 1):(i * n1)] = ((i - 1) * (n1 + n2)) .+ (1:n1)
    end
    V = ones(Float64, length(a))
    return sparse(I, J, V, length(a) + length(b), length(a))
  end,
  function (cache, a, b)
    n1, n2 = size(a, 1), size(b, 1)
    J = collect(1:length(b))
    I = zeros(Int, length(b))
    for i in 1:size(b, 2)
      I[((i - 1) * n2 + 1):(i * n2)] = ((i - 1) * (n1 + n2) + n1) .+ (1:n2)
    end
    V = ones(Float64, length(b))
    return sparse(I, J, V, length(a) + length(b), length(b))
  end,
]

@add_rule function sum(cache, a; dims = Colon())
  val = sum(a; dims = dims)
  #cache["val"] = val
  return val
end Function[function (cache, a; dims = Colon())
  if dims == Colon()
    return ones(1, length(a))
    #elseif isa(dims, Int)
  else
    #val = cache["val"]
    val = sum(a; dims = dims)
    I, J, V = zeros(Int, length(a)), zeros(Int, length(a)), ones(T, length(a))
    mask = [!(dim in dims) for dim in 1:ndims(a)]
    l2c = CartesianIndices(a)
    c2l_reduced = LinearIndices(Tuple(map(s -> 1:s, collect(size(a))[mask])))
    for li in 1:length(a)
      ci = l2c[li]
      J[li] = li
      I[li] = c2l_reduced[CartesianIndex(collect(Tuple(ci))[mask]...)]
    end
    return sparse(I, J, V)
  end
end]

@add_rule function dropdims(cache, a; dims)
  return dropdims(a; dims = dims)
end Function[function (cache, a; dims)
  #if isa(dims, Int)
  #  return sparse(1.0 * I, length(a), length(a))
  #else
  #  error("That type of [sum] is not supported")
  #  return nothing
  #end
  return sparse(1.0 * I, length(a), length(a))
end]
##$#############################################################################
##^# multiplication ############################################################
import Base: *

@add_rule function *(cache, a, b)
  return a * b
end [
  function (cache, a, b)
    b = isa(b, UniformScaling) ? b.λ : b
    if length(b) == 1
      return sparse(I, length(a), length(a)) * (size(b) == () ? b : b[1])
    elseif length(a) == 1
      return reshape(b, :, 1)
    else
      if ndims(b) == 1 && !isa(b, Adjoint)
        # we use a workaround here, k
        return kron(b, sparse(I, size(a, 1), size(a, 1)))' 
      else
        return kron(b', sparse(I, size(a, 1), size(a, 1)))
      end
      #return kron(b', sparse(I, size(a, 1), size(a, 1)))
    end
  end,
  function (cache, a, b)
    a = isa(a, UniformScaling) ? a.λ : a
    if length(a) == 1
      return sparse(I, length(b), length(b)) * (size(a) == () ? a : a[1])
    elseif length(b) == 1
      return reshape(a, :, 1)
    else
      return kron(sparse(I, size(b, 2), size(b, 2)), a)
    end
  end,
]

@add_rule function *(cache, a, b::UniformScaling)
  return a * b
end [function (cache, a, b)
  return b.λ * I
end, function (cache, a, b)
  return nothing
end]

@add_rule function *(cache, a::UniformScaling, b)
  return a * b
end [function (cache, a, b)
  return nothing
end, function (cache, a, b)
  return a.λ * I
end]

*(a::Union{Real,AbstractArray}, b::Tensor{T}) where {T} = Tensor{T}(a) * b
*(a::Tensor{T}, b::Union{AbstractArray,Real}) where {T} = a * Tensor{T}(b)
##$#############################################################################
##^# kronecker product #########################################################
import Base: kron

function is_diag(x::SparseMatrixCSC)
  (size(x, 1) == size(x, 2)) || (return false)
  (length(diag(x).nzval) == length(x.nzval) == size(x, 1)) || (return false)
  #(length(unique(x.nzval)) == 1) || (return false)
  return true
end

@add_rule function kron(cache, a, b::SparseMatrixCSC)
  return kron(sparse(a), b)
end [
  function (cache, a, b)
    @assert is_diag(b)
    n = size(b, 1)

    I, J, V = Int[], Int[], Float64[]
    for i in 1:size(a, 2)
      append!(J, repeat((size(a, 1) * (i - 1)) .+ (1:size(a, 1)), n))
      off = size(a, 1) * n^2 * (i - 1)
      indices = [
        off .+ (1:n:(n * size(a, 1))) .+ ((n * size(a, 1)) * (j - 1) + j - 1)
        for j in 1:n
      ]
      append!(I, reduce(vcat, indices))
      append!(V, repeat(b.nzval; inner = size(a, 1)))
    end
    return sparse(I, J, V, length(a) * n^2, length(a))
  end,
  (cache, a, b) -> nothing,
]

@add_rule function kron(cache, a::SparseMatrixCSC, b)
  return kron(a, sparse(b))
end [
  (cache, a, b) -> nothing,
  function (cache, a, b)
    @assert is_diag(a)
    n = size(a, 1)

    I, J, V = Int[], Int[], Float64[]
    for i in 1:n
      append!(J, 1:length(b))
      off1 = (i - 1) * (length(b) * n) + (i - 1) * size(b, 1)
      indices = [
        off1 .+ (1:size(b, 1)) .+ (j - 1) * size(b, 1) * n
        for j in 1:size(b, 2)
      ]
      append!(I, reduce(vcat, indices))
      append!(V, a.nzval[i] * ones(length(b)))
    end
    return sparse(I, J, V, length(b) * n^2, length(b))
  end,
]
##$#############################################################################
##^# reduce operator ###########################################################
import Base: reduce

function reduce_hcat_jacobian(
  i::Int,
  cache,
  arg_list::Vector{Union{Tensor{T},AbstractArray{T},T,Real}},
) where {T}
  #csizesum, clengthsum = cache["csizesum"], cache["clengthsum"]
  csizesum = (
    cumsum(size(arg, 1) for arg in arg_list),
    cumsum(size(arg, 2) for arg in arg_list),
  )
  clengthsum = cumsum(length(arg) for arg in arg_list)
  idxshift, n = (i > 1 ? clengthsum[i - 1] : 0), length(arg_list[i])
  J = collect(1:n) .+ idxshift
  J = collect(1:n)
  V = ones(T, n)
  return sparse(I, J, V, clengthsum[end], n)
end

function reduce_vcat_jacobian(
  i::Int,
  cache,
  arg_list::Vector{Union{Tensor{T},AbstractArray{T},T,Real}},
) where {T}
  #csizesum, clengthsum = cache["csizesum"], cache["clengthsum"]
  csizesum = (
    cumsum(size(arg, 1) for arg in arg_list),
    cumsum(size(arg, 2) for arg in arg_list),
  )
  clengthsum = cumsum(length(arg) for arg in arg_list)
  idxshift, n = (i > 1 ? csizesum[i - 1] : 0), length(arg_list[i])
  arg = arg_list[i]
  m, n = size(arg, 1), size(arg, 2)
  I =
    reduce(vcat, [(1:m) .+ idxshift .+ (j - 1) * csizesum[1][end] for j in 1:n])
  J = collect(1:n)
  V = ones(T, n)
  return sparse(I, J, V, clengthsum[end], n)
end

if !@isdefined reduce_df_map
  const reduce_df_map = Dict{Function,Function}(
    vcat => reduce_vcat_jacobian,
    hcat => reduce_hcat_jacobian,
  )
end

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
  #cache["csizesum"] = (
  #  cumsum(size(arg, 1) for arg in arg_list),
  #  cumsum(size(arg, 2) for arg in arg_list),
  #)
  #cache["clengthsum"] = cumsum(length(arg) for arg in arg_list)
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

ShapeT = Union{Colon,Int,AbstractUnitRange}
@add_rule function getindex(cache, a, idxs::ShapeT)
  return a[idxs]
end [
  function (cache, a, idxs)
    idxs = isa(idxs, Colon) ? (1:length(a)) : idxs
    J = collect(size(idxs) == () ? [idxs] : idxs)
    I = collect(1:length(idxs))
    V = ones(length(idxs))
    return sparse(I, J, V, length(idxs), length(a))
  end,
  (cache, a, idxs) -> nothing,
]

@add_rule function getindex(cache, a, idxs1::ShapeT, idxs2::ShapeT)
  return a[idxs1, idxs2]
end [
  function (cache, a, idxs1, idxs2)
    idxs1 = isa(idxs1, Colon) ? (1:size(a, 1)) : idxs1
    idxs2 = isa(idxs2, Colon) ? (1:size(a, 2)) : idxs2
    idxs1 = size(idxs1) == () ? [idxs1] : collect(idxs1)
    idxs2 = size(idxs2) == () ? [idxs2] : collect(idxs2)
    J = repeat(idxs1; outer = length(idxs2))
    m, l = size(a, 1), length(idxs1)
    for i in 1:length(idxs2)
      J[((i - 1) * l + 1):(i * l)] .+= m * (idxs2[i] - 1)
    end
    I = collect(1:length(J))
    V = ones(length(I))
    return sparse(I, J, V, length(I), length(a))
  end,
  (cache, a, idxs1, idxs2) -> nothing,
  (cache, a, idxs1, idxs2) -> nothing,
]
##$#############################################################################
##^# reshape ###################################################################
import Base: reshape

@add_rule function reshape(cache, a, idxs1::ShapeT)
  return reshape(a, idxs1)
end [
  function (cache, a, idxs1)
    return sparse(1.0 * I, length(a), length(a))
  end,
  (cache, a, idxs1) -> nothing,
]

@add_rule function reshape(cache, a, idxs1::ShapeT, idxs2::ShapeT)
  return reshape(a, idxs1, idxs2)
end [
  function (cache, a, idxs1, idxs2)
    return sparse(1.0 * I, length(a), length(a))
  end,
  (cache, a, idxs1, idxs2) -> nothing,
  (cache, a, idxs1, idxs2) -> nothing,
]

@add_rule function reshape(
  cache,
  a,
  idxs1::ShapeT,
  idxs2::ShapeT,
  idxs3::ShapeT,
)
  return reshape(a, idxs1, idxs2, idxs3)
end [
  function (cache, a, idxs1, idxs2, idxs3)
    return sparse(1.0 * I, length(a), length(a))
  end,
  (cache, a, idxs1, idxs2, idxs3) -> nothing,
  (cache, a, idxs1, idxs2, idxs3) -> nothing,
  (cache, a, idxs1, idxs2, idxs3) -> nothing,
]
##$#############################################################################
##^# elementwise operators #####################################################
import Base: map, broadcast
import .Broadcast: broadcasted, materialize

if !@isdefined unary_df_map
  const unary_df_map = Dict{Function,Union{Function,Nothing}}(
    abs => sign,
    sin => cos,
    cos => x -> -sin(x),
    tan => x -> sec(x)^2,
    sign => x -> one(x),
    exp => exp,
    log => x -> 1.0 / x,
  )
end
if !@isdefined unary_d2f_map
  const unary_d2f_map = Dict{Function,Union{Function,Nothing}}(
    abs => nothing,
    sin => x -> -sin(x),
    cos => x -> -cos(x),
    tan => x -> 2 * sec(x)^2 * tan(x),
    sign => nothing,
    exp => exp,
    log => x -> -1.0 / x^2,
  )
end

if !@isdefined binary_df_map
  const binary_df_map =
    Dict{Function,Tuple{Union{Function,Nothing},Union{Function,Nothing}}}(
      Base.:+ => ((x, y) -> 1.0, (x, y) -> 1.0),
      Base.:- => ((x, y) -> 1.0, (x, y) -> -1.0),
      Base.:* => ((x, y) -> y, (x, y) -> x),
      Base.:^ => ((x, y) -> y * x^(y - 1.0), (x, y) -> (x^y) * log(x)),
      Base.:/ => ((x, y) -> 1 / y, (x, y) -> -x / y^2),
      max => ((x, y) -> one(x) * (x >= y), (x, y) -> one(y) * (y >= x)),
      min => ((x, y) -> one(x) * (x <= y), (x, y) -> one(y) * (y <= x)),
    )
end
if !@isdefined binary_d2f_map
  const binary_d2f_map =
    Dict{Function,Tuple{Union{Function,Nothing},Union{Function,Nothing}}}(
      Base.:+ => (nothing, nothing),
      Base.:- => (nothing, nothing),
      Base.:* => (nothing, nothing),
      Base.:^ =>
        ((x, y) -> y * (y - 1) * x^(y - 2.0), (x, y) -> (x^y) * log(x)^2),
      Base.:/ => (nothing, (x, y) -> 2 * x / y^3),
      max => (nothing, nothing),
      min => (nothing, nothing),
    )
end

@add_rule function map(cache, f::Function, a)
  return map(f, a)
end [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    if haskey(unary_df_map, f) && unary_df_map[f] != nothing
      (unary_df_map[f] == nothing) && (return nothing)
      val = unary_df_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
] [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    if haskey(unary_d2f_map, f) && unary_d2f_map[f] != nothing
      (unary_d2f_map[f] == nothing) && (return nothing)
      val = unary_d2f_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      #return spdiagm(0 => val)
      return sparse(
        map(i -> (i - 1) * length(a) + i, 1:length(a)),
        1:length(a),
        val,
        length(a)^2,
        length(a),
      )
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
    if haskey(unary_df_map, f) && unary_df_map[f] != nothing
      (unary_df_map[f] == nothing) && (return nothing)
      val = unary_df_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
] [
  (cache, f, a) -> nothing,
  function (cache, f, a)
    if haskey(unary_d2f_map, f) && unary_d2f_map[f] != nothing
      val = unary_d2f_map[f].(a)
      val = size(val) == () ? [val] : reshape(val, :)
      #return spdiagm(0 => val)
      return sparse(
        map(i -> (i - 1) * length(a) + i, 1:length(a)),
        1:length(a),
        val,
        length(a)^2,
        length(a),
      )
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
    if haskey(binary_df_map, f) && binary_df_map[f][1] != nothing
      val = binary_df_map[f][1].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
  function (cache, f, a, b)
    if haskey(binary_df_map, f) && binary_df_map[f][2] != nothing
      val = binary_df_map[f][2].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      return spdiagm(0 => val)
    else
      return nothing
    end
  end,
] [
  (cache, f, a, b) -> nothing,
  function (cache, f, a, b)
    if haskey(binary_d2f_map, f) && binary_d2f_map[f][1] != nothing
      val = binary_d2f_map[f][1].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      #return spdiagm(0 => val)
      return sparse(
        map(i -> (i - 1) * length(a) + i, 1:length(a)),
        1:length(val),
        val,
        length(val)^2,
        length(val),
      )
    else
      return nothing
    end
  end,
  function (cache, f, a, b)
    if haskey(binary_d2f_map, f) && binary_d2f_map[f][2] != nothing
      val = binary_d2f_map[f][2].(a, b)
      val = size(val) == () ? [val] : reshape(val, :)
      #return spdiagm(0 => val)
      return sparse(
        map(i -> (i - 1) * length(a) + i, 1:length(a)),
        1:length(val),
        val,
        length(val)^2,
        length(val),
      )
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
#@add_rule function softmaxish(cache, x; scale = 1)
#  return softmaxish(x; scale = scale)
#end Function[(cache, x; scale = 1) -> softmaxish_jacobian(x; scale = scale)]
#
#@add_rule function softminish(cache, x; scale = 1)
#  return softminish(x; scale = scale)
#end Function[(cache, x; scale = 1) -> softmaxish_jacobian(-x; scale = scale)]
#
#@add_rule function softmaxish(cache, x, n; scale = 1)
#  return softmaxish(x, n; scale = scale)
#end [
#  (cache, x, n; scale = 1) -> softmaxish_jacobian(x, n; scale = scale)
#  (cache, x, n; scale = 1) -> nothing
#]
#
#@add_rule function softminish(cache, x, n; scale = 1)
#  return softminish(x, n; scale = scale)
#end [
#  (cache, x, n; scale = 1) -> softmaxish_jacobian(-x, n; scale = scale),
#  (cache, x, n; scale = 1) -> nothing,
#]
#
#function softmaxish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
#  return softmaxish(x, Tensor{T}(n); scale = scale)
#end
#function softminish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
#  return softminish(x, Tensor{T}(n); scale = scale)
#end

@add_rule function lsemaxish(cache, x; scale = 1)
  return lsemaxish(x; scale = scale)
end Function[(cache, x; scale = 1) -> lsemaxish_jacobian(x; scale = scale)] Function[(
  cache,
  x;
  scale = 1,
) -> lsemaxish_hessian(x; scale = scale)]

@add_rule function lseminish(cache, x; scale = 1)
  return lseminish(x; scale = scale)
end Function[(cache, x; scale = 1) -> lsemaxish_jacobian(-x; scale = scale)] Function[(
  cache,
  x;
  scale = 1,
) -> -lsemaxish_hessian(-x; scale = scale)]

@add_rule function lsemaxish(cache, x, n; scale = 1)
  return lsemaxish(x, n; scale = scale)
end Function[(
  cache,
  x,
  n;
  scale = 1,
) -> lsemaxish_jacobian(x, n; scale = scale)] Function[(
  cache,
  x,
  n;
  scale = 1,
) -> lsemaxish_hessian(x, n; scale = scale)]

@add_rule function lseminish(cache, x, n; scale = 1)
  return lseminish(x, n; scale = scale)
end Function[(
  cache,
  x,
  n;
  scale = 1,
) -> lsemaxish_jacobian(-x, n; scale = scale)] Function[(
  cache,
  x,
  n;
  scale = 1,
) -> -lsemaxish_hessian(-x, n; scale = scale)]

function lsemaxish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
  return lsemaxish(x, Tensor{T}(T(n)); scale = scale)
end
function lseminish(x::Tensor{T}, n::Int; scale::Real = 1) where {T}
  return lseminish(x, Tensor{T}(T(n)); scale = scale)
end
##$#############################################################################
