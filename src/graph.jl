##^# graph differentiation #####################################################
function find_used_nodes!(
  colormap::Dict{Tensor{T},Bool},
  node::Tensor{T},
  input::Tensor{T},
) where {T}
  if node == input
    colormap[node] = true
    return
  end

  tensor_parents = filter(node_ -> isa(node_, Tensor), node.parents)
  #tensor_parents_orgs = deepcopy(tensor_parents)
  #node_org = deepcopy(node)

  if length(tensor_parents) == 0
    colormap[node] = false
    return
  end

  for node_ in tensor_parents
    find_used_nodes!(colormap, node_, input)
  end

  if any(map(
    node_ -> haskey(colormap, node_) && colormap[node_],
    tensor_parents,
  ))
    colormap[node] = true
  end
  return
end

function find_used_nodes(node::Tensor{T}, input::Tensor{T}) where {T}
  colormap = Dict{Tensor{T},Bool}()
  find_used_nodes!(colormap, node, input)
  return colormap
end

function toposort(
  nodes::Vector{Tensor{T}},
  to_consider::Set{Tensor{T}} = Set{Tensor{T}}(),
) where {T}
  to_sort = Set{Tensor{T}}(nodes)
  already_sorted = Set{Tensor{T}}()
  ordered = Tensor{T}[]
  while length(to_sort) > 0
    for node in to_sort
      if all(map(
        parent -> !in(parent, to_consider) || in(parent, already_sorted),
        node.parents,
      ))
        push!(already_sorted, node)
        push!(ordered, node)
        pop!(to_sort, node)
      end
    end
  end
  return ordered
end

function compute_jacobian(
  output::Tensor{T},
  input::Tensor{T};
  jacobian::Union{AbstractArray{T},T,UniformScaling{T},Nothing} = nothing,
) where {T}
  jacobian = jacobian != nothing ? jacobian : T(1)
  colormap = find_used_nodes(output, input)
  jacobians = Dict{
    Tensor{T},
    Union{Tensor{T},T,AbstractArray{T},UniformScaling{T},Nothing},
  }(
    input => jacobian,
  )
  nodes = collect(keys(colormap))
  to_consider = Set(keys(colormap))
  nodes = toposort(nodes, to_consider)
  for node in nodes
    (node == input) && (continue)
    J = nothing
    mask =
      length(node.parents) > 0 ?
      map(
        parent -> isa(parent, Tensor) && get(colormap, parent, false),
        node.parents,
      ) :
      Bool[]
    if length(mask) > 0 && any(mask)
      for (i, val) in enumerate(mask)
        (!val) && (continue)

        # determine jacobian arguments
        parameters = node.parameters != nothing ? node.parameters : ()
        args = map(
          #parent -> isa(parent, Tensor) ? parent.value : parent,
          parent -> isa(parent, Tensor) ? parent : parent,
          node.parents,
        )

        # compute the specific Dg of a particular parent
        if isa(node.jacobian_fns, AbstractArray)
          Dg = node.jacobian_fns[i](node.cache, args...; parameters...)
        else
          Dg = node.jacobian_fns(i, node.cache, args...; parameters...)
        end

        # obtain the cached Df
        Df = jacobians[node.parents[i]]

        # apply chain rule
        J_ = jacobian_chain_rule(Dg, Df)

        # sum jacobians
        (J_ != nothing) && (J = J != nothing ? J + J_ : J_)
      end
      jacobians[node] = J
    else
      jacobians[node] = nothing
    end
  end
  return jacobians[output]
end

function compute_hessian(
  output::Tensor{T},
  input::Tensor{T};
  jacobian::Union{AbstractArray{T},T,Nothing} = nothing,
  hessian::Union{AbstractArray{T},T,Nothing} = nothing,
) where {T}
  jacobian =
    jacobian != nothing ? jacobian :
    sparse(T(1) * I, length(input), length(input))
  hessian = hessian != nothing ? hessian : nothing
  colormap = find_used_nodes(output, input)

  jacobians =
    Dict{Tensor{T},Union{T,AbstractArray{T},Nothing}}(input => jacobian)
  hessians = Dict{Tensor{T},Union{T,AbstractArray{T},Nothing}}(input => hessian)

  nodes = collect(keys(colormap))
  to_consider = Set(keys(colormap))
  nodes = toposort(nodes, to_consider)
  for node in nodes
    (node == input) && (continue)
    J, H = nothing, nothing
    mask =
      length(node.parents) > 0 ?
      map(
        parent -> isa(parent, Tensor) && get(colormap, parent, false),
        node.parents,
      ) :
      Bool[]
    if length(mask) > 0 && any(mask)
      for (i, val) in enumerate(mask)
        (!val) && (continue)

        # determine jacobian arguments
        parameters = node.parameters != nothing ? node.parameters : ()
        #args = map(
        #  parent -> isa(parent, Tensor) ? parent.value : parent,
        #  node.parents,
        #)
        args =
          map(parent -> isa(parent, Tensor) ? parent : parent, node.parents)

        # compute the specific Dg and Hf of a particular parent
        if isa(node.jacobian_fns, AbstractArray)
          Dg = node.jacobian_fns[i](node.cache, args...; parameters...)
        else
          Dg = node.jacobian_fns(i, node.cache, args...; parameters...)
        end
        if isa(node.hessian_fns, AbstractArray)
          Hg = node.hessian_fns[i](node.cache, args...; parameters...)
        elseif isa(node.hessian_fns, Function)
          Hg = node.hessian_fns(i, node.cache, args...; parameters...)
        else
          Hg = nothing
        end

        # obtain the cached Df and Hf & apply chain rule
        Df, Hf = jacobians[node.parents[i]], hessians[node.parents[i]]
        J_ = jacobian_chain_rule(Dg, Df)
        m, p = length(args[i]), length(node.value)
        @bp
        H_ = hessian_chain_rule(m, p, Dg, Hg, Df, Hf)
        #(H_ != nothing) && (display(collect(H_)))

        # sum jacobians
        (J_ != nothing) && (J = J != nothing ? J + J_ : J_)
        (H_ != nothing) && (H = H != nothing ? H + H_ : H_)
      end
      jacobians[node], hessians[node] = J, H
    else
      jacobians[node], hessians[node] = nothing, nothing
    end
  end
  return jacobians[output], hessians[output]
end
##$#############################################################################
##^# utility functions #########################################################
function jacobian_chain_rule(
  Dg::Union{Tensor,<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
  Df::Union{Tensor,<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
)
  (Dg == nothing || Df == nothing) && (return nothing)
  return Dg * Df
end

function hessian_chain_rule(
  m::Int,
  p::Int,
  Dg::Union{<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
  Hg::Union{<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
  Df::Union{<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
  Hf::Union{<:Real,AbstractArray{<:Real},UniformScaling,Nothing},
) where {T}
  cond1 = !(Df == nothing || Hg == nothing)
  cond2 = !(Dg == nothing || Hf == nothing)
  (!cond1 && !cond2) && (return nothing)
  #println(repeat("+", 80))
  #display(typeof(Dg) <: AbstractArray ? collect(Dg) : Dg)
  #display(typeof(Hg) <: AbstractArray ? collect(Hg) : Hg)
  #display(typeof(Df) <: AbstractArray ? collect(Df) : Df)
  #display(typeof(Hf) <: AbstractArray ? collect(Hf) : Hf)

  #p = size(Df) == () ? size(Hg, 1) : size(Dg, 1)
  if cond1
    #Hh1 = isa(Df, UniformScaling) ? Df.λ ^ 2 * Hg :
    Hh1 =
      isa(Df, UniformScaling) ? Hg :
      kron(sparse(1.0 * I, p, p), size(Df) == () ? Df : sparse(Df))' * Hg * Df
  else
    Hh1 = nothing
  end

  #println()
  #display(typeof(Hh1) <: AbstractArray ? collect(Hh1) : Hh1)
  if !cond2
    #println(repeat("-", 80))
    return Hh1
  end

  n = size(Df) == () ? div(size(Hf, 1), size(Dg, 2)) : size(Df, 2)
  Hh2 =
    isa(Dg, UniformScaling) ? Dg.λ * Hf : kron(Dg, sparse(1.0 * I, n, n)) * Hf

  Hh = Hh1 == nothing ? Hh2 : Hh1 + Hh2

  #display(typeof(Hh2) <: AbstractArray ? collect(Hh2) : Hh2)
  #println(repeat("-", 80))
  return Hh
end
##$#############################################################################
