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
  tensor_parents_orgs = deepcopy(tensor_parents)
  node_org = deepcopy(node)

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
  jacobian::Union{AbstractArray{T},T,Nothing} = T(1),
) where {T}
  jacobian = jacobian != nothing ? jacobian : T(1)
  colormap = find_used_nodes(output, input)
  jacobians =
    Dict{Tensor{T},Union{T,AbstractArray{T},Nothing}}(input => jacobian)
  nodes = collect(keys(colormap))
  to_consider = Set(keys(colormap))
  nodes = toposort(nodes, to_consider)
  for node in nodes
    (node == input) && (continue)
    J = nothing
    mask = length(node.parents) > 0 ?
      map(
      parent -> isa(parent, Tensor) && get(colormap, parent, false),
      node.parents,
    ) :
      Bool[]
    if length(mask) > 0 && any(mask)
      for (i, val) in enumerate(mask)
        (!val) && (continue)
        if node.parameters != nothing
          Dg = node.jacobian_fns[i](
            node.cache,
            map(
              parent -> isa(parent, Tensor) ? parent.value : parent,
              node.parents,
            )...;
            node.parameters...
          )
        else
          Dg = node.jacobian_fns[i](
            node.cache,
            map(
              parent -> isa(parent, Tensor) ? parent.value : parent,
              node.parents,
            )...,
          )
        end
        Df = jacobians[node.parents[i]]
        J_ = jacobian_chain_rule(Dg, Df)
        (J_ != nothing) && (J = J != nothing ? J + J_ : J_)
      end
      jacobians[node] = J
    else
      jacobians[node] = nothing
    end
  end
  return jacobians[output]
end
##$#############################################################################
##^# utility functions #########################################################
function jacobian_chain_rule(
  Dg::Union{<:Real,AbstractArray{<:Real},Nothing},
  Df::Union{<:Real,AbstractArray{<:Real},Nothing},
)
  (Dg == nothing || Df == nothing) && (return nothing)
  return Dg * Df
end

function hessian_chain_rule(
  Dg::Union{AbstractArray{T},Nothing},
  Hg::Union{AbstractArray{T},Nothing},
  Df::Union{AbstractArray{T},Nothing},
  Hf::Union{AbstractArray{T},Nothing},
) where {T}
  cond1 = !(Df == nothing || Hg == nothing)
  cond2 = !(Dg == nothing || Hf == nothing)
  (!cond1 && !cond2) && (return nothing)

  p = size(Dg, 1)
  Hh1 = cond1 ? kron(Df', sparse(1.0 * I, p, p)) * Hg * Df : nothing

  (!cond2) && (return Hh1)

  n = size(Df, 2)
  Hh2 = kron(Dg, sparse(1.0 * I, n, n)) * Hf

  Hh = Hh1 == nothing ? Hh2 : Hh1 + Hh2
  return Hh
end
##$#############################################################################
