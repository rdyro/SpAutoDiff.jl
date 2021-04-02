function reduce_sum(x; dims = nothing)
  @assert dims != nothing
  return dropdims(sum(x; dims = dims); dims = dims)
end

function stack(x_list; dims = 1)
  @assert length(dims) == 1
  if dims == 1
    return reduce(vcat, [reshape(x, 1, size(x)...) for x in x_list])
  elseif dims == ndims(x_list[1]) + 1 || dims == -1
    return reshape(
      reduce(hcat, [reshape(x, :, 1) for x in x_list]),
      size(x_list[1])...,
      :,
    )
  else
    return reduce(
      (a, b) -> cat(a, b; dims = dims),
      [
        reshape(x, size(x)[1:(dims - 1)]..., 1, size(x)[dims:end]...)
        for x in x_list
      ],
    )
  end
end
