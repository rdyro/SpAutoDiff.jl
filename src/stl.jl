lt(sig1, val; scale::Real = 1) = sig1 .- val
gt(sig1, val; scale::Real = 1) = val .- sig1
equal(sig1, val; scale::Real = 1) = -abs.(sig1 .- val)
negation(sig1; scale::Real = 1) = -sig1
and(sig1, sig2; scale::Real = 1) = lseminish(vcat(sig1, sig2); scale = scale)
or(sig1, sig2; scale::Real = 1) = lsemaxish(vcat(sig1, sig2); scale = scale)
always(sig1; scale::Real = 1) = lseminish(sig1; scale = scale)
eventually(sig1; scale::Real = 1) = lsemaxish(sig1; scale = scale)
function until(sig1, sig2; scale::Real = 1, use_outer_max::Bool = false)
  v_list = [
    lseminish(
      vcat(sig2[i], lseminish(sig1[1:i]; scale = scale));
      scale = scale,
    ) for i = 1:length(sig1)
  ]
  v = reduce(vcat, v_list)
  if use_outer_max && isa(v, Tensor)
    idx = argmax(v.value)
    return v_list[idx]
  else
    return lsemaxish(v; scale = scale)
  end
end
