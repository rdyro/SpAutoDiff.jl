using Profile
using BenchmarkTools, StatProfilerHTML, ProfileView

include(joinpath(@__DIR__, "../src/naked_SpAutoDiff.jl"))
if !@isdefined ID
  using IntelligentDriving
  ID = IntelligentDriving
end

function scp_sad_fns_gen(obj_fn::Function)
  args_ref = Ref{Tuple}()

  function f_fn(U, X_prev, U_prev, reg, Ft, ft, params...)
    udim, N = size(U_prev)
    xdim = size(X_prev, 1)

    U = reshape(U, udim, N)
    X = reshape(Ft * reshape(U, :) + ft, xdim, N)
    J = obj_fn(X, U, params...)
    reg_x, reg_u = reg
    dX, dU = X[:, 1:(end - 1)] - X_prev[:, 2:end], U - U_prev
    J_reg_x = 0.5 * reg_x * dot(reshape(dX, :), reshape(dX, :))
    J_reg_u = 0.5 * reg_u * dot(reshape(dU, :), reshape(dU, :))

    return J + J_reg_x + J_reg_u
  end

  function f_fn_(arg1)
    return f_fn(arg1, args_ref[]...)
  end

  function g_fn!(ret, arg1)
    arg1_ = Tensor(arg1)
    ret_ = f_fn(arg1_, args_ref[]...)
    J = compute_jacobian(ret_, arg1_)
    ret[:] = J[:]
    return
  end

  function h_fn!(ret, arg1)
    arg1_ = Tensor(arg1)
    ret_ = f_fn(arg1_, args_ref[]...)
    J, H = compute_hessian(ret_, arg1_)
    ret[:, :] = H
  end

  return f_fn_, g_fn!, h_fn!, args_ref
end

function quad_prod(x, Q_diag)
  return dot(reshape(Q_diag, :), reshape(x, :) .^ 2)
end

function obj_fn(X, U, params...)                                                 
  Q_diag, R_diag, X_ref, U_ref = params                                          
  xdim, N = size(X_ref)                                                          
  X = X[:, (end - N + 1):end]                                                    
  #Jx = sum(quad_prod(X - X_ref, Q_diag))                                        
  #Ju = sum(quad_prod(U - U_ref, R_diag))                                        
  Jx = quad_prod(X - X_ref, Q_diag)                                              
  Ju = quad_prod(U - U_ref, R_diag)                                              
  J = Jx + Ju                                                                    
  return J                                                                       
end

xdim, udim, N = 4, 2, 20
x0 = 0.0 * ones(4)
X_prev, U_prev = repeat(x0, 1, N), zeros(udim, N)
U = 1e-3 * randn(udim, N)
P = repeat(reshape([0.1, 1.0, 1.0], :, 1), 1, N)

Q_diag = repeat([1e0, 1e0, 1e-3, 1e-3], 1, N)
R_diag = repeat(1e-2 * ones(udim), 1, N)
X_ref, U_ref = 2 * ones(xdim, N), zeros(udim, N)
params = Q_diag, R_diag, X_ref, U_ref

f = stack(
  map(i -> ID.unicycle_f(X_prev[:, i], U_prev[:, i], P[:, i]), 1:N);
  dims = 2,
)
fx = stack(
  map(i -> ID.unicycle_fx(X_prev[:, i], U_prev[:, i], P[:, i]), 1:N);
  dims = 3,
)
fu = stack(
  map(i -> ID.unicycle_fu(X_prev[:, i], U_prev[:, i], P[:, i]), 1:N);
  dims = 3,
)

if !@isdefined f_fn
  f_fn, g_fn!, h_fn!, args_ref = scp_sad_fns_gen(obj_fn)
end

function test()
  reg = (1e-1, 1e-1)
  Ft, ft = ID.linearized_dynamics(x0, f, fx, fu, X_prev, U_prev)
  args_ref[] = (X_prev, U_prev, reg, Ft, ft, Q_diag, R_diag, X_ref, U_ref)
  global obj = f_fn(Tensor(U))
  global J, H = zeros(length(U)), zeros(length(U), length(U))
  g_fn!(J, U)
  h_fn!(H, U)

  @time h_fn!(H, U)
  @time h_fn!(H, U)
  @profview h_fn!(H, U)

  #@time f_fn(Tensor(U))
  #@time f_fn(Tensor(U))
  #@time f_fn(U)
  #@time f_fn(U)
  #@time g_fn!(J, U)
  #@time g_fn!(J, U)
  #@profilehtml g_fn!(J, U)
  #@btime $g_fn!($J, $U)

  #global J2 = jacobian_gen(f_fn)(U)

  return
end
test()
