function [controller, total_costs, info] = al_iLQR_1( ...
        ic, initial_controls, ilqr_iters, regularizer, ...
        dyn, costfn, term_costfn, con_params, al_opts, verbose)
% AL_ILQR  方案A：内联 iLQR + 每次迭代动态重建变化率代价
%
% 与原版的关键区别：
%   - iLQR 内层不再调用独立的 iLQR()，而是内联实现
%   - 每次前向传播被接受后，用新轨迹的 u(t-1) 重建变化率约束代价
%   - 后向传播始终使用最新的代价函数 → 梯度/Hessian 与实际轨迹一致

if nargin < 9  || isempty(al_opts),  al_opts = struct; end
if nargin < 10,                      verbose = true;   end

% ---- AL 参数 ----
al_iters = get_field(al_opts, 'al_iters', 10);
mu_max   = get_field(al_opts, 'mu_max',   1e8);
beta     = get_field(al_opts, 'beta',     10);
con_tol  = get_field(al_opts, 'con_tol',  1e-3);

% ---- 分层罚参数 ----
mu_state = get_field(al_opts, 'mu0',      1);
mu_ctrl  = get_field(al_opts, 'mu0_ctrl', 10);
mu_rate  = get_field(al_opts, 'mu0_rate', 100);

% ---- 初始化 ----
N  = size(initial_controls, 1);
n  = size(ic, 1);
m  = size(initial_controls, 2);

nc_state = con_params.nc_state;
nc_ctrl  = con_params.nc_ctrl;
nc_rate  = con_params.nc_rate;
nc_term  = con_params.nc_term;

lambdas     = zeros(N, nc_state + nc_ctrl + nc_rate);
lambda_term = zeros(nc_term, 1);

u_current = initial_controls;

if isfield(al_opts, 'u_init')
    u_prev_init = al_opts.u_init;
else
    u_prev_init = initial_controls(1, :)';
end

% iLQR 收敛阈值
rel_tol = 1e-6;
abs_tol = 1e-8;

% 线搜索步长候选
alpha_list = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001];

% 历史违反量
prev_max_viol_state = inf;
prev_max_viol_ctrl  = inf;
prev_max_viol_rate  = inf;

% ========================================================================
%  AL 外层循环
% ========================================================================
for outer = 1:al_iters

    % === 1. 构建基础 AL 代价（仅状态+控制约束，不含变化率）===
    base_costfn_cells = cell(N, 1);
    for t = 1:N
        lam_base = lambdas(t, 1:nc_state+nc_ctrl)';
        mu_vec_base = [repmat(mu_state, nc_state, 1);
                       repmat(mu_ctrl,  nc_ctrl,  1)];

        base_costfn_cells{t} = make_al_base_cost(...
            costfn, lam_base, mu_vec_base, con_params);
    end

    % 终端代价（只有状态约束，不涉及变化率）
    al_term_costfn = make_al_terminal_cost_layered(...
        term_costfn, lambda_term, repmat(mu_state, nc_term, 1), con_params);

    % === 2. 内联 iLQR 求解（核心改动）===
    [ctrl_out, costs_inner, info_inner] = run_ilqr_with_rate_rebuild( ...
        ic, u_current, ilqr_iters, regularizer, ...
        dyn, base_costfn_cells, al_term_costfn, ...
        lambdas, mu_rate, con_params, u_prev_init, ...
        alpha_list, rel_tol, abs_tol);

    xs = ctrl_out.states;
    us = ctrl_out.controls;

    % === 3. 评估约束违反（分类）===
    [max_viol_state, max_viol_ctrl, max_viol_rate] = eval_constraint_violations(...
        xs, us, con_params, u_prev_init, N, nc_state, nc_ctrl);

    % 终端约束
    [g_term_all, ~, ~] = constraint_eval(xs(end,:)', zeros(m,1), con_params);
    viol_term = max(max(g_term_all(1:nc_term), 0));
    max_viol_state = max(max_viol_state, viol_term);

    max_viol = max([max_viol_state, max_viol_ctrl, max_viol_rate]);

    if verbose
        fprintf('AL %2d | mu:[s=%.1e c=%.1e r=%.1e] | viol:[s=%.2e c=%.2e r=%.2e] | %s\n', ...
            outer, mu_state, mu_ctrl, mu_rate, ...
            max_viol_state, max_viol_ctrl, max_viol_rate, info_inner.status);
    end

    % === 4. 收敛判断 ===
    if max_viol < con_tol
        if verbose
            fprintf('AL converged: max_viol=%.4e < tol=%.4e\n', max_viol, con_tol);
        end
        break;
    end

    % === 5. 更新乘子 ===
    for t = 1:N
        if t == 1, u_prev_t = u_prev_init;
        else,      u_prev_t = us(t-1, :)';
        end
        [g_t, ~, ~] = constraint_eval(xs(t,:)', us(t,:)', con_params, u_prev_t);

        for i = 1:nc_state
            lambdas(t, i) = max(0, lambdas(t,i) + mu_state * g_t(i));
        end
        for i = 1:nc_ctrl
            idx = nc_state + i;
            lambdas(t, idx) = max(0, lambdas(t,idx) + mu_ctrl * g_t(idx));
        end
        for i = 1:nc_rate
            idx = nc_state + nc_ctrl + i;
            lambdas(t, idx) = max(0, lambdas(t,idx) + mu_rate * g_t(idx));
        end
    end

    % 终端乘子
    g_term = g_term_all(1:nc_term);
    lambda_term = max(0, lambda_term + mu_state * g_term);

    % === 6. 分层更新罚参数（更温和的条件）===
    if max_viol_state > 0.5 * prev_max_viol_state
        mu_state = min(mu_state * beta, mu_max);
        if verbose, fprintf('  -> mu_state -> %.2e\n', mu_state); end
    end
    if max_viol_ctrl > 0.5 * prev_max_viol_ctrl
        mu_ctrl = min(mu_ctrl * beta, mu_max);
        if verbose, fprintf('  -> mu_ctrl  -> %.2e\n', mu_ctrl); end
    end
    if max_viol_rate > 0.5 * prev_max_viol_rate
        mu_rate = min(mu_rate * beta, mu_max);
        if verbose, fprintf('  -> mu_rate  -> %.2e\n', mu_rate); end
    end

    prev_max_viol_state = max_viol_state;
    prev_max_viol_ctrl  = max_viol_ctrl;
    prev_max_viol_rate  = max_viol_rate;

    % warm start
    u_current = us;
end

controller   = ctrl_out;
total_costs  = costs_inner;

info = info_inner;
info.al_iters       = min(outer, al_iters);
info.max_viol       = max_viol;
info.mu_final       = mu_state;
info.lambdas        = lambdas;
info.lambda_term    = lambda_term;
info.mu_state_final = mu_state;
info.mu_ctrl_final  = mu_ctrl;
info.mu_rate_final  = mu_rate;
end

% =========================================================================
%  内联 iLQR：每次接受新轨迹后重建变化率约束代价
% =========================================================================
function [controller, total_costs, info] = run_ilqr_with_rate_rebuild( ...
    ic, u_init, iters, regularizer, ...
    dyn, base_costfn_cells, term_costfn, ...
    lambdas, mu_rate, con_params, u_prev_init, ...
    alpha_list, rel_tol, abs_tol)

N = size(u_init, 1);
n = size(ic, 1);
m = size(u_init, 2);

nc_state = con_params.nc_state;
nc_ctrl  = con_params.nc_ctrl;
nc_rate  = con_params.nc_rate;

controller = struct;
controller.K = zeros(N, m, n);
controller.k = zeros(N, m);
controller.states   = zeros(N+1, n);
controller.controls = u_init;

total_costs = nan(iters, 1);

% --- 用当前控制构建完整代价（基础 + 变化率）---
full_cells = rebuild_full_cost_cells(...
    base_costfn_cells, lambdas, mu_rate, con_params, u_init, u_prev_init);

% 初始前向 rollout
[xs, us, J_stage] = fwd_pass_a(ic, controller, dyn, full_cells, term_costfn, con_params);
J_curr = sum(J_stage);

controller.states   = xs;
controller.controls = us;

rho = regularizer;

info = struct;
info.status = 'max_iters';
info.iterations_run = 0;
info.final_cost = J_curr;
info.rho = rho;

for i = 1:iters

    % ★ 每次后向传播前，用最新轨迹重建变化率代价 ★
    full_cells = rebuild_full_cost_cells(...
        base_costfn_cells, lambdas, mu_rate, con_params, us, u_prev_init);

    % 后向传播
    controller_bp = back_pass(xs, us, dyn, full_cells, term_costfn, rho);

    % 线搜索
    accepted  = false;
    best_cost = inf;
    best_xs = []; best_us = []; best_J_stage = []; best_ctrl = [];

    for j = 1:length(alpha_list)
        alpha = alpha_list(j);

        ctrl_try       = controller_bp;
        ctrl_try.k     = alpha * controller_bp.k;
        ctrl_try.controls = us;

        % ★ 前向传播中也用最新代价 ★
        [xs_try, us_try, J_try_stage] = fwd_pass_a(...
            ic, ctrl_try, dyn, full_cells, term_costfn, con_params);
        J_try = sum(J_try_stage);

        if J_try < best_cost
            best_cost    = J_try;
            best_xs      = xs_try;
            best_us      = us_try;
            best_J_stage = J_try_stage;
            best_ctrl    = ctrl_try;
        end

        if J_try < J_curr
            accepted = true;
            break;
        end
    end

    if accepted
        xs     = best_xs;
        us     = best_us;
        J_stage = best_J_stage;
        J_curr = best_cost;

        controller = best_ctrl;
        controller.states   = xs;
        controller.controls = us;

        rho = max(rho / 2, 1e-6);
        total_costs(i) = J_curr;

        abs_imp = abs(sum(J_stage) - J_curr);
        rel_imp = abs_imp / max(1, abs(J_curr));

        if abs_imp < abs_tol || rel_imp < rel_tol
            info.status = 'converged';
            info.iterations_run = i;
            info.final_cost = J_curr;
            info.rho = rho;
            total_costs = total_costs(1:i);
            return;
        end
    else
        rho = min(rho * 10, 1e8);
        total_costs(i) = J_curr;

        if rho >= 1e8
            info.status = 'regularization_limit';
            info.iterations_run = i;
            info.final_cost = J_curr;
            info.rho = rho;
            total_costs = total_costs(1:i);
            return;
        end
    end
end

valid = ~isnan(total_costs);
total_costs = total_costs(valid);
info.status = 'max_iters';
info.iterations_run = length(total_costs);
info.final_cost = J_curr;
info.rho = rho;
end

% =========================================================================
%  重建完整代价 = 基础代价 + 变化率 AL 代价
% =========================================================================
function full_cells = rebuild_full_cost_cells( ...
    base_cells, lambdas, mu_rate, con_params, us, u_prev_init)
% 用当前轨迹的控制序列作为变化率约束的参考点，
% 将基础代价（状态+控制约束）与变化率 AL 代价合并。

N = length(base_cells);
nc_state = con_params.nc_state;
nc_ctrl  = con_params.nc_ctrl;
nc_rate  = con_params.nc_rate;

full_cells = cell(N, 1);
for t = 1:N
    if t == 1
        u_prev_t = u_prev_init;
    else
        u_prev_t = us(t-1, :)';   % ← 当前轨迹的实际控制
    end

    lam_rate_t = lambdas(t, nc_state+nc_ctrl+1 : nc_state+nc_ctrl+nc_rate)';
    mu_vec_rate = repmat(mu_rate, nc_rate, 1);

    full_cells{t} = make_combined_cost( ...
        base_cells{t}, lam_rate_t, mu_vec_rate, con_params, u_prev_t);
end
end

% =========================================================================
%  合并代价函数：基础(状态+控制约束) + 变化率约束
% =========================================================================
function fn = make_combined_cost(base_fn, lam_rate, mu_vec_rate, con, u_prev)
    fn = @(x, u) combined_cost_impl(x, u, base_fn, lam_rate, mu_vec_rate, con, u_prev);
end

function [c, cx, cu, cxx, cxu, cuu] = combined_cost_impl( ...
    x, u, base_fn, lam_rate, mu_vec_rate, con, u_prev)

% 基础代价（原始二次代价 + 状态约束 + 控制约束的 AL 项）
[c, cx, cu, cxx, cxu, cuu] = base_fn(x, u);

% 变化率约束: 只依赖 u 和 u_prev，与 x 无关
du = u - u_prev;
m = length(u);

% 6 个变化率约束值及梯度
g_rate = zeros(6, 1);
gu_rate = zeros(6, m);

g_rate(1) = -du(1) - con.d_alpha_max;   gu_rate(1,1) = -1;
g_rate(2) =  du(1) - con.d_alpha_max;   gu_rate(2,1) =  1;
g_rate(3) = -du(2) - con.d_mu_max;      gu_rate(3,2) = -1;
g_rate(4) =  du(2) - con.d_mu_max;      gu_rate(4,2) =  1;
g_rate(5) = -du(3) - con.d_tss_max;     gu_rate(5,3) = -1;
g_rate(6) =  du(3) - con.d_tss_max;     gu_rate(6,3) =  1;

for i = 1:6
    mu_i = mu_vec_rate(i);
    s = g_rate(i) + lam_rate(i) / mu_i;
    if s > 0
        c   = c   + mu_i/2 * g_rate(i)^2 + lam_rate(i) * g_rate(i);
        coeff = mu_i * g_rate(i) + lam_rate(i);
        cu  = cu  + coeff * gu_rate(i,:)';
        cuu = cuu + mu_i * (gu_rate(i,:)' * gu_rate(i,:));
        % gx = 0 → cx, cxx, cxu 不变
    end
end
end

% =========================================================================
%  基础 AL 阶段代价（仅状态+控制约束，不含变化率）
% =========================================================================
function fn = make_al_base_cost(base_costfn, lam_base, mu_vec_base, con)
    fn = @(x, u) al_base_cost_impl(x, u, base_costfn, lam_base, mu_vec_base, con);
end

function [c, cx, cu, cxx, cxu, cuu] = al_base_cost_impl( ...
    x, u, base_costfn, lam_base, mu_vec_base, con)

[c, cx, cu, cxx, cxu, cuu] = base_costfn(x, u);

nc_base = con.nc_state + con.nc_ctrl;

% 只评估前 nc_base 个约束（不含变化率）
% 直接内联计算，避免调用 constraint_eval 带上不需要的变化率部分
n = length(x);
m = length(u);

g  = zeros(nc_base, 1);
gx = zeros(nc_base, n);
gu = zeros(nc_base, m);

% 状态约束
g(1) = con.z_min - x(3);           gx(1,3) = -1;
g(2) = con.V_min - x(4);           gx(2,4) = -1;
g(3) = x(5) - con.gamma_max;       gx(3,5) =  1;
g(4) = -x(5) - con.gamma_max;      gx(4,5) = -1;

% 控制约束
g(5)  = con.alpha_min - u(1);      gu(5,1)  = -1;
g(6)  = u(1) - con.alpha_max;      gu(6,1)  =  1;
g(7)  = con.mu_min - u(2);         gu(7,2)  = -1;
g(8)  = u(2) - con.mu_max;         gu(8,2)  =  1;
g(9)  = con.tss_min - u(3);        gu(9,3)  = -1;
g(10) = u(3) - con.tss_max;        gu(10,3) =  1;

for i = 1:nc_base
    mu_i = mu_vec_base(i);
    s = g(i) + lam_base(i) / mu_i;
    if s > 0
        c   = c   + mu_i/2 * g(i)^2 + lam_base(i) * g(i);
        coeff = mu_i * g(i) + lam_base(i);
        cx  = cx  + coeff * gx(i,:)';
        cu  = cu  + coeff * gu(i,:)';
        cxx = cxx + mu_i * (gx(i,:)' * gx(i,:));
        cuu = cuu + mu_i * (gu(i,:)' * gu(i,:));
        cxu = cxu + mu_i * (gx(i,:)' * gu(i,:));
    end
end
end

% =========================================================================
%  AL 终端代价（与原版相同）
% =========================================================================
function fn = make_al_terminal_cost_layered(base_term_costfn, lam_term, mu_vec, con)
    fn = @(x) al_terminal_cost_impl(x, base_term_costfn, lam_term, mu_vec, con);
end

function [c, cx, cxx] = al_terminal_cost_impl(x, base_term_costfn, lam_term, mu_vec, con)
[c, cx, cxx] = base_term_costfn(x);
nc_term = con.nc_term;

n = length(x);
g  = zeros(nc_term, 1);
gx = zeros(nc_term, n);

g(1) = con.z_min - x(3);           gx(1,3) = -1;
g(2) = con.V_min - x(4);           gx(2,4) = -1;
g(3) = x(5) - con.gamma_max;       gx(3,5) =  1;
g(4) = -x(5) - con.gamma_max;      gx(4,5) = -1;

for i = 1:nc_term
    mu_i = mu_vec(i);
    s = g(i) + lam_term(i) / mu_i;
    if s > 0
        c   = c   + mu_i/2 * g(i)^2 + lam_term(i) * g(i);
        coeff = mu_i * g(i) + lam_term(i);
        cx  = cx  + coeff * gx(i,:)';
        cxx = cxx + mu_i * (gx(i,:)' * gx(i,:));
    end
end
end

% =========================================================================
%  前向传播（限幅与约束定义一致）
% =========================================================================
function [states, controls, costs] = fwd_pass_a( ...
    x0, controller, dyn, costfn_cells, term_costfn, con)

N = size(controller.k, 1);
n = size(x0, 1);
m = size(controller.k, 2);

states   = zeros(N+1, n);
controls = zeros(N, m);
costs    = zeros(N+1, 1);
states(1,:) = x0(:)';

% 限幅范围与约束定义一致（仅留极小余量防数值问题）
margin = 1.02;
a_lo = con.alpha_min * margin;   a_hi = con.alpha_max * margin;
m_lo = con.mu_min    * margin;   m_hi = con.mu_max    * margin;
t_lo = max(con.tss_min - 0.01, 0);
t_hi = min(con.tss_max + 0.01, 1);

for t = 1:N
    Kt    = squeeze(controller.K(t,:,:));
    x_nom = controller.states(t,:)';
    u_nom = controller.controls(t,:)';
    x_cur = states(t,:)';
    kff   = controller.k(t,:)';

    u = u_nom + Kt * (x_cur - x_nom) + kff;

    u(1) = min(max(u(1), a_lo), a_hi);
    u(2) = min(max(u(2), m_lo), m_hi);
    u(3) = min(max(u(3), t_lo), t_hi);

    controls(t,:) = u';

    [c, ~, ~, ~, ~, ~] = costfn_cells{t}(x_cur, u);
    costs(t) = c;

    [x_next, ~, ~, ~, ~, ~] = dyn(x_cur, u);
    states(t+1,:) = x_next';
end

[tc, ~, ~] = term_costfn(states(end,:)');
costs(end) = tc;
end

% =========================================================================
%  约束违反量统计
% =========================================================================
function [mv_s, mv_c, mv_r] = eval_constraint_violations( ...
    xs, us, con, u_prev_init, N, nc_s, nc_c)

mv_s = 0; mv_c = 0; mv_r = 0;

for t = 1:N
    if t == 1, u_prev_t = u_prev_init;
    else,      u_prev_t = us(t-1,:)';
    end
    [g_t, ~, ~] = constraint_eval(xs(t,:)', us(t,:)', con, u_prev_t);

    mv_s = max(mv_s, max(max(g_t(1:nc_s), 0)));
    mv_c = max(mv_c, max(max(g_t(nc_s+1:nc_s+nc_c), 0)));
    mv_r = max(mv_r, max(max(g_t(nc_s+nc_c+1:end), 0)));
end
end

% =========================================================================
function v = get_field(s, name, default)
    if isfield(s, name), v = s.(name); else, v = default; end
end