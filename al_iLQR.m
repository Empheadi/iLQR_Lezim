function [controller, total_costs, info] = al_iLQR( ...
        ic, initial_controls, ilqr_iters, regularizer, ...
        dyn, costfn, term_costfn, con_params, al_opts, verbose)
% AL_ILQR  增广拉格朗日 + iLQR
%
% 修复：不再把 rate_cfg 传入内层 iLQR。
% 变化率约束由 fwd_pass 的硬限幅保证——但仅在【执行阶段】（run_iLQR.m）。
% 优化阶段若同时限幅，back_pass 计算的下降方向 k 会被截断，
% fwd_pass 实际代价不降，线搜索对所有 alpha 全部失败 → rho 爆至 1e8。

if nargin < 9  || isempty(al_opts),  al_opts = struct; end
if nargin < 10,                      verbose = true;   end

% ---- AL 参数 ----
al_iters = get_field(al_opts, 'al_iters', 15);
mu_max   = get_field(al_opts, 'mu_max',   1e6);
beta     = get_field(al_opts, 'beta',     3);
con_tol  = get_field(al_opts, 'con_tol',  1e-3);

% ---- 分层罚参数 ----
mu_state = get_field(al_opts, 'mu0',      1);
mu_ctrl  = get_field(al_opts, 'mu0_ctrl', 10);

% ---- 维度 ----
N  = size(initial_controls, 1);
n  = size(ic, 1);
m  = size(initial_controls, 2);

nc_state = con_params.nc_state;
nc_ctrl  = con_params.nc_ctrl;
nc       = con_params.nc;
nc_term  = con_params.nc_term;

% ---- 初始化乘子 ----
lambdas     = zeros(N, nc);
lambda_term = zeros(nc_term, 1);

u_current = initial_controls;

% ---- 历史违反量 ----
prev_max_viol_state = inf;
prev_max_viol_ctrl  = inf;

% ====================================================================
%  AL 外层循环
% ====================================================================
for outer = 1:al_iters

    % === 1. 构建 AL 阶段代价 ===
    al_costfn_cells = cell(N, 1);
    for t = 1:N
        lam_t = lambdas(t, :)';
        mu_vec = [repmat(mu_state, nc_state, 1);
                  repmat(mu_ctrl,  nc_ctrl,  1)];

        al_costfn_cells{t} = make_al_stage_cost( ...
            costfn, lam_t, mu_vec, con_params);
    end

    % === 2. 终端代价 ===
    al_term_costfn = make_al_terminal_cost( ...
        term_costfn, lambda_term, repmat(mu_state, nc_term, 1), con_params);

    % === 3. 调用 iLQR ===
    % 传入控制边界（与 AL 约束一致），但不传变化率字段 → fwd_pass 只做边界限幅，不做变化率限幅。
    % 这保证 fwd_pass 的控制范围与 AL 罚项描述的可行域一致，
    % 消除 backward pass / forward pass 之间的模型—现实不匹配。
    bounds_cfg = struct( ...
        'alpha_min', con_params.alpha_min, 'alpha_max', con_params.alpha_max, ...
        'mu_min',    con_params.mu_min,    'mu_max',    con_params.mu_max, ...
        'tss_min',   con_params.tss_min,   'tss_max',   con_params.tss_max);

    [ctrl_out, costs_inner, info_inner] = iLQR( ...
        ic, u_current, ilqr_iters, regularizer, ...
        dyn, al_costfn_cells, al_term_costfn, verbose, bounds_cfg);

    xs = ctrl_out.states;
    us = ctrl_out.controls;

    % === 4. 评估约束违反 ===
    max_viol_state = 0;
    max_viol_ctrl  = 0;

    for t = 1:N
        [g_t, ~, ~] = constraint_eval(xs(t,:)', us(t,:)', con_params);
        viol_s = max(max(g_t(1:nc_state), 0));
        viol_c = max(max(g_t(nc_state+1:nc), 0));
        max_viol_state = max(max_viol_state, viol_s);
        max_viol_ctrl  = max(max_viol_ctrl,  viol_c);
    end

    % 终端
    u_dummy = zeros(m, 1);
    [g_term_all, ~, ~] = constraint_eval(xs(end,:)', u_dummy, con_params);
    g_term = g_term_all(1:nc_term);
    viol_term = max(max(g_term, 0));
    max_viol_state = max(max_viol_state, viol_term);

    max_viol = max(max_viol_state, max_viol_ctrl);

    if verbose
        fprintf('AL %2d | mu:[s=%.1e c=%.1e] | viol:[s=%.2e c=%.2e] | %s\n', ...
            outer, mu_state, mu_ctrl, ...
            max_viol_state, max_viol_ctrl, info_inner.status);
    end

    % === 5. 收敛判断 ===
    if max_viol < con_tol
        if verbose
            fprintf('AL converged: max_viol=%.4e < tol=%.4e\n', max_viol, con_tol);
        end
        break;
    end

    % === 6. 更新乘子 ===
    for t = 1:N
        [g_t, ~, ~] = constraint_eval(xs(t,:)', us(t,:)', con_params);
        for i = 1:nc_state
            lambdas(t,i) = max(0, lambdas(t,i) + mu_state * g_t(i));
        end
        for i = 1:nc_ctrl
            idx = nc_state + i;
            lambdas(t,idx) = max(0, lambdas(t,idx) + mu_ctrl * g_t(idx));
        end
    end

    lambda_term = max(0, lambda_term + mu_state * g_term);

    % === 7. 更新罚参数 ===
    if max_viol_state > 0.5 * prev_max_viol_state
        mu_state = min(mu_state * beta, mu_max);
        if verbose, fprintf('  -> mu_state -> %.2e\n', mu_state); end
    end
    if max_viol_ctrl > 0.5 * prev_max_viol_ctrl
        mu_ctrl = min(mu_ctrl * beta, mu_max);
        if verbose, fprintf('  -> mu_ctrl  -> %.2e\n', mu_ctrl); end
    end

    prev_max_viol_state = max_viol_state;
    prev_max_viol_ctrl  = max_viol_ctrl;

    % warm start
    u_current = us;
end

controller  = ctrl_out;
total_costs = costs_inner;

info = info_inner;
info.al_iters       = min(outer, al_iters);
info.max_viol       = max_viol;
info.mu_final       = mu_state;
info.lambdas        = lambdas;
info.lambda_term    = lambda_term;
info.mu_state_final = mu_state;
info.mu_ctrl_final  = mu_ctrl;
end

% =========================================================================
%  AL 阶段代价
% =========================================================================
function fn = make_al_stage_cost(base_costfn, lam, mu_vec, con)
    fn = @(x, u) al_stage_cost_impl(x, u, base_costfn, lam, mu_vec, con);
end

function [c, cx, cu, cxx, cxu, cuu] = al_stage_cost_impl( ...
    x, u, base_costfn, lam, mu_vec, con)

[c, cx, cu, cxx, cxu, cuu] = base_costfn(x, u);

[g, gx, gu] = constraint_eval(x, u, con);
nc = length(g);

for i = 1:nc
    mu_i = mu_vec(i);
    s = g(i) + lam(i) / mu_i;
    if s > 0
        c     = c   + mu_i/2 * g(i)^2 + lam(i) * g(i);
        coeff = mu_i * g(i) + lam(i);
        cx    = cx  + coeff * gx(i,:)';
        cu    = cu  + coeff * gu(i,:)';
        cxx   = cxx + mu_i * (gx(i,:)' * gx(i,:));
        cuu   = cuu + mu_i * (gu(i,:)' * gu(i,:));
        cxu   = cxu + mu_i * (gx(i,:)' * gu(i,:));
    end
end
end

% =========================================================================
%  AL 终端代价
% =========================================================================
function fn = make_al_terminal_cost(base_fn, lam_term, mu_vec, con)
    fn = @(x) al_terminal_cost_impl(x, base_fn, lam_term, mu_vec, con);
end

function [c, cx, cxx] = al_terminal_cost_impl(x, base_fn, lam_term, mu_vec, con)
[c, cx, cxx] = base_fn(x);
nc_term = con.nc_term;

u_dummy = zeros(max(con.nc_ctrl, 1), 1);
[g_all, gx_all, ~] = constraint_eval(x, u_dummy, con);

for i = 1:nc_term
    mu_i = mu_vec(i);
    s = g_all(i) + lam_term(i) / mu_i;
    if s > 0
        c     = c   + mu_i/2 * g_all(i)^2 + lam_term(i) * g_all(i);
        coeff = mu_i * g_all(i) + lam_term(i);
        cx    = cx  + coeff * gx_all(i,:)';
        cxx   = cxx + mu_i * (gx_all(i,:)' * gx_all(i,:));
    end
end
end

% =========================================================================
function v = get_field(s, name, default)
    if isfield(s, name), v = s.(name); else, v = default; end
end