function [controller, total_costs, info] = iLQR( ...
    ic, initial_controls, iters, regularizer, ...
    dyn, costfn, term_costfn, verbose, rate_cfg)
% ILQR  带线搜索的 iLQR
%
% 参数 1-8 与原版一致。新增第 9 个可选参数:
%   rate_cfg : struct，传递给 fwd_pass 做变化率硬限幅（见 fwd_pass.m）
%              省略或 [] 则不做变化率限幅（向后兼容）。

if nargin < 8 || isempty(verbose),  verbose  = true; end
if nargin < 9,                      rate_cfg = [];   end

N = size(initial_controls, 1);
n = size(ic, 1);
m = size(initial_controls, 2);

controller = struct;
controller.K        = zeros(N, m, n);
controller.k        = zeros(N, m);
controller.states   = zeros(N + 1, n);
controller.controls = initial_controls;

total_costs = nan(iters, 1);

rel_tol = 1e-6;
abs_tol = 1e-8;

% 初始 forward rollout（带硬限幅）
[xs, us, J_stage] = fwd_pass(ic, controller, dyn, costfn, term_costfn, rate_cfg);
J_curr = sum(J_stage);

controller.states   = xs;
controller.controls = us;

alpha_list = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, ...
              0.05, 0.01, 0.005, 0.001];

rho = regularizer;

info = struct;
info.status         = 'max_iters';
info.iterations_run = 0;
info.final_cost     = J_curr;
info.rho            = rho;

for i = 1:iters
    J_old = J_curr;

    controller_bp = back_pass(xs, us, dyn, costfn, term_costfn, rho);

    accepted  = false;
    best_cost = inf;

    for j = 1:length(alpha_list)
        alpha = alpha_list(j);

        ctrl_try          = controller_bp;
        ctrl_try.k        = alpha * controller_bp.k;
        ctrl_try.controls = us;

        [xs_try, us_try, J_try_stage] = fwd_pass( ...
            ic, ctrl_try, dyn, costfn, term_costfn, rate_cfg);
        J_try = sum(J_try_stage);

        if J_try < best_cost
            best_cost       = J_try;
            best_xs         = xs_try;
            best_us         = us_try;
            best_J_stage    = J_try_stage;
            best_controller = ctrl_try;
        end

        if J_try < J_curr
            accepted = true;
            break;
        end
    end

    if accepted
        xs      = best_xs;
        us      = best_us;
        J_stage = best_J_stage;
        J_curr  = best_cost;

        controller          = best_controller;
        controller.states   = xs;
        controller.controls = us;

        rho = regularizer;
        total_costs(i) = J_curr;

        abs_improve = abs(J_old - J_curr);
        rel_improve = abs_improve / max(1, abs(J_old));

        if abs_improve < abs_tol || rel_improve < rel_tol
            info.status         = 'converged';
            info.iterations_run = i;
            info.final_cost     = J_curr;
            info.rho            = rho;
            total_costs         = total_costs(1:i);
            if verbose
                fprintf('iLQR: %s, iters=%d, cost=%.6e, rho=%.3e\n', ...
                    info.status, i, J_curr, rho);
            end
            return;
        end

    else
        rho = min(rho * 10, 1e8);
        total_costs(i) = J_curr;

        if rho >= 1e8
            info.status         = 'regularization_limit';
            info.iterations_run = i;
            info.final_cost     = J_curr;
            info.rho            = rho;
            total_costs         = total_costs(1:i);
            if verbose
                fprintf('iLQR: %s, iters=%d, cost=%.6e, rho=%.3e\n', ...
                    info.status, i, J_curr, rho);
            end
            return;
        end
    end
end

valid_idx   = ~isnan(total_costs);
total_costs = total_costs(valid_idx);

info.status         = 'max_iters';
info.iterations_run = length(total_costs);
info.final_cost     = J_curr;
info.rho            = rho;

if verbose
    fprintf('iLQR: %s, iters=%d, cost=%.6e, rho=%.3e\n', ...
        info.status, info.iterations_run, J_curr, rho);
end
end