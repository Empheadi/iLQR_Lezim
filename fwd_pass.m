function [states, controls, costs] = fwd_pass( ...
    x0, controller, dyn, costfn, term_costfn, rate_cfg)
% FWD_PASS  iLQR 前向传播
%
% 新增可选参数:
%   rate_cfg : struct，硬限幅配置（可选，省略则不做变化率限幅）
%     .d_alpha_max  : 迎角每步最大变化 (rad)
%     .d_mu_max     : 滚转每步最大变化 (rad)
%     .d_tss_max    : 油门每步最大变化
%     .u_prev_init  : m x 1，第一步的参考控制
%     .alpha_min/max, .mu_min/max, .tss_min/max : 控制边界
%
% costfn: 函数句柄 或 cell 数组

if nargin < 6, rate_cfg = []; end
has_rate = ~isempty(rate_cfg);

N = size(controller.k, 1);
n = size(x0, 1);
m = size(controller.k, 2);

states   = zeros(N + 1, n);
controls = zeros(N, m);
costs    = zeros(N + 1, 1);
states(1,:) = x0(:)';

use_cell_cost = iscell(costfn);

% 控制边界（从 rate_cfg 获取或使用宽松默认值）
if has_rate
    a_lo = rate_cfg.alpha_min;   a_hi = rate_cfg.alpha_max;
    m_lo = rate_cfg.mu_min;      m_hi = rate_cfg.mu_max;
    t_lo = rate_cfg.tss_min;     t_hi = rate_cfg.tss_max;
else
    a_lo = -20*pi/180;  a_hi = 30*pi/180;
    m_lo = -45*pi/180;  m_hi = 45*pi/180;
    t_lo = 0;           t_hi = 1;
end

for t = 1:N

    Kt    = squeeze(controller.K(t, :, :));
    x_nom = controller.states(t, :)';
    u_nom = controller.controls(t, :)';
    x_cur = states(t, :)';
    kff   = controller.k(t, :)';

    u = u_nom + Kt * (x_cur - x_nom) + kff;

    % ---- 1. 控制边界限幅 ----
    u(1) = min(max(u(1), a_lo), a_hi);
    u(2) = min(max(u(2), m_lo), m_hi);
    u(3) = min(max(u(3), t_lo), t_hi);

    % ---- 2. 变化率硬限幅（核心改动）----
    if has_rate
        if t == 1
            u_prev = rate_cfg.u_prev_init;
        else
            u_prev = controls(t-1, :)';
        end

        du = u - u_prev;
        du(1) = max(min(du(1),  rate_cfg.d_alpha_max), -rate_cfg.d_alpha_max);
        du(2) = max(min(du(2),  rate_cfg.d_mu_max),    -rate_cfg.d_mu_max);
        du(3) = max(min(du(3),  rate_cfg.d_tss_max),   -rate_cfg.d_tss_max);
        u = u_prev + du;

        % 限幅后再次确保控制边界（变化率限幅可能不会超出，但安全起见）
        u(1) = min(max(u(1), a_lo), a_hi);
        u(2) = min(max(u(2), m_lo), m_hi);
        u(3) = min(max(u(3), t_lo), t_hi);
    end

    controls(t, :) = u';

    if use_cell_cost
        [c, ~, ~, ~, ~, ~] = costfn{t}(x_cur, u);
    else
        [c, ~, ~, ~, ~, ~] = costfn(x_cur, u);
    end
    costs(t) = c;

    [x_next, ~, ~, ~, ~, ~] = dyn(x_cur, u);
    states(t+1, :) = x_next';
end

[terminal_cost, ~, ~] = term_costfn(states(end, :)');
costs(end) = terminal_cost;

end