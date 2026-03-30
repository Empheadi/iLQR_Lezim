function [g, gx, gu] = constraint_eval(x, u, con)
% CONSTRAINT_EVAL  计算 AL 约束 g_i(x,u) <= 0 的值及一阶导数
%
% 变化率约束已移除（由 fwd_pass 硬限幅保证），不再需要 u_prev 参数。
%
% 输入:
%   x   : n x 1  状态
%   u   : m x 1  控制
%   con :        约束参数（来自 constraint_def）
%
% 输出:
%   g   : nc x 1   约束值
%   gx  : nc x n   对状态的雅可比
%   gu  : nc x m   对控制的雅可比

n  = length(x);
m  = length(u);
nc = con.nc;

g  = zeros(nc, 1);
gx = zeros(nc, n);
gu = zeros(nc, m);

% ---- 状态约束 (基础 4 个) ----
g(1) = con.z_min - x(3);           gx(1,3) = -1;
g(2) = con.V_min - x(4);           gx(2,4) = -1;
g(3) = x(5) - con.gamma_max;       gx(3,5) =  1;
g(4) = -x(5) - con.gamma_max;      gx(4,5) = -1;

idx = 4;

% ---- 禁飞区约束 (可选) ----
if con.enable_nfz
    idx = idx + 1;
    dx_nfz = x(1) - con.nfz_cx;
    dy_nfz = x(2) - con.nfz_cy;
    % g = r^2 - dist^2 <= 0  →  要求 dist >= r
    g(idx)    = con.nfz_r^2 - dx_nfz^2 - dy_nfz^2;
    gx(idx,1) = -2 * dx_nfz;
    gx(idx,2) = -2 * dy_nfz;
end

% ---- 控制约束 (6 个) ----
g(idx+1) = con.alpha_min - u(1);    gu(idx+1, 1) = -1;
g(idx+2) = u(1) - con.alpha_max;    gu(idx+2, 1) =  1;
g(idx+3) = con.mu_min - u(2);       gu(idx+3, 2) = -1;
g(idx+4) = u(2) - con.mu_max;       gu(idx+4, 2) =  1;
g(idx+5) = con.tss_min - u(3);      gu(idx+5, 3) = -1;
g(idx+6) = u(3) - con.tss_max;      gu(idx+6, 3) =  1;

end