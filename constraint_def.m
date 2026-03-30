function [con_params] = constraint_def(opts)
% CONSTRAINT_DEF  定义 AL 处理的不等式约束 g_i(x,u) <= 0
%
% 变化率约束已从 AL 中移除，改为 fwd_pass 硬限幅执行。
% 禁飞区为可选项，通过 opts.enable_nfz = true 开启。
%
% 用法:
%   con = constraint_def();                        % 无禁飞区
%   con = constraint_def(struct('enable_nfz',true)) % 有禁飞区
%
% --- AL 约束列表 ---
%
% 状态约束:
%   c1: z_min   - x(3)       <= 0   高度下界
%   c2: V_min   - x(4)       <= 0   速度下界
%   c3: x(5)    - gamma_max  <= 0   航迹倾角上界
%   c4: -x(5)   - gamma_max  <= 0   航迹倾角下界
%  [c5: nfz_r^2 - dist^2     <= 0   禁飞区 (可选)]
%
% 控制约束:
%   alpha_min - u(1) <= 0,   u(1) - alpha_max <= 0
%   mu_min    - u(2) <= 0,   u(2) - mu_max    <= 0
%   tss_min   - u(3) <= 0,   u(3) - tss_max   <= 0

if nargin < 1, opts = struct; end

con_params = struct;

% ---- 状态约束参数 ----
con_params.z_min     = 2500;
con_params.V_min     = 120;
con_params.gamma_max = deg2rad(30);

% ---- 控制约束参数 ----
con_params.alpha_min = -15 * pi/180;
con_params.alpha_max =  25 * pi/180;
con_params.mu_min    = -30 * pi/180;
con_params.mu_max    =  30 * pi/180;
con_params.tss_min   = 0.08;
con_params.tss_max   = 1.0;

% ---- 控制变化率参数（fwd_pass 硬限幅用，不进 AL）----
con_params.d_alpha_max = 5  * pi/180;
con_params.d_mu_max    = 10  * pi/180;
con_params.d_tss_max   = 0.1;

% ---- 禁飞区（可选）----
if isfield(opts, 'enable_nfz') && opts.enable_nfz
    con_params.enable_nfz = true;
    con_params.nfz_cx = get_field(opts, 'nfz_cx', 400);
    con_params.nfz_cy = get_field(opts, 'nfz_cy', 350);
    con_params.nfz_r  = get_field(opts, 'nfz_r',  100);

    con_params.nc_state = 5;       % 4 基础 + 1 禁飞区
    con_params.nc_ctrl  = 6;
    con_params.nc       = 11;
    con_params.nc_term  = 5;
else
    con_params.enable_nfz = false;
    con_params.nc_state = 4;
    con_params.nc_ctrl  = 6;
    con_params.nc       = 10;
    con_params.nc_term  = 4;
end

end

function v = get_field(s, name, default)
    if isfield(s, name), v = s.(name); else, v = default; end
end