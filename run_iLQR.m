close all; clear; clc;

%% 1. 基础配置
dt = 0.1;
T_total = 10;
N_sim = round(T_total / dt);

T_pred = 5;
N_pred = round(T_pred / dt);

replan_every = 3;
N_replan = ceil(N_sim / replan_every);

mass = 9100;
g = 9.81;
S = 45;
AeroData = load("Admire_Aero_Data.mat");

%% 2. 初始与目标状态
x_init = [0; 0; 2500; 165.2797216041641; 0; 0];
x_goal = [1200; 700; 2700; 150; 0; 0];

%% 3. 权重
Q  = diag([1e2, 1e2, 1e3, 2.5e4, 1e1, 1e1]);
Qf = 50 * Q;
R  = diag([1e2, 1e2, 1e2]);

%% 4. 动力学
dyn = dynamics_iLQR(mass, g, dt, S, AeroData);

%% 5. 配平控制
u_trim = [0.052559877559830; 0; 0.199700000000000];
u_guess = repmat(u_trim', N_pred, 1);

%% 6. 约束定义
% --- 无禁飞区 ---
con_params = constraint_def();

% --- 有禁飞区 ---
% nfz_opts = struct('enable_nfz', true, 'nfz_cx', 400, 'nfz_cy', 350, 'nfz_r', 100);
% con_params = constraint_def(nfz_opts);

%% 7. AL-iLQR 参数（不再有 mu0_rate）
regularizer = 10;
ilqr_iters  = 50;

al_opts = struct;
al_opts.al_iters = 15;
al_opts.mu0      = 5;       % 状态约束罚参数
al_opts.mu0_ctrl = 50;      % 控制约束罚参数
al_opts.mu_max   = 1e6;
al_opts.beta     = 3;
al_opts.con_tol  = 1e-3;
al_opts.u_init   = u_trim;

%% 8. 记录量
n = length(x_init);
m = length(u_trim);

actual_states   = zeros(N_sim + 1, n);
actual_controls = zeros(N_sim, m);
actual_states(1, :) = x_init';

planned_states   = zeros(N_replan, N_pred + 1, n);
planned_controls = zeros(N_replan, N_pred, m);
replan_steps     = zeros(N_replan, 1);

ddp_cost_history = cell(N_replan, 1);

x_cur = x_init;
replan_id = 0;

current_plan_controls = repmat(u_trim', N_pred, 1);
plan_idx = 1;

u_last_applied = u_trim;

%% 9. 滚动时域主循环
for k = 1:N_sim

    do_replan = (k == 1) || mod(k-1, replan_every) == 0;

    if do_replan
        replan_id = replan_id + 1;

        progress_end = min((k + N_pred - 1) / N_sim, 1.0);
        target_local = x_init + progress_end * (x_goal - x_init);

        [costfn_k, term_costfn_k] = cost_file(Q, R, Qf, target_local, u_trim);

        al_opts.u_init = u_last_applied;

        % ====== 调用 AL-iLQR ======
        [controller, total_costs, info] = al_iLQR( ...
            x_cur, u_guess, ilqr_iters, regularizer, ...
            dyn, costfn_k, term_costfn_k, con_params, al_opts, true);

        ddp_cost_history{replan_id} = total_costs;
        planned_states(replan_id, :, :)  = controller.states;
        planned_controls(replan_id, :, :) = controller.controls;
        replan_steps(replan_id) = k;

        current_plan_controls = controller.controls;
        plan_idx = 1;

        fprintf('\n===== MPC step %02d/%02d | sim %03d/%03d | t=%.2fs =====\n', ...
            replan_id, N_replan, k, N_sim, (k-1)*dt);
        fprintf('AL iters=%d | max_viol=%.4e | mu=%.2e | iLQR: %s\n', ...
            info.al_iters, info.max_viol, info.mu_final, info.status);
        fprintf('state = [%.2f, %.2f, %.2f, %.2f, %.4f, %.4f]\n', ...
            x_cur(1), x_cur(2), x_cur(3), x_cur(4), x_cur(5), x_cur(6));

        shift = min(replan_every, N_pred - 1);
        u_guess = [controller.controls(shift+1:end, :);
                   repmat(controller.controls(end, :), shift, 1)];
    end

    % 执行控制（同样做硬限幅）
    u_apply = current_plan_controls(plan_idx, :)';

    % 控制边界
    u_apply(1) = min(max(u_apply(1), con_params.alpha_min), con_params.alpha_max);
    u_apply(2) = min(max(u_apply(2), con_params.mu_min),    con_params.mu_max);
    u_apply(3) = min(max(u_apply(3), con_params.tss_min),   con_params.tss_max);

    % 变化率硬限幅
    du = u_apply - u_last_applied;
    du(1) = max(min(du(1),  con_params.d_alpha_max), -con_params.d_alpha_max);
    du(2) = max(min(du(2),  con_params.d_mu_max),    -con_params.d_mu_max);
    du(3) = max(min(du(3),  con_params.d_tss_max),   -con_params.d_tss_max);
    u_apply = u_last_applied + du;

    % 限幅后再确保边界
    u_apply(1) = min(max(u_apply(1), con_params.alpha_min), con_params.alpha_max);
    u_apply(2) = min(max(u_apply(2), con_params.mu_min),    con_params.mu_max);
    u_apply(3) = min(max(u_apply(3), con_params.tss_min),   con_params.tss_max);

    [x_next, ~, ~, ~, ~, ~] = dyn(x_cur, u_apply);

    actual_controls(k, :) = u_apply';
    actual_states(k + 1, :) = x_next';

    u_last_applied = u_apply;
    x_cur = x_next;
    plan_idx = min(plan_idx + 1, N_pred);
end

%% 10. 结果检查
final_state = actual_states(end, 1:3);
target_values = x_goal(1:3)';
abs_error = final_state - target_values;

fprintf('\n===== FINAL RESULT =====\n');
fprintf('Final: x=%.4f, y=%.4f, z=%.4f\n', final_state);
fprintf('Error: x=%.4f m, y=%.4f m, z=%.4f m\n', abs(abs_error));

%% 11. AL 约束违反量检查（不含变化率）
fprintf('\n===== AL CONSTRAINT SATISFACTION =====\n');
max_viol_all = 0;
for k = 1:N_sim
    [g_k, ~, ~] = constraint_eval( ...
        actual_states(k,:)', actual_controls(k,:)', con_params);
    max_viol_all = max(max_viol_all, max(max(g_k, 0)));
end
fprintf('Max AL constraint violation: %.6e\n', max_viol_all);

% 禁飞区专项检查
if con_params.enable_nfz
    min_dist_nfz = inf;
    for k = 1:N_sim+1
        dx_nfz = actual_states(k,1) - con_params.nfz_cx;
        dy_nfz = actual_states(k,2) - con_params.nfz_cy;
        dist_k = sqrt(dx_nfz^2 + dy_nfz^2);
        min_dist_nfz = min(min_dist_nfz, dist_k);
    end
    fprintf('NFZ: center=(%.0f,%.0f), r=%.0f m | min dist=%.2f m (margin=%.2f m)\n', ...
        con_params.nfz_cx, con_params.nfz_cy, con_params.nfz_r, ...
        min_dist_nfz, min_dist_nfz - con_params.nfz_r);
end

%% 12. 变化率硬约束检查（应 100% 满足）
du_actual = diff(actual_controls, 1, 1);
n_viol_alpha = sum(abs(du_actual(:,1)) > con_params.d_alpha_max + 1e-10);
n_viol_mu    = sum(abs(du_actual(:,2)) > con_params.d_mu_max    + 1e-10);
n_viol_tss   = sum(abs(du_actual(:,3)) > con_params.d_tss_max   + 1e-10);

fprintf('\n===== RATE HARD CONSTRAINT CHECK (should be 0) =====\n');
fprintf('Alpha rate violations: %d / %d\n', n_viol_alpha, size(du_actual,1));
fprintf('Mu    rate violations: %d / %d\n', n_viol_mu,    size(du_actual,1));
fprintf('TSS   rate violations: %d / %d\n', n_viol_tss,   size(du_actual,1));

%% ====== 绘图 ======

%% 13. 3D 轨迹
figure('Name','3D Trajectory','NumberTitle','off');
hold on; grid on; box on;
plot3(x_init(1), x_init(2), x_init(3), 'g.', 'MarkerSize', 20, 'DisplayName', 'Start');
plot3(x_goal(1), x_goal(2), x_goal(3), 'rx', 'MarkerSize', 20, 'LineWidth', 2, 'DisplayName', 'Goal');
plot3(actual_states(:,1), actual_states(:,2), actual_states(:,3), ...
      'b-', 'LineWidth', 1.5, 'DisplayName', 'MPC Closed-loop');
for i = 1:replan_id
    ps = squeeze(planned_states(i,:,:));
    plot3(ps(:,1), ps(:,2), ps(:,3), '--r', 'LineWidth', 0.8, 'HandleVisibility', 'off');
end
if con_params.enable_nfz
    theta_nfz = linspace(0, 2*pi, 100);
    nfz_x = con_params.nfz_cx + con_params.nfz_r * cos(theta_nfz);
    nfz_y = con_params.nfz_cy + con_params.nfz_r * sin(theta_nfz);
    z_lo = min(actual_states(:,3)) - 100;
    z_hi = max(actual_states(:,3)) + 100;
    [TH, ZZ] = meshgrid(theta_nfz, linspace(z_lo, z_hi, 2));
    CX = con_params.nfz_cx + con_params.nfz_r * cos(TH);
    CY = con_params.nfz_cy + con_params.nfz_r * sin(TH);
    surf(CX, CY, ZZ, 'FaceColor','r','FaceAlpha',0.15,'EdgeColor','none','DisplayName','NFZ');
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('MPC Trajectory'); legend('Location','best'); axis equal; view(3);

%% 14. 状态图
figure('Name','States','NumberTitle','off');
t = linspace(0, T_total, size(actual_states,1));
subplot(2,2,1); hold on;
plot(t, actual_states(:,1), 'LineWidth',1.2);
plot(t, actual_states(:,2), 'LineWidth',1.2);
plot(t, actual_states(:,3), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Position'); legend('x','y','z'); title('Positions'); grid on; box on;
subplot(2,2,2);
plot(t, actual_states(:,4), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('V (m/s)'); title('Velocity'); grid on; box on;
subplot(2,2,3); hold on;
plot(t, actual_states(:,5), 'LineWidth',1.2);
plot(t, actual_states(:,6), 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('rad'); legend('\gamma','\psi'); title('Angles'); grid on; box on;

%% 15. 控制图
t_u = linspace(0, T_total - dt, size(actual_controls,1));
figure('Name','Controls','NumberTitle','off');
subplot(2,2,1); hold on;
plot(t_u, actual_controls(:,1)*180/pi, 'LineWidth',0.8);
plot(t_u, actual_controls(:,2)*180/pi, 'LineWidth',0.8);
xlabel('Time (s)'); ylabel('deg'); legend('\alpha','\mu'); title('Angle Inputs'); grid on; box on;
subplot(2,2,2);
plot(t_u, actual_controls(:,3), 'LineWidth',0.8);
xlabel('Time (s)'); ylabel('TSS'); title('Throttle'); grid on; box on;

%% 16. 控制变化率
t_du = linspace(dt, T_total - dt, size(du_actual,1));
d_alpha_lim = con_params.d_alpha_max * 180/pi;
d_mu_lim    = con_params.d_mu_max    * 180/pi;
d_tss_lim   = con_params.d_tss_max;

figure('Name','Control Rates','NumberTitle','off');
subplot(3,1,1); hold on; grid on; box on;
plot(t_du, du_actual(:,1)*180/pi, 'b-', 'LineWidth',1.2);
yline( d_alpha_lim, 'r--', 'LineWidth',1);
yline(-d_alpha_lim, 'r--', 'LineWidth',1);
xlabel('Time (s)'); ylabel('\Delta\alpha (deg/step)'); title('AoA Rate');
ylim([-d_alpha_lim*1.5, d_alpha_lim*1.5]);

subplot(3,1,2); hold on; grid on; box on;
plot(t_du, du_actual(:,2)*180/pi, 'b-', 'LineWidth',1.2);
yline( d_mu_lim, 'r--', 'LineWidth',1);
yline(-d_mu_lim, 'r--', 'LineWidth',1);
xlabel('Time (s)'); ylabel('\Delta\mu (deg/step)'); title('Bank Rate');
ylim([-d_mu_lim*1.5, d_mu_lim*1.5]);

subplot(3,1,3); hold on; grid on; box on;
plot(t_du, du_actual(:,3), 'b-', 'LineWidth',1.2);
yline( d_tss_lim, 'r--', 'LineWidth',1);
yline(-d_tss_lim, 'r--', 'LineWidth',1);
xlabel('Time (s)'); ylabel('\DeltaTSS'); title('Throttle Rate');
ylim([-d_tss_lim*1.5, d_tss_lim*1.5]);

%% 17. 距离目标
goal_mat = repmat(x_goal', size(actual_states,1), 1);
pos_dist = sqrt(sum((actual_states(:,1:3) - goal_mat(:,1:3)).^2, 2));
figure('Name','Distance to Goal','NumberTitle','off');
plot(linspace(0,T_total,length(pos_dist)), pos_dist, 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Distance (m)'); title('Distance to Goal'); grid on; box on;

%% 18. 2D 俯视图
figure('Name','Top-Down','NumberTitle','off');
hold on; grid on; box on; axis equal;
if con_params.enable_nfz
    theta_plot = linspace(0, 2*pi, 200);
    fill(con_params.nfz_cx + con_params.nfz_r*cos(theta_plot), ...
         con_params.nfz_cy + con_params.nfz_r*sin(theta_plot), ...
         'r', 'FaceAlpha',0.2, 'EdgeColor','r', 'LineWidth',1.5, 'DisplayName','NFZ');
end
plot(actual_states(:,1), actual_states(:,2), 'b-', 'LineWidth',1.8, 'DisplayName','Trajectory');
plot(x_init(1), x_init(2), 'gs', 'MarkerSize',12, 'MarkerFaceColor','g', 'DisplayName','Start');
plot(x_goal(1), x_goal(2), 'rp', 'MarkerSize',14, 'MarkerFaceColor','r', 'DisplayName','Goal');
xlabel('X (m)'); ylabel('Y (m)'); title('Top-Down View'); legend('Location','best');