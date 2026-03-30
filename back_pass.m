function [controller] = back_pass(states, controls, dyn, costfn, term_costfn, regularizer)
% iLQR backward pass
% costfn: 函数句柄 或 cell 数组（AL模式下每步不同）
%
% 修复：每步对 Vxx 做 PSD 截断，防止数值差分 Jacobian 积累的负特征值
% 沿时间反向传播，导致 Quu 不定 → rho 爆炸 → k≈0 → 线搜索失败。

horizon = size(controls, 1);
n = size(states, 2);
m = size(controls, 2);

controller = struct;
controller.K = zeros(horizon, m, n);
controller.k = zeros(horizon, m);
controller.states = states;
controller.controls = controls;

[~, Vx, Vxx] = term_costfn(states(end, :)');
Vx = Vx(:);
Vxx = 0.5 * (Vxx + Vxx');

% ---- [修复] 终端 Vxx 也做 PSD 截断 ----
Vxx = project_psd(Vxx);

rho = regularizer;
rho_scale = 10;
rho_max = 1e10;

use_cell_cost = iscell(costfn);

for t = horizon:-1:1

    x = states(t, :)';
    u = controls(t, :)';

    if use_cell_cost
        [~, lx, lu, lxx, lxu, luu] = costfn{t}(x, u);
    else
        [~, lx, lu, lxx, lxu, luu] = costfn(x, u);
    end

    [~, fx, fu, ~, ~, ~] = dyn(x, u);

    Qx  = lx  + fx' * Vx;
    Qu  = lu  + fu' * Vx;
    Qxx = lxx + fx' * Vxx * fx;
    Qxu = lxu + fx' * Vxx * fu;
    Qux = Qxu';
    Quu = luu + fu' * Vxx * fu;

    Qxx = 0.5 * (Qxx + Qxx');
    Quu = 0.5 * (Quu + Quu');

    success = false;
    rho_local = rho;

    while ~success
        Quu_reg = Quu + rho_local * eye(m);
        [R, p] = chol(Quu_reg);
        if p == 0
            success = true;
        else
            rho_local = rho_local * rho_scale;
            if rho_local > rho_max
                error('back_pass: Quu regularization exceeded maximum.');
            end
        end
    end

    k = -(R \ (R' \ Qu));
    K = -(R \ (R' \ Qux));

    controller.k(t, :) = k';
    controller.K(t, :, :) = reshape(K, [1, m, n]);

    Vx  = Qx  + K' * Quu * k + K' * Qu + Qxu * k;
    Vxx = Qxx + K' * Quu * K + K' * Qux + Qxu * K;
    Vxx = 0.5 * (Vxx + Vxx');

    % ---- [修复] PSD 投影：截断负特征值，阻断数值误差积累 ----
    % 若 Vxx 有负特征值（数值差分 Jacobian 累积误差所致），
    % 下一步 Quu = luu + fu'*Vxx*fu 将不定，rho 被迫爆炸到上限，
    % 导致 k≈0，线搜索失败，iLQR 输出 regularization_limit。
    Vxx = project_psd(Vxx);

    rho = max(rho_local / rho_scale, 1e-9);
end

end

% =========================================================================
%  将对称矩阵投影到正半定锥（截断负特征值为 0）
% =========================================================================
function M_psd = project_psd(M)
    [V, D] = eig(M);
    d = real(diag(D));
    d = max(d, 0);          % 负特征值截为 0
    M_psd = V * diag(d) * V';
    M_psd = 0.5 * (M_psd + M_psd');   % 确保数值对称
end