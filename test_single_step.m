% TEST_SINGLE_STEP  Run ONE MPC planning step with full diagnostics.
%
% Use this script to locate where the solver gets stuck.
% It isolates the first planning call and times each component.

close all; clear; clc;

fprintf('========== test_single_step: Setting up... ==========\n');

%% 1. Setup (identical to run_iLQR.m)
dt = 0.1;
T_pred = 5;
N_pred = round(T_pred / dt);

mass = 9100;
g = 9.81;
S = 45;
AeroData = load("Admire_Aero_Data.mat");

x_init = [0; 0; 2500; 165.2797216041641; 0; 0];
x_goal = [1200; 700; 2700; 150; 0; 0];

Q  = diag([1e2, 1e2, 1e3, 2.5e4, 1e1, 1e1]);
Qf = 50 * Q;
R  = diag([1e2, 1e2, 1e2]);

dyn = dynamics_iLQR(mass, g, dt, S, AeroData);

u_trim = [0.052559877559830; 0; 0.199700000000000];
u_guess = repmat(u_trim', N_pred, 1);

con_params = constraint_def();

regularizer = 10;
ilqr_iters  = 50;

al_opts = struct;
al_opts.al_iters = 15;
al_opts.mu0      = 5;
al_opts.mu0_ctrl = 50;
al_opts.mu_max   = 1e6;
al_opts.beta     = 3;
al_opts.con_tol  = 1e-3;
al_opts.u_init   = u_trim;

%% 2. Build cost function for first MPC step
progress_end = min(N_pred / (round(10/dt)), 1.0);
target_local = x_init + progress_end * (x_goal - x_init);
[costfn, term_costfn] = cost_file(Q, R, Qf, target_local, u_trim);

fprintf('========== Setup complete ==========\n');
fprintf('  x_init  = [%.1f, %.1f, %.1f, %.2f, %.4f, %.4f]\n', x_init);
fprintf('  target  = [%.1f, %.1f, %.2f, %.2f, %.4f, %.4f]\n', target_local);
fprintf('  N_pred  = %d steps (%.1f s horizon)\n', N_pred, T_pred);
fprintf('  iLQR: max %d iters, regularizer = %.1f\n', ilqr_iters, regularizer);
fprintf('  AL:   max %d outer iters\n\n', al_opts.al_iters);

%% 3. Test dynamics (sanity check)
fprintf('--- Test: dynamics at trim ---\n');
tic;
[x_next, fx, fu, ~, ~, ~] = dyn(x_init, u_trim);
t_dyn = toc;
fprintf('  One dynamics+Jacobian call: %.4f s\n', t_dyn);
fprintf('  x_next = [%.2f, %.2f, %.2f, %.4f, %.6f, %.6f]\n', x_next);
fprintf('  ||fx|| = %.4e, ||fu|| = %.4e\n', norm(fx, 'fro'), norm(fu, 'fro'));
fprintf('  Estimated time for %d forward passes: %.1f s\n\n', ...
    N_pred, t_dyn * N_pred);

%% 4. Test backward pass alone
fprintf('--- Test: single backward pass (trim trajectory) ---\n');
% Build a trajectory from open-loop trim
xs_test = zeros(N_pred + 1, 6);
us_test = repmat(u_trim', N_pred, 1);
xs_test(1, :) = x_init';
for t = 1:N_pred
    [xn, ~, ~, ~, ~, ~] = dyn(xs_test(t,:)', us_test(t,:)');
    xs_test(t+1, :) = xn';
end

tic;
[ctrl_bp, dV_test] = back_pass(xs_test, us_test, dyn, costfn, term_costfn, regularizer);
t_bp = toc;
fprintf('  Backward pass time: %.3f s\n', t_bp);
fprintf('  dV = [%.4e, %.4e]\n', dV_test(1), dV_test(2));
fprintf('  max|k| = %.4e, max|K| = %.4e\n', ...
    max(abs(ctrl_bp.k(:))), max(abs(ctrl_bp.K(:))));

if dV_test(1) > 0
    fprintf('  WARNING: dV(1) > 0 means backward pass did NOT find a descent direction!\n');
end
fprintf('\n');

%% 5. Test single forward pass
fprintf('--- Test: single forward pass (alpha=1.0) ---\n');
ctrl_bp.controls = us_test;
tic;
[xs_fp, us_fp, J_fp] = fwd_pass(x_init, ctrl_bp, dyn, costfn, term_costfn);
t_fp = toc;
J_total = sum(J_fp);
fprintf('  Forward pass time: %.3f s\n', t_fp);
fprintf('  Total cost J = %.6e\n', J_total);

% Compare with nominal cost
J_nom = 0;
for t = 1:N_pred
    [c, ~, ~, ~, ~, ~] = costfn(xs_test(t,:)', us_test(t,:)');
    J_nom = J_nom + c;
end
[ct, ~, ~] = term_costfn(xs_test(end,:)');
J_nom = J_nom + ct;
fprintf('  Nominal cost (trim) = %.6e\n', J_nom);
fprintf('  Improvement = %.6e (%.2f%%)\n\n', J_nom - J_total, ...
    100 * (J_nom - J_total) / max(1, abs(J_nom)));

%% 6. Run full al_iLQR with verbose=true
fprintf('========== Running al_iLQR (verbose=true) ==========\n\n');
tic;
[controller, total_costs, info] = al_iLQR( ...
    x_init, u_guess, ilqr_iters, regularizer, ...
    dyn, costfn, term_costfn, con_params, al_opts, true);
t_al = toc;

fprintf('\n========== al_iLQR finished ==========\n');
fprintf('  Total time:    %.2f s\n', t_al);
fprintf('  Status:        %s\n', info.status);
fprintf('  AL iters:      %d\n', info.al_iters);
fprintf('  Max violation:  %.4e\n', info.max_viol);
fprintf('  Final rho:     %.2e\n', info.rho);
fprintf('  Final cost:    %.6e\n', info.final_cost);
fprintf('  mu_state:      %.2e\n', info.mu_state_final);
fprintf('  mu_ctrl:       %.2e\n', info.mu_ctrl_final);

%% 7. Check resulting trajectory
fprintf('\n--- Resulting trajectory check ---\n');
xs_out = controller.states;
us_out = controller.controls;
fprintf('  x(1)   = [%.2f, %.2f, %.2f, %.2f, %.4f, %.4f]\n', xs_out(1,:));
fprintf('  x(end) = [%.2f, %.2f, %.2f, %.2f, %.4f, %.4f]\n', xs_out(end,:));
fprintf('  u range: alpha=[%.2f, %.2f] deg, mu=[%.2f, %.2f] deg, TSS=[%.3f, %.3f]\n', ...
    min(us_out(:,1))*180/pi, max(us_out(:,1))*180/pi, ...
    min(us_out(:,2))*180/pi, max(us_out(:,2))*180/pi, ...
    min(us_out(:,3)), max(us_out(:,3)));

% Constraint check
max_viol = 0;
for t = 1:N_pred
    [g_t, ~, ~] = constraint_eval(xs_out(t,:)', us_out(t,:)', con_params);
    max_viol = max(max_viol, max(max(g_t, 0)));
end
fprintf('  Max constraint violation: %.4e\n', max_viol);

fprintf('\n========== Done. ==========\n');
