function [dyn] = dynamics_iLQR(m, g, dt, S, AeroData)

nx = 6;
nu = 3;
eps_x = [1e-2; 1e-2; 1e-2; 1e-3; 1e-4; 1e-4];
eps_u = [1e-4; 1e-4; 1e-3];

% ===== 关键优化: 预构建 griddedInterpolant (比 interp2 快 5-10 倍) =====
% interp2 每次调用都要重建插值结构；griddedInterpolant 只建一次
[Mg, Ag] = ndgrid(AeroData.Mach_vec, AeroData.Alpha_vec);
F_CL = griddedInterpolant(Mg, Ag, AeroData.CL_Table', 'linear', 'nearest');
F_CD = griddedInterpolant(Mg, Ag, AeroData.CD_Table', 'linear', 'nearest');

% 预存查表边界（避免每次 min/max）
Mach_lo = AeroData.Mach_vec(1);    Mach_hi = AeroData.Mach_vec(end);
Alpha_lo = AeroData.Alpha_vec(1);  Alpha_hi = AeroData.Alpha_vec(end);

T_max_sl = 80000;
rho_sl = 1.225;

    function x_dot = continuous_dynamics(state, control)
        Z     = state(3);
        V     = max(state(4), 1e-3);
        Gamma = max(min(state(5), pi/2 - 1e-3), -pi/2 + 1e-3);
        Chi   = state(6);

        Alpha = control(1);
        Mu    = control(2);
        TSS   = control(3);

        z_pos = max(Z, 0);

        % 内联大气模型（避免函数调用开销）
        if z_pos < 11000
            T_atm = 288.15 - 0.0065 * z_pos;
            P_atm = 101325 * (T_atm / 288.15)^5.2561;
        else
            T_atm = 216.65;
            P_atm = 22632 * exp(-0.0001576 * (z_pos - 11000));
        end
        rho = P_atm / (287.05 * T_atm);
        sos = sqrt(1.4 * 287.05 * T_atm);

        % 气动系数 — 用预构建的 griddedInterpolant
        Mach = max(min(V / sos, Mach_hi), Mach_lo);
        Alpha_deg = max(min(Alpha * (180/pi), Alpha_hi), Alpha_lo);

        CL = F_CL(Mach, Alpha_deg);
        CD = F_CD(Mach, Alpha_deg);

        Q_dyn = 0.5 * rho * V^2;
        L_aero = Q_dyn * S * CL;
        D_aero = Q_dyn * S * CD;
        Thrust = T_max_sl * (rho / rho_sl) * TSS;

        cosG = cos(Gamma); sinG = sin(Gamma);
        cosC = cos(Chi);   sinC = sin(Chi);
        cosA = cos(Alpha); sinA = sin(Alpha);
        cosM = cos(Mu);    sinM = sin(Mu);

        TsA_plus_L = Thrust * sinA + L_aero;

        x_dot = zeros(6, 1);
        x_dot(1) = V * cosG * cosC;
        x_dot(2) = V * cosG * sinC;
        x_dot(3) = V * sinG;
        x_dot(4) = (Thrust * cosA - D_aero) / m - g * sinG;
        x_dot(5) = TsA_plus_L * cosM / (m * V) - (g / V) * cosG;
        x_dot(6) = TsA_plus_L * sinM / (m * V * cosG);
    end

    function F = discrete_dynamics(state, control)
        k1 = continuous_dynamics(state, control);
        k2 = continuous_dynamics(state + dt*0.5*k1, control);
        k3 = continuous_dynamics(state + dt*0.5*k2, control);
        k4 = continuous_dynamics(state + dt*k3, control);
        F = state + dt*(k1 + 2*k2 + 2*k3 + k4) / 6;
    end

    % ===== 优化 2: 合并 fx 和 fu 到一次循环, 减少重复计算 =====
    function [fx, fu] = calc_jacobians(state, control)
        f0 = discrete_dynamics(state, control);  % 基准点只算一次

        fx = zeros(nx, nx);
        for i = 1:nx
            h = eps_x(i);
            sp = state; sp(i) = sp(i) + h;
            sm = state; sm(i) = sm(i) - h;
            fx(:, i) = (discrete_dynamics(sp, control) - discrete_dynamics(sm, control)) / (2*h);
        end

        fu = zeros(nx, nu);
        for i = 1:nu
            h = eps_u(i);
            cp = control; cp(i) = cp(i) + h;
            cm = control; cm(i) = cm(i) - h;
            fu(:, i) = (discrete_dynamics(state, cp) - discrete_dynamics(state, cm)) / (2*h);
        end
    end

    dyn = @(state, control) dynamics_wrapper_fast(state(:), control(:), ...
        @discrete_dynamics, @calc_jacobians);
end

function [f, fx, fu, fxx, fxu, fuu] = dynamics_wrapper_fast(state, control, F_fun, jac_fun)
    f = F_fun(state, control);
    [fx, fu] = jac_fun(state, control);
    fxx = []; fxu = []; fuu = [];
end