function [L, D] = qidong(z, V, alpha, S, AeroData)
% 气动函数
% 输入:
%   z      : 高度(m)
%   V      : 速度(m/s)
%   alpha  : 迎角(rad)
%   S      : 参考面积(m^2)
%   AeroData: 气动表结构体
% 输出:
%   L      : 升力(N)
%   D      : 阻力(N)

    % 基本保护
    Alpha_deg = alpha * 180/pi;
    z_pos = max(z, 0);
    Vt = max(V, 1e-3);

    % 大气
    [rho, sos] = simple_atmosphere(z_pos);

    % 马赫数
    Mach = Vt / sos;

    % ===== 查表范围裁剪（比 isnan->0.1 更稳）=====
    Mach = min(max(Mach, min(AeroData.Mach_vec)), max(AeroData.Mach_vec));
    Alpha_deg = min(max(Alpha_deg, min(AeroData.Alpha_vec)), max(AeroData.Alpha_vec));

    % 气动系数插值
    CL = interp2(AeroData.Mach_vec, AeroData.Alpha_vec, AeroData.CL_Table, ...
                 Mach, Alpha_deg, 'linear');
    CD = interp2(AeroData.Mach_vec, AeroData.Alpha_vec, AeroData.CD_Table, ...
                 Mach, Alpha_deg, 'linear');

    % 最后兜底
    if isnan(CL), CL = 0.1; end
    if isnan(CD), CD = 0.1; end

    % 升阻力
    Q_dyn = 0.5 * rho * Vt^2;
    L = Q_dyn * S * CL;
    D = Q_dyn * S * CD;
end

function [rho, sos] = simple_atmosphere(h)
    if h < 11000
        T = 288.15 - 0.0065 * h;
        P = 101325 * (T/288.15)^(5.2561);
        rho = P / (287.05 * T);
        sos = sqrt(1.4 * 287.05 * T);
    else
        T = 216.65;
        P = 22632 * exp(-0.0001576 * (h - 11000));
        rho = P / (287.05 * T);
        sos = sqrt(1.4 * 287.05 * T);
    end
end