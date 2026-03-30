% function [c, c_terminal] = cost_file(Q, R, Qf, target_state, u_trim)
% 
% % cost is a function of (x, u)
% % terminal cost is a function of (x)
% 
%     function [c, cx, cu, cxx, cxu, cuu] = costfn(x, u)
% 
%         n = length(x);
%         m = length(u);
% 
%         % ---------- 参考误差 ----------
%         err_x = x - target_state;
%         err_u = u - u_trim;
% 
%         % ---------- 基础二次代价 ----------
%         c   = 0.5 * (err_x' * Q * err_x + err_u' * R * err_u);
%         cx  = Q * err_x;
%         cu  = R * err_u;
%         cxx = Q;
%         cuu = R;
%         cxu = zeros(n, m);
% 
%         % ---------- 软约束参数 ----------
%         z_min = 2500;                 % 例：最低安全高度
%         V_min = 120;                  % 例：最低安全速度
%         gamma_max = deg2rad(20);      % 最大航迹倾角
% 
%         wz = 1e6;
%         wv = 1e5;
%         wg = 1e5;
% 
%         % ---------- 高度软约束 x(3) >= z_min ----------
%         if x(3) < z_min
%             dz = z_min - x(3);
%             c = c + wz * dz^2;
% 
%             cx(3) = cx(3) - 2*wz*dz;
%             cxx(3,3) = cxx(3,3) + 2*wz;
%         end
% 
%         % ---------- 速度软约束 x(4) >= V_min ----------
%         if x(4) < V_min
%             dv = V_min - x(4);
%             c = c + wv * dv^2;
% 
%             cx(4) = cx(4) - 2*wv*dv;
%             cxx(4,4) = cxx(4,4) + 2*wv;
%         end
% 
%         % ---------- 航迹倾角软约束 |x(5)| <= gamma_max ----------
%         if x(5) > gamma_max
%             dg = x(5) - gamma_max;
%             c = c + wg * dg^2;
% 
%             cx(5) = cx(5) + 2*wg*dg;
%             cxx(5,5) = cxx(5,5) + 2*wg;
%         elseif x(5) < -gamma_max
%             dg = x(5) + gamma_max;
%             c = c + wg * dg^2;
% 
%             cx(5) = cx(5) + 2*wg*dg;
%             cxx(5,5) = cxx(5,5) + 2*wg;
%         end
% 
%     end
% 
%     function [c, cx, cxx] = term_costfn(x)
% 
%         err = x - target_state;
% 
%         c   = 0.5 * (err' * Qf * err);
%         cx  = Qf * err;
%         cxx = Qf;
% 
%         % 如有需要，终端也可加软约束，方法同上
%     end
% 
% c = @costfn;
% c_terminal = @term_costfn;
% 
% end
function [c, c_terminal] = cost_file(Q, R, Qf, target_state, u_trim)

    function [c, cx, cu, cxx, cxu, cuu] = costfn(x, u)
        err_x = x - target_state;
        err_u = u - u_trim;

        c   = 0.5 * (err_x' * Q * err_x + err_u' * R * err_u);
        cx  = Q * err_x;
        cu  = R * err_u;
        cxx = Q;
        cxu = zeros(length(x), length(u));
        cuu = R;
    end

    function [c, cx, cxx] = term_costfn(x)
        err_x = x - target_state;
        c   = 0.5 * (err_x' * Qf * err_x);
        cx  = Qf * err_x;
        cxx = Qf;
    end

    c = @costfn;
    c_terminal = @term_costfn;
end