function D = NLS_Operator(IC_func, gamma, space, time)
%     Returns an object of class spinop. The operator is Nonlinear Schrodinger
%      Equation on a periodic space domain, time domain [0,1], and Dirichlet BCs.
%     u_t = i / 2 * u_xx - i * gamma * |u|^2 u
%     Inputs:
%           IC_func : chebfun; specifies the IC
%           gamma : float; the coefficient of the nonlinear term. -1 for focusing
%                         and +1 for defocusing.
%           space : [float float]; the spatial boundaries. Periodic boundary
%                         conditions will be enforced.
%           time : array of floats; time points to save evaluations at
    D = spinop(space, time);
    D.lin = @(u) 1i / 2 * diff(u,2);
    D.nonlin = @(u) -1i * gamma * abs(u).^2 .* u ;
    D.init = IC_func;
end
