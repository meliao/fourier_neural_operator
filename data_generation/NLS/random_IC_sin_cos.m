function out_cfun = random_IC_sin_cos(X_bounds, n_fourier_coef, seed)
%     Generates a random periodic function with controlled frequency on a
%     discrete set of points.
%     Inputs:
%         n_fourier_coef : int. The frequency. This corresponds to the number
%                                 of Fourier coefficients that will be saved.
%         seed : int. The random seed for the RNG
    rng(seed);
    rdata = randn(n_fourier_coef, 4);
    ic_func = @(x) 0;
    for i = 1:n_fourier_coef
        draw = rdata(i,:);
        k = i - 1;
        new_func = @(x)  draw(1) * sin(k * pi * x + draw(2)) + draw(3) * cos(k * pi * x + draw(4));
        ic_func = @(x) ic_func(x) + new_func(x);
    end
    ic_vals = ic_func(linspace(X_bounds(1), X_bounds(2), 1024));
    max_ic_val = max(abs(ic_vals));
    ic_func = @(x) (1 / max_ic_val) * ic_func(x);
    out_cfun = chebfun(ic_func, X_bounds);
end
