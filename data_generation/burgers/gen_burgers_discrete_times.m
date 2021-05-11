% Load chebfun package
addpath('~/projects/emulator/src/MATLAB/chebfun')
% number of realizations to generate
% N = 1100;
fprintf(1, "N is set to %i \n", N);
fprintf(1, "seed is set to %i \n", seed);
fprintf(1, "s is set to %i \n", s);
fprintf(1, "tmax is set to %i \n", tmax);
fprintf(1, "Output filepath:\n");
fprintf(1, out_fp);

% set random seed
rng(seed);

% parameters for the Gaussian random field
gamma = 2.5;
tau = 7;
sigma = 7^(2);

% viscosity
visc = 1/5000;

% grid size
% s = 1024;
% steps = 200;


input = zeros(N, s);

output = zeros(N, tmax+1, s);

tspan = linspace(0,tmax,(200*tmax)+1);
x = linspace(0,1,s+1);
for j=1:N
    u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
    u = burgers1(u0, tspan, s, visc);

    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);


    for k=0:tmax
        idx = (k * 200) + 1;
        fprintf(1, "Idx is %i \n", idx);
        output(j,k+1,:) = u{idx}.values;
    end
    disp(j);
end

a=input;
u=output;
% u=output(:,end,:);
% u = reshape(u, [N,s]);

save(out_fp, 'a', 'u')
