%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRINT OUT NECESSARY PARAMETERS
X_BOUNDS = [-pi pi];
fprintf(1, 'X_BOUNDS are [%f, %f]\n', X_BOUNDS(1), X_BOUNDS(2));
fprintf(1, 'MAX_IC_FREQ is %i \n', MAX_IC_FREQ);
fprintf(1, 'N_X_POINTS is %i \n', N_X_POINTS);
fprintf(1, 'N_TRAINING_EXAMPLES is %i \n', N_TRAINING_EXAMPLES);
fprintf(1, 'TMAX is %i \n', TMAX);
fprintf(1, 'STEP_SIZE is %f \n', STEP_SIZE);
fprintf(1, 'GAMMA is %f \n', GAMMA);
fprintf(1, 'SEED is %i \n', SEED);
fprintf(1, 'FP_OUT is %s \n', FP_OUT);
fprintf(1, 'PLOT_PRE is %s \n', PLOT_PRE);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOAD CHEBFUN PACKAGE
addpath('~/projects/emulator/src/MATLAB/chebfun');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET UP TIME / SPACE GRIDS

n_tsteps = TMAX / STEP_SIZE + 1;
n_tsteps_per_unit_time = 1 / STEP_SIZE;
time_grid = linspace(0, TMAX, n_tsteps);
x_grid = linspace(X_BOUNDS(1), X_BOUNDS(2), N_X_POINTS);
output = zeros(N_TRAINING_EXAMPLES, TMAX+1, N_X_POINTS);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATE RANDOM ICs, SOLVE NLS, RECORD RESULTS
rng(SEED);
for i=1:N_TRAINING_EXAMPLES
  IC = random_IC_exp(X_BOUNDS, MAX_IC_FREQ);
  NLS_Op = NLS_Operator(IC, GAMMA, X_BOUNDS, time_grid);
  u = spin(NLS_Op, N_X_POINTS, STEP_SIZE, 'plot', 'off');
  soln_vals = zeros(n_tsteps, N_X_POINTS);
  for j=1:n_tsteps
    soln_vals(j,:) = u{j}.values;
  end
  fname = strcat('NLS_randomIC_', num2str(i));
  fp_plot = strcat(PLOT_PRE, fname, '.png');
  imwrite(abs(soln_vals), fp_plot);
  fprintf(1, 'Making plot %s\n', fname);
  for j=0:TMAX
    j_idx = 1 + (j*n_tsteps_per_unit_time);
    output(i,j+1,:) = u{j_idx}.values;
  end
end
save(FP_OUT, 'output');
