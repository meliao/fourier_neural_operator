%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOAD CHEBFUN PACKAGE
addpath('~/projects/emulator/src/MATLAB/chebfun');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET TRAINING DATA PARAMETERS
X_BOUNDS = [-pi pi];
MAX_IC_FREQ = 10;
N_X_POINTS = 1024;
N_TRAINING_EXAMPLES = 10;
TMAX = 10;
STEP_SIZE = 0.001;
GAMMA = -1;
FP_OUT = '~/2021_Spring_Courses/Becca_Reading_Group/NLS_solver/data/2021-06-04_Matlab_NLS_data.mat';


PLOT_DIR = '~/projects/fourier_neural_operator/data/2021-06-04_training_data_test_plots/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET UP TIME / SPACE GRIDS

n_tsteps = TMAX / STEP_SIZE + 1;
n_tsteps_per_unit_time = 1 / STEP_SIZE;
time_grid = linspace(0, TMAX, n_tsteps);
x_grid = linspace(X_BOUNDS(1), X_BOUNDS(2), N_X_POINTS);
output = zeros(N_TRAINING_EXAMPLES, TMAX+1, N_X_POINTS);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATE RANDOM ICs
rng(SEED);
for i=1:N_TRAINING_EXAMPLES
  IC = random_IC_sin_cos(X_BOUNDS, MAX_IC_FREQ);
  NLS_Op = NLS_Operator(IC, GAMMA, X_BOUNDS, time_grid);
  u = spin(NLS_Op, N_X_POINTS, STEP_SIZE, 'plot', 'off');
  soln_vals = zeros(n_tsteps, N_X_POINTS);
  for j=1:n_tsteps
    soln_vals(j,:) = u{j}.values;
  end
  fname = strcat('NLS_randomIC_', num2str(i));
  fp_plot = strcat(PLOT_DIR, fname, '.png');
  imwrite(abs(soln_vals), fp_plot);
  fprintf('Making plot ', fname, '\n');
  for j=0:TMAX
    j_idx = 1 + (j*n_tsteps_per_unit_time)
    output(i,j+1,:) = u{j_idx}.values;
  end
end
