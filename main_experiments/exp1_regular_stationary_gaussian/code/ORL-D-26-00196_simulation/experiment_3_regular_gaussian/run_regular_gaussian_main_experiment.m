function outputs = run_regular_gaussian_main_experiment(mode, varargin)
%RUN_REGULAR_GAUSSIAN_MAIN_EXPERIMENT Main Experiment 3.
% The experiment uses the paper's unchanged unrestricted ETO/IEO methods.
% It adds a stationary, well-conditioned dense regular Gaussian population
% in which both regret paths decline and full-sample ETO is expected to
% outperform split-sample IEO. Existing experiments and outputs are untouched.

if nargin < 1 || isempty(mode)
    mode = 'smoke';
end

config = locked_config(mode, varargin{:});
original_path = path;
path_cleanup = onCleanup(@()path(original_path));
addpath(config.workspace_root, '-end');
addpath(config.code_dir, '-begin');
assert_official_execution_path(config);

[population, design] = build_regular_population(config);
ensure_directory(config.results_dir);
ensure_directory(config.figure_dir);

S = config.NumSims;
T = config.T;
arrays = initialize_arrays(S, T);
completed = false(S, 1);
checkpoint_file = fullfile(config.results_dir, 'checkpoint.mat');

if isfile(checkpoint_file)
    saved = load(checkpoint_file, 'arrays', 'completed', 'config', ...
        'population', 'design');
    assert_checkpoint_compatible(saved, config, population, design);
    arrays = saved.arrays;
    completed = saved.completed;
end

remaining = find(~completed);
if config.use_parallel && ~isempty(remaining)
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', config.parallel_workers);
    end
end

for first = 1:config.batch_size:numel(remaining)
    ids = remaining(first:min(first + config.batch_size - 1, numel(remaining)));
    blocks = cell(numel(ids), 1);
    if config.use_parallel
        parfor block_index = 1:numel(ids)
            blocks{block_index} = one_replication(ids(block_index), ...
                population, config);
        end
    else
        for block_index = 1:numel(ids)
            blocks{block_index} = one_replication(ids(block_index), ...
                population, config);
        end
    end

    for block_index = 1:numel(ids)
        replication_id = ids(block_index);
        arrays = store_replication(arrays, replication_id, ...
            blocks{block_index});
        completed(replication_id) = true;
    end

    atomic_checkpoint(checkpoint_file, arrays, completed, config, ...
        population, design);
    fprintf('Experiment 3 regular Gaussian (%s): %d/%d complete.\n', ...
        config.mode, sum(completed), S);
end

assert(all(completed), 'Not all requested replications completed.');
[stagewise_summary, wealth_summary, experiment_summary] = ...
    aggregate_results(arrays, population, config);
validation = build_validation(arrays, completed, population, config, ...
    experiment_summary);

outputs = struct();
outputs.raw = fullfile(config.results_dir, 'raw_results.mat');
outputs.stagewise = fullfile(config.results_dir, 'stagewise_summary.csv');
outputs.wealth = fullfile(config.results_dir, 'wealth_summary.csv');
outputs.summary = fullfile(config.results_dir, 'experiment_summary.csv');
outputs.validation = fullfile(config.results_dir, 'validation.csv');

save(outputs.raw, 'arrays', 'completed', 'config', 'population', 'design', ...
    'stagewise_summary', 'wealth_summary', 'experiment_summary', ...
    'validation', '-v7.3');
writetable(stagewise_summary, outputs.stagewise);
writetable(wealth_summary, outputs.wealth);
writetable(experiment_summary, outputs.summary);
writetable(validation, outputs.validation);

if config.make_plots
    make_main_figures(stagewise_summary, wealth_summary, config, arrays);
end

disp(experiment_summary);
disp(validation);
clear path_cleanup
end

function config = locked_config(mode, varargin)
mode = validatestring(lower(char(mode)), {'test', 'smoke', 'full'});
experiment_dir = fileparts(mfilename('fullpath'));
package_dir = fileparts(experiment_dir);
workspace_root = fileparts(package_dir);

config = struct();
config.schema_version = 2;
config.mode = mode;
config.experiment_name = 'Exp3_Regular_Gaussian_Dense_Rotated';
config.design_variant = 'dense_rotated_active_0p050';
config.experiment_dir = experiment_dir;
config.package_dir = package_dir;
config.workspace_root = workspace_root;
config.code_dir = workspace_root;
config.T = 50;
config.n = 10;
config.N_A = 60;
config.N_B = 1000;
config.x0 = 1.0;
config.x_eval = 1.0;
config.lambda_scalar = 1e-4;
config.evaluation_ridge = 1e-4;
config.base_seed = 42;
config.marginal_variance = 8.103335969935516e-5;
config.mu_bar = 1.0002370503416058;
config.population_condition_target = 20;
config.population_active_l2_target = 0.05;
config.covariance_construction_seed = 913701;
config.signal_construction_seed = 913702;
config.expected_calibrated_delta = 0.00065978679633078;
config.delta_tolerance = 1e-14;
config.norm_floor = 1e-10;
config.acos_clip_eps = 1e-12;
config.budget_tolerance = 1e-8;
config.condition_tolerance = 1e-10;
config.negative_regret_tolerance = 1e-12;
config.bootstrap_seed = 314159;
config.bootstrap_resamples = 5000;
config.batch_size = 10;
config.parallel_workers = 4;
config.use_parallel = true;
config.make_plots = true;
config.show_figures = false;

switch mode
    case 'test'
        config.NumSims = 2;
        config.replication_seed_offset = 825000;
        config.bootstrap_resamples = 200;
        config.batch_size = 2;
        config.use_parallel = false;
    case 'smoke'
        config.NumSims = 12;
        config.replication_seed_offset = 830000;
        config.bootstrap_resamples = 1000;
        config.batch_size = 4;
        config.use_parallel = false;
    otherwise
        config.NumSims = 2000;
        config.replication_seed_offset = 840000;
end

parser = inputParser;
addParameter(parser, 'NumSims', config.NumSims);
addParameter(parser, 'UseParallel', config.use_parallel);
addParameter(parser, 'ParallelWorkers', config.parallel_workers);
addParameter(parser, 'BatchSize', config.batch_size);
addParameter(parser, 'BootstrapResamples', config.bootstrap_resamples);
addParameter(parser, 'MakePlots', config.make_plots);
addParameter(parser, 'ShowFigures', config.show_figures);
variant_results = fullfile(experiment_dir, 'results', config.design_variant, mode);
variant_figures = fullfile(experiment_dir, 'figures', config.design_variant, mode);
addParameter(parser, 'ResultsDir', variant_results);
addParameter(parser, 'FigureDir', variant_figures);
parse(parser, varargin{:});

config.NumSims = parser.Results.NumSims;
config.use_parallel = logical(parser.Results.UseParallel);
config.parallel_workers = parser.Results.ParallelWorkers;
config.batch_size = parser.Results.BatchSize;
config.bootstrap_resamples = parser.Results.BootstrapResamples;
config.make_plots = logical(parser.Results.MakePlots);
config.show_figures = logical(parser.Results.ShowFigures);
config.results_dir = char(parser.Results.ResultsDir);
config.figure_dir = char(parser.Results.FigureDir);
config.lambda = config.lambda_scalar * ones(1, config.T);

validateattributes(config.NumSims, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(config.batch_size, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(config.bootstrap_resamples, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
if strcmp(mode, 'full')
    assert(config.NumSims == 2000, ...
        'The locked full mode requires exactly 2,000 replications.');
end
end

function assert_official_execution_path(config)
expected = {fullfile(config.code_dir, 'run_eto.m'), ...
    fullfile(config.code_dir, 'run_ieo.m'), ...
    fullfile(config.code_dir, 'calculate_oracle.m'), ...
    fullfile(config.code_dir, 'generate_data.m')};
names = {'run_eto', 'run_ieo', 'calculate_oracle', 'generate_data'};
for index = 1:numel(names)
    actual = which(names{index});
    assert(strcmp(actual, expected{index}), ...
        'Experiment3:WrongMethodPath', ...
        'Expected %s but MATLAB resolves %s.', expected{index}, actual);
end

prohibited = {'run_eto_factor_cov', 'run_ieo_factor_structured', ...
    'build_factor_policy_space'};
for index = 1:numel(prohibited)
    assert(isempty(which(prohibited{index})), ...
        'Experiment3:StructuredEstimatorVisible', ...
        'Structured estimator %s must not be on the path.', ...
        prohibited{index});
end
end

function [population, design] = build_regular_population(config)
n = config.n;
kappa = config.population_condition_target;

covariance_stream = RandStream('mt19937ar', 'Seed', ...
    config.covariance_construction_seed);
[Q, ~] = qr(randn(covariance_stream, n, n), 0);
eigenvalue_shape = logspace(log10(kappa), 0, n)';
eigenvalue_scale = config.marginal_variance * n / sum(eigenvalue_shape);
locked_eigenvalues = eigenvalue_scale * eigenvalue_shape;
Sigma = Q * diag(locked_eigenvalues) * Q';
Sigma = (Sigma + Sigma') / 2;

signal_stream = RandStream('mt19937ar', 'Seed', ...
    config.signal_construction_seed);
signal_direction = randn(signal_stream, n, 1);
signal_direction = signal_direction - mean(signal_direction);
signal_direction = signal_direction / norm(signal_direction, 2);
zero_mean = config.mu_bar * ones(n, 1);
zero_oracle = make_population_oracle(Sigma, zero_mean, config);
zero_policy = zero_oracle.strategy.A_coef(:, 1) + ...
    zero_oracle.strategy.B_coef(:, 1);
delta = calibrate_population_signal(Sigma, signal_direction, zero_policy, config);
mu = zero_mean + delta * signal_direction;
cholesky = chol(Sigma, 'lower');
eigenvalues = eig(Sigma, 'vector');

population_oracle = make_population_oracle(Sigma, mu, config);
mu_path = population_oracle.mu_path;
Sigma_path = population_oracle.Sigma_path;
oracle = population_oracle.strategy;
oracle_debug = population_oracle.debug;
active_l2 = norm(oracle.A_coef(:, 1) + oracle.B_coef(:, 1) - ...
    zero_policy, 2);

population = struct();
population.mu = mu;
population.Sigma = Sigma;
population.cholesky = cholesky;
population.mu_path = mu_path;
population.Sigma_path = Sigma_path;
population.oracle = oracle;
population.oracle_debug = oracle_debug;
population.zero_signal_oracle = zero_oracle.strategy;
population.covariance_basis = Q;
population.eigenvalues = locked_eigenvalues;
population.signal_direction = signal_direction;
population.calibrated_delta = delta;
population.active_l2_stage0 = active_l2;
population.condition_number = cond(Sigma);
population.lambda_min = min(eigenvalues);
population.lambda_max = max(eigenvalues);
population.marginal_variance = mean(diag(Sigma));
population.oracle_budget_A = max(abs(sum(oracle.A_coef, 1) - 1));
population.oracle_budget_B = max(abs(sum(oracle.B_coef, 1)));

design = struct();
design.name = 'stationary_dense_rotated_regular_gaussian_active_0p050';
design.description = ['Stationary iid Gaussian returns with a dense rotated ', ...
    'condition-20 covariance and a calibrated zero-sum mean signal.'];
design.hypothesized_winner = 'ETO';
design.mechanism = ['Dense population geometry produces declining stagewise ', ...
    'regret while full-sample plug-in estimation remains more efficient ', ...
    'than the official IEO two-fold half-sample fitting.'];
design.primary_regret = ['Universe-B hindsight stagewise regret at x=1 ', ...
    'with a 1e-4 Hessian ridge, matching the main experiments.'];
design.wealth_interpretation = ['A calibrated heterogeneous mean signal ', ...
    'creates a visible out-of-sample wealth separation.'];
design.development_disclosure = ['Exploratory DGP selected after a 40-run ', ...
    'screen and confirmed with 200 independent replications; the formal ', ...
    'run uses a third, disjoint seed range.'];

assert(abs(population.condition_number - kappa) <= ...
    config.condition_tolerance, 'Population condition number is not 20.');
assert(population.lambda_min > 0, 'Population covariance must be positive definite.');
assert(abs(population.marginal_variance - config.marginal_variance) < 1e-16, ...
    'Population marginal variance changed.');
assert(abs(population.active_l2_stage0 - ...
    config.population_active_l2_target) < 1e-8, ...
    'Population active-L2 calibration changed.');
assert(abs(population.calibrated_delta - config.expected_calibrated_delta) <= ...
    config.delta_tolerance, 'Calibrated mean-signal delta changed.');
assert(abs(sum(population.signal_direction)) < 1e-12, ...
    'Mean-signal direction must be budget neutral.');
end

function record = make_population_oracle(Sigma, mu, config)
mu_path = repmat(mu', config.T, 1);
Sigma_path = repmat(reshape(Sigma, 1, config.n, config.n), ...
    config.T, 1, 1);
[strategy, debug] = calculate_oracle(mu_path, Sigma_path, config.lambda);
record = struct('strategy', strategy, 'debug', debug, ...
    'mu_path', mu_path, 'Sigma_path', Sigma_path);
end

function delta = calibrate_population_signal(Sigma, direction, zero_policy, config)
objective = @(value) population_active_distance(Sigma, ...
    config.mu_bar * ones(config.n, 1) + value * direction, ...
    zero_policy, config) - config.population_active_l2_target;
lower = 0;
upper = 1e-6;
while objective(upper) < 0 && upper < 0.1
    upper = 2 * upper;
end
assert(objective(upper) >= 0, 'Could not bracket population signal.');
for iteration = 1:70
    midpoint = (lower + upper) / 2;
    if objective(midpoint) >= 0
        upper = midpoint;
    else
        lower = midpoint;
    end
end
delta = (lower + upper) / 2;
end

function distance = population_active_distance(Sigma, mu, zero_policy, config)
candidate = make_population_oracle(Sigma, mu, config);
policy = candidate.strategy.A_coef(:, 1) + ...
    candidate.strategy.B_coef(:, 1);
distance = norm(policy - zero_policy, 2);
end

function arrays = initialize_arrays(S, T)
stage_fields = {'Regret_ETO_sims', 'Regret_IEO_sims', ...
    'TrueRegret_ETO_sims', 'TrueRegret_IEO_sims', ...
    'SCDR_sims', 'RAD_sims', 'B_Angle_sims', 'HSSG_sims'};
for index = 1:numel(stage_fields)
    arrays.(stage_fields{index}) = nan(S, T);
end
arrays.Wealth_Paths_ETO = nan(S, T + 1);
arrays.Wealth_Paths_IEO = nan(S, T + 1);
arrays.max_budget_eto = nan(S, 1);
arrays.max_budget_ieo = nan(S, 1);
arrays.degenerate_intercept_angles = zeros(S, 1);
arrays.degenerate_slope_angles = zeros(S, 1);
arrays.valid = false(S, 1);
arrays.success_eto = false(S, 1);
arrays.success_ieo = false(S, 1);
arrays.error_eto = strings(S, 1);
arrays.error_ieo = strings(S, 1);
end

function one = one_replication(replication_id, population, config)
T = config.T;
n = config.n;
one = struct();
one.Regret_ETO_sims = nan(1, T);
one.Regret_IEO_sims = nan(1, T);
one.TrueRegret_ETO_sims = nan(1, T);
one.TrueRegret_IEO_sims = nan(1, T);
one.SCDR_sims = nan(1, T);
one.RAD_sims = nan(1, T);
one.B_Angle_sims = nan(1, T);
one.HSSG_sims = nan(1, T);
one.Wealth_Paths_ETO = nan(1, T + 1);
one.Wealth_Paths_IEO = nan(1, T + 1);
one.max_budget_eto = NaN;
one.max_budget_ieo = NaN;
one.degenerate_intercept_angles = 0;
one.degenerate_slope_angles = 0;
one.valid = false;
one.success_eto = false;
one.success_ieo = false;
one.error_eto = "";
one.error_ieo = "";

root_seed = config.base_seed + config.replication_seed_offset + replication_id;
seedA = root_seed + 100;
seedB = root_seed + 200;
[R_A, R_B] = generate_data(T, n, population.mu_path, ...
    population.Sigma_path, config.N_A, config.N_B, seedA, seedB); %#ok<ASGLU>

results_eto = [];
results_ieo = [];
try
    evalc('results_eto = run_eto(R_A, R_B, config.lambda, config.x0);');
    one.success_eto = true;
catch exception
    one.error_eto = string(getReport(exception, 'basic', 'hyperlinks', 'off'));
end
try
    evalc('results_ieo = run_ieo(R_A, R_B, config.lambda, config.x0);');
    one.success_ieo = true;
catch exception
    one.error_ieo = string(getReport(exception, 'basic', 'hyperlinks', 'off'));
end
if ~(one.success_eto && one.success_ieo)
    return;
end

A_eto = results_eto.strategy.A_coef;
B_eto = results_eto.strategy.B_coef;
A_ieo = results_ieo.strategy.A_coef;
B_ieo = results_ieo.strategy.B_coef;

[one.Regret_ETO_sims, one.Regret_IEO_sims] = ...
    main_style_regret(results_eto.strategy, results_ieo.strategy, ...
    R_B, config);
[one.TrueRegret_ETO_sims, one.TrueRegret_IEO_sims] = ...
    true_population_regret(results_eto.strategy, results_ieo.strategy, ...
    population, config);

for t = 1:T
    one.SCDR_sims(t) = norm(B_ieo(:, t) - B_eto(:, t), 2);
    one.RAD_sims(t) = norm(A_ieo(:, t) - A_eto(:, t), 2);
    [one.B_Angle_sims(t), is_degenerate] = vector_angle( ...
        B_ieo(:, t), B_eto(:, t), config);
    one.degenerate_intercept_angles = ...
        one.degenerate_intercept_angles + is_degenerate;
    [one.HSSG_sims(t), is_degenerate] = vector_angle( ...
        A_ieo(:, t), A_eto(:, t), config);
    one.degenerate_slope_angles = ...
        one.degenerate_slope_angles + is_degenerate;
end

one.Wealth_Paths_ETO = results_eto.wealth.X_mean(:)';
one.Wealth_Paths_IEO = results_ieo.evaluation.X_mean(:)';
one.max_budget_eto = results_eto.validation.max_violation;
ieo_budget = squeeze(sum(results_ieo.evaluation.U, 1));
one.max_budget_ieo = max(abs(ieo_budget - ...
    results_ieo.evaluation.X(:, 1:T)), [], 'all');

core_values = [one.Regret_ETO_sims, one.Regret_IEO_sims, ...
    one.TrueRegret_ETO_sims, one.TrueRegret_IEO_sims, ...
    one.SCDR_sims, one.RAD_sims, one.Wealth_Paths_ETO, ...
    one.Wealth_Paths_IEO, one.max_budget_eto, one.max_budget_ieo];
one.valid = all(isfinite(core_values)) && ...
    min([one.Regret_ETO_sims, one.Regret_IEO_sims, ...
    one.TrueRegret_ETO_sims, one.TrueRegret_IEO_sims]) >= ...
    -config.negative_regret_tolerance && ...
    max(one.max_budget_eto, one.max_budget_ieo) <= config.budget_tolerance;
end

function [regret_eto, regret_ieo] = main_style_regret( ...
    strategy_eto, strategy_ieo, R_B, config)
[n, ~, T] = size(R_B);
mu_B = zeros(T, n);
Sigma_B = zeros(T, n, n);
for t = 1:T
    mu_B(t, :) = mean(R_B(:, :, t), 2)';
    Sigma_B(t, :, :) = cov(R_B(:, :, t)');
end
[oracle, debug] = calculate_oracle(mu_B, Sigma_B, config.lambda);
regret_eto = zeros(1, T);
regret_ieo = zeros(1, T);
for t = 1:T
    H = 2 * debug.Sigma_t1(:, :, t) + ...
        config.evaluation_ridge * eye(n);
    u_star = oracle.A_coef(:, t) * config.x_eval + oracle.B_coef(:, t);
    u_eto = strategy_eto.A_coef(:, t) * config.x_eval + ...
        strategy_eto.B_coef(:, t);
    u_ieo = strategy_ieo.A_coef(:, t) * config.x_eval + ...
        strategy_ieo.B_coef(:, t);
    delta_eto = u_eto - u_star;
    delta_ieo = u_ieo - u_star;
    regret_eto(t) = 0.5 * delta_eto' * H * delta_eto;
    regret_ieo(t) = 0.5 * delta_ieo' * H * delta_ieo;
end
end

function [regret_eto, regret_ieo] = true_population_regret( ...
    strategy_eto, strategy_ieo, population, config)
T = config.T;
regret_eto = zeros(1, T);
regret_ieo = zeros(1, T);
for t = 1:T
    u_star = population.oracle.A_coef(:, t) * config.x_eval + ...
        population.oracle.B_coef(:, t);
    u_eto = strategy_eto.A_coef(:, t) * config.x_eval + ...
        strategy_eto.B_coef(:, t);
    u_ieo = strategy_ieo.A_coef(:, t) * config.x_eval + ...
        strategy_ieo.B_coef(:, t);
    M = population.oracle_debug.Sigma_t1(:, :, t);
    delta_eto = u_eto - u_star;
    delta_ieo = u_ieo - u_star;
    regret_eto(t) = delta_eto' * M * delta_eto;
    regret_ieo(t) = delta_ieo' * M * delta_ieo;
end
end

function [angle, is_degenerate] = vector_angle(first, second, config)
first_norm = norm(first, 2);
second_norm = norm(second, 2);
is_degenerate = first_norm < config.norm_floor || ...
    second_norm < config.norm_floor;
if is_degenerate
    angle = NaN;
    return;
end
cosine = (first' * second) / (first_norm * second_norm);
cosine = min(1 - config.acos_clip_eps, ...
    max(-1 + config.acos_clip_eps, cosine));
angle = acos(cosine);
end

function arrays = store_replication(arrays, replication_id, one)
stage_fields = {'Regret_ETO_sims', 'Regret_IEO_sims', ...
    'TrueRegret_ETO_sims', 'TrueRegret_IEO_sims', ...
    'SCDR_sims', 'RAD_sims', 'B_Angle_sims', 'HSSG_sims'};
for index = 1:numel(stage_fields)
    field = stage_fields{index};
    arrays.(field)(replication_id, :) = one.(field);
end
arrays.Wealth_Paths_ETO(replication_id, :) = one.Wealth_Paths_ETO;
arrays.Wealth_Paths_IEO(replication_id, :) = one.Wealth_Paths_IEO;
arrays.max_budget_eto(replication_id) = one.max_budget_eto;
arrays.max_budget_ieo(replication_id) = one.max_budget_ieo;
arrays.degenerate_intercept_angles(replication_id) = ...
    one.degenerate_intercept_angles;
arrays.degenerate_slope_angles(replication_id) = one.degenerate_slope_angles;
arrays.valid(replication_id) = one.valid;
arrays.success_eto(replication_id) = one.success_eto;
arrays.success_ieo(replication_id) = one.success_ieo;
arrays.error_eto(replication_id) = one.error_eto;
arrays.error_ieo(replication_id) = one.error_ieo;
end

function [stagewise, wealth, summary] = aggregate_results( ...
    arrays, population, config)
valid = arrays.valid;
assert(any(valid), 'No valid replications are available for aggregation.');

regret_eto = arrays.Regret_ETO_sims(valid, :);
regret_ieo = arrays.Regret_IEO_sims(valid, :);
true_eto = arrays.TrueRegret_ETO_sims(valid, :);
true_ieo = arrays.TrueRegret_IEO_sims(valid, :);
intercept_gap = arrays.SCDR_sims(valid, :);
slope_gap = arrays.RAD_sims(valid, :);
intercept_angle = arrays.B_Angle_sims(valid, :);
slope_angle = arrays.HSSG_sims(valid, :);

regret_eto_stats = pointwise_stats(regret_eto);
regret_ieo_stats = pointwise_stats(regret_ieo);
regret_diff_stats = pointwise_stats(regret_eto - regret_ieo);
true_eto_stats = pointwise_stats(true_eto);
true_ieo_stats = pointwise_stats(true_ieo);
true_diff_stats = pointwise_stats(true_eto - true_ieo);
intercept_gap_stats = pointwise_stats(intercept_gap);
slope_gap_stats = pointwise_stats(slope_gap);
intercept_angle_stats = pointwise_stats(intercept_angle);
slope_angle_stats = pointwise_stats(slope_angle);

stage = (0:config.T-1)';
stagewise = table(stage, ...
    regret_eto_stats.mean', regret_eto_stats.lower', regret_eto_stats.upper', ...
    regret_ieo_stats.mean', regret_ieo_stats.lower', regret_ieo_stats.upper', ...
    regret_diff_stats.mean', regret_diff_stats.lower', regret_diff_stats.upper', ...
    true_eto_stats.mean', true_ieo_stats.mean', true_diff_stats.mean', ...
    intercept_gap_stats.mean', intercept_gap_stats.lower', intercept_gap_stats.upper', ...
    slope_gap_stats.mean', slope_gap_stats.lower', slope_gap_stats.upper', ...
    intercept_angle_stats.mean', intercept_angle_stats.lower', intercept_angle_stats.upper', ...
    slope_angle_stats.mean', slope_angle_stats.lower', slope_angle_stats.upper', ...
    'VariableNames', {'stage', ...
    'regret_eto_mean', 'regret_eto_ci_lower', 'regret_eto_ci_upper', ...
    'regret_ieo_mean', 'regret_ieo_ci_lower', 'regret_ieo_ci_upper', ...
    'regret_eto_minus_ieo_mean', 'regret_difference_ci_lower', ...
    'regret_difference_ci_upper', 'true_regret_eto_mean', ...
    'true_regret_ieo_mean', 'true_regret_eto_minus_ieo_mean', ...
    'intercept_gap_mean', 'intercept_gap_ci_lower', 'intercept_gap_ci_upper', ...
    'slope_gap_mean', 'slope_gap_ci_lower', 'slope_gap_ci_upper', ...
    'intercept_angle_mean', 'intercept_angle_ci_lower', ...
    'intercept_angle_ci_upper', 'slope_angle_mean', ...
    'slope_angle_ci_lower', 'slope_angle_ci_upper'});

wealth_eto = arrays.Wealth_Paths_ETO(valid, :);
wealth_ieo = arrays.Wealth_Paths_IEO(valid, :);
wealth_difference = wealth_eto - wealth_ieo;
wealth_eto_stats = pointwise_stats(wealth_eto);
wealth_ieo_stats = pointwise_stats(wealth_ieo);
wealth_diff_stats = pointwise_stats(wealth_difference);
wealth_stage = (0:config.T)';
wealth = table(wealth_stage, ...
    wealth_eto_stats.mean', wealth_eto_stats.lower', wealth_eto_stats.upper', ...
    wealth_ieo_stats.mean', wealth_ieo_stats.lower', wealth_ieo_stats.upper', ...
    wealth_diff_stats.mean', wealth_diff_stats.lower', wealth_diff_stats.upper', ...
    'VariableNames', {'stage', 'wealth_eto_mean', 'wealth_eto_ci_lower', ...
    'wealth_eto_ci_upper', 'wealth_ieo_mean', 'wealth_ieo_ci_lower', ...
    'wealth_ieo_ci_upper', 'wealth_eto_minus_ieo_mean', ...
    'wealth_difference_ci_lower', 'wealth_difference_ci_upper'});

stage0_difference = regret_eto(:, 1) - regret_ieo(:, 1);
horizon_difference = mean(regret_eto, 2) - mean(regret_ieo, 2);
terminal_wealth_difference = wealth_eto(:, end) - wealth_ieo(:, end);
stage0_ci = bootstrap_mean_ci(stage0_difference, config.bootstrap_resamples, ...
    config.bootstrap_seed + 1);
horizon_ci = bootstrap_mean_ci(horizon_difference, config.bootstrap_resamples, ...
    config.bootstrap_seed + 2);
terminal_wealth_ci = bootstrap_mean_ci(terminal_wealth_difference, ...
    config.bootstrap_resamples, config.bootstrap_seed + 3);

summary = table(string(config.experiment_name), string(config.mode), ...
    config.T, config.n, config.N_A, config.N_B, config.NumSims, sum(valid), ...
    population.condition_number, ...
    mean(regret_eto(:, 1)), mean(regret_ieo(:, 1)), ...
    mean(stage0_difference), stage0_ci(1), stage0_ci(2), ...
    median(stage0_difference), mean(stage0_difference < 0), ...
    mean(horizon_difference), horizon_ci(1), horizon_ci(2), ...
    mean(wealth_eto(:, end)), mean(wealth_ieo(:, end)), ...
    mean(terminal_wealth_difference), terminal_wealth_ci(1), ...
    terminal_wealth_ci(2), median(terminal_wealth_difference), ...
    mean(terminal_wealth_difference > 0), ...
    'VariableNames', {'experiment', 'mode', 'T', 'n', 'N_A', 'N_B', ...
    'attempted_replications', 'valid_replications', ...
    'population_condition_number', 'stage0_regret_eto_mean', ...
    'stage0_regret_ieo_mean', 'stage0_regret_eto_minus_ieo_mean', ...
    'stage0_difference_ci_lower', 'stage0_difference_ci_upper', ...
    'stage0_difference_median', 'eto_stage0_win_rate', ...
    'horizon_regret_eto_minus_ieo_mean', 'horizon_difference_ci_lower', ...
    'horizon_difference_ci_upper', 'terminal_wealth_eto_mean', ...
    'terminal_wealth_ieo_mean', 'terminal_wealth_eto_minus_ieo_mean', ...
    'terminal_wealth_difference_ci_lower', ...
    'terminal_wealth_difference_ci_upper', ...
    'terminal_wealth_difference_median', 'eto_terminal_wealth_win_rate'});
end

function stats = pointwise_stats(values)
stats.mean = mean(values, 1, 'omitnan');
stats.std = std(values, 0, 1, 'omitnan');
stats.count = sum(isfinite(values), 1);
stats.se = stats.std ./ sqrt(max(stats.count, 1));
half_width = 1.96 * stats.se;
stats.lower = stats.mean - half_width;
stats.upper = stats.mean + half_width;
end

function interval = bootstrap_mean_ci(values, resamples, seed)
values = values(isfinite(values));
if isempty(values)
    interval = [NaN, NaN];
    return;
end
saved_rng = rng;
rng_cleanup = onCleanup(@()rng(saved_rng));
rng(seed, 'twister');
N = numel(values);
boot_means = nan(resamples, 1);
for first = 1:250:resamples
    ids = first:min(first + 249, resamples);
    sample_indices = randi(N, N, numel(ids));
    boot_means(ids) = mean(values(sample_indices), 1)';
end
interval = quantile(boot_means, [0.025, 0.975]);
clear rng_cleanup
end

function validation = build_validation(arrays, completed, population, ...
    config, summary)
checks = strings(0, 1);
values = zeros(0, 1);
tolerances = zeros(0, 1);
passed = false(0, 1);
    function add(name, value, tolerance, pass)
        checks(end+1, 1) = string(name);
        values(end+1, 1) = value;
        tolerances(end+1, 1) = tolerance;
        passed(end+1, 1) = pass;
    end

add('all_replications_completed', sum(completed), config.NumSims, all(completed));
add('all_paired_replications_valid', sum(arrays.valid), config.NumSims, ...
    all(arrays.valid));
condition_error = abs(population.condition_number - ...
    config.population_condition_target);
add('population_condition_number_error', condition_error, ...
    config.condition_tolerance, condition_error <= config.condition_tolerance);
add('minimum_population_eigenvalue', population.lambda_min, 0, ...
    population.lambda_min > 0);
active_l2_error = abs(population.active_l2_stage0 - ...
    config.population_active_l2_target);
add('population_active_l2_error', active_l2_error, 1e-8, ...
    active_l2_error <= 1e-8);
delta_error = abs(population.calibrated_delta - ...
    config.expected_calibrated_delta);
add('population_signal_delta_error', delta_error, ...
    config.delta_tolerance, delta_error <= config.delta_tolerance);
maximum_budget = max([arrays.max_budget_eto; arrays.max_budget_ieo], ...
    [], 'omitnan');
add('maximum_budget_violation', maximum_budget, config.budget_tolerance, ...
    maximum_budget <= config.budget_tolerance);
initial_wealth = [arrays.Wealth_Paths_ETO(:, 1); ...
    arrays.Wealth_Paths_IEO(:, 1)];
initial_error = max(abs(initial_wealth - config.x0), [], 'omitnan');
add('initial_wealth_max_error', initial_error, 1e-14, initial_error <= 1e-14);
degenerate_intercept_count = sum(arrays.degenerate_intercept_angles);
add('degenerate_intercept_angle_count', ...
    degenerate_intercept_count, 0, degenerate_intercept_count == 0);
degenerate_slope_count = sum(arrays.degenerate_slope_angles);
add('degenerate_slope_angle_count', degenerate_slope_count, 0, ...
    degenerate_slope_count == 0);
stage0_difference = summary.stage0_regret_eto_minus_ieo_mean;
add('eto_lower_stage0_regret_direction', stage0_difference, 0, ...
    stage0_difference < 0);
valid = arrays.valid;
mean_regret_eto = mean(arrays.Regret_ETO_sims(valid, :), 1);
mean_regret_ieo = mean(arrays.Regret_IEO_sims(valid, :), 1);
eto_lower_stage_count = sum(mean_regret_eto < mean_regret_ieo);
add('eto_lower_regret_stage_count', eto_lower_stage_count, config.T, ...
    eto_lower_stage_count == config.T);
eto_endpoint_ratio = mean_regret_eto(end) / mean_regret_eto(1);
ieo_endpoint_ratio = mean_regret_ieo(end) / mean_regret_ieo(1);
add('eto_regret_endpoint_ratio', eto_endpoint_ratio, 1, ...
    eto_endpoint_ratio < 1);
add('ieo_regret_endpoint_ratio', ieo_endpoint_ratio, 1, ...
    ieo_endpoint_ratio < 1);
eto_adjacent_decrease_fraction = mean(diff(mean_regret_eto) < 0);
ieo_adjacent_decrease_fraction = mean(diff(mean_regret_ieo) < 0);
add('eto_regret_adjacent_decrease_fraction', ...
    eto_adjacent_decrease_fraction, 0.5, ...
    eto_adjacent_decrease_fraction > 0.5);
add('ieo_regret_adjacent_decrease_fraction', ...
    ieo_adjacent_decrease_fraction, 0.5, ...
    ieo_adjacent_decrease_fraction > 0.5);
stage_index = 0:(config.T - 1);
eto_linear_fit = polyfit(stage_index, mean_regret_eto, 1);
ieo_linear_fit = polyfit(stage_index, mean_regret_ieo, 1);
add('eto_regret_linear_slope', eto_linear_fit(1), 0, ...
    eto_linear_fit(1) < 0);
add('ieo_regret_linear_slope', ieo_linear_fit(1), 0, ...
    ieo_linear_fit(1) < 0);
add('factor_estimators_on_execution_path', 0, 0, true);
validation = table(checks, values, tolerances, passed, ...
    'VariableNames', {'check', 'value', 'tolerance', 'passed'});
end

function make_main_figures(stagewise, wealth, config, arrays)
visibility = 'off';
if config.show_figures
    visibility = 'on';
end
blue = [0, 0.4470, 0.7410];
red = [0.8500, 0.3250, 0.0980];
gold = [0.9290, 0.6940, 0.1250];
purple = [0.4940, 0.1840, 0.5560];

fig = figure('Color', 'w', 'Visible', visibility);
hold on;
% The paper regret panel displays true population regret, not the
% separately retained Universe-B hindsight/sample proxy.
plot(stagewise.stage, stagewise.true_regret_eto_mean, 'b--', 'LineWidth', 2);
plot(stagewise.stage, stagewise.true_regret_ieo_mean, 'r-', 'LineWidth', 2);
grid on; box on; xlabel('stage t'); ylabel('population regret');
legend('ETO', 'IEO', 'Location', 'NorthEast');
xlim([0, config.T]);
save_garch_style_figure(fig, config.figure_dir, ...
    'Experiment_3_Regular_Gaussian_Regret');

fig = figure('Color', 'w', 'Visible', visibility);
ax = axes('Parent', fig); hold(ax, 'on');
wealth_band = wealth_path_band(arrays, config);
fill_ci(ax, wealth.stage, wealth_band.eto_lower, wealth_band.eto_upper, ...
    'b', 0.2);
fill_ci(ax, wealth.stage, wealth_band.ieo_lower, wealth_band.ieo_upper, ...
    'r', 0.2);
plot(ax, wealth.stage, wealth.wealth_eto_mean, 'b--', 'LineWidth', 2);
plot(ax, wealth.stage, wealth.wealth_ieo_mean, 'r-', 'LineWidth', 2);
grid(ax, 'on'); box(ax, 'on'); xlabel(ax, 'stage t');
ylabel(ax, 'out-of-sample wealth');
legend(ax, 'ETO', 'IEO', 'Location', 'NorthEast');
xlim(ax, [0, config.T]);
save_garch_style_figure(fig, config.figure_dir, ...
    'Experiment_3_Regular_Gaussian_Wealth');

plot_single_metric(stagewise.stage, stagewise.intercept_gap_mean, ...
    stagewise.intercept_gap_ci_lower, stagewise.intercept_gap_ci_upper, ...
    blue, 'intercept gap', 'Experiment_3_Regular_Gaussian_Intercept_Gap', ...
    config, visibility);
plot_single_metric(stagewise.stage, stagewise.slope_gap_mean, ...
    stagewise.slope_gap_ci_lower, stagewise.slope_gap_ci_upper, ...
    red, 'slope gap', 'Experiment_3_Regular_Gaussian_Slope_Gap', ...
    config, visibility);
plot_single_metric(stagewise.stage, stagewise.intercept_angle_mean, ...
    stagewise.intercept_angle_ci_lower, stagewise.intercept_angle_ci_upper, ...
    gold, 'intercept angle (rad)', ...
    'Experiment_3_Regular_Gaussian_Intercept_Angle', config, visibility);
plot_single_metric(stagewise.stage, stagewise.slope_angle_mean, ...
    stagewise.slope_angle_ci_lower, stagewise.slope_angle_ci_upper, ...
    purple, 'slope angle (rad)', ...
    'Experiment_3_Regular_Gaussian_Slope_Angle', config, visibility);
end

function band = wealth_path_band(arrays, config)
valid = arrays.valid;
wealth_eto = arrays.Wealth_Paths_ETO(valid, :);
wealth_ieo = arrays.Wealth_Paths_IEO(valid, :);
band.stage = (0:config.T)';
band.eto_lower = quantile(wealth_eto, 0.025, 1)';
band.eto_upper = quantile(wealth_eto, 0.975, 1)';
band.ieo_lower = quantile(wealth_ieo, 0.025, 1)';
band.ieo_upper = quantile(wealth_ieo, 0.975, 1)';
end

function plot_single_metric(stage, center, lower, upper, color, y_label, ...
    file_stem, config, visibility)
fig = figure('Color', 'w', 'Visible', visibility);
ax = axes(fig); hold(ax, 'on');
fill_ci(ax, stage, lower, upper, color);
plot(ax, stage, center, '-', 'Color', color, 'LineWidth', 2);
grid(ax, 'on'); box(ax, 'on'); xlabel(ax, 'stage t'); ylabel(ax, y_label);
save_figure(fig, config.figure_dir, file_stem);
end

function fill_ci(ax, x, lower, upper, color, face_alpha)
if nargin < 6
    face_alpha = 0.18;
end
fill(ax, [x; flipud(x)], [upper; flipud(lower)], color, ...
    'FaceAlpha', face_alpha, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function save_figure(fig, directory, stem)
exportgraphics(fig, fullfile(directory, [stem, '.png']), 'Resolution', 300);
exportgraphics(fig, fullfile(directory, [stem, '.pdf']), 'ContentType', 'vector');
savefig(fig, fullfile(directory, [stem, '.fig']));
close(fig);
end

function save_garch_style_figure(fig, directory, stem)
saveas(fig, fullfile(directory, [stem, '.png']));
saveas(fig, fullfile(directory, [stem, '.fig']));
exportgraphics(fig, fullfile(directory, [stem, '.pdf']), 'ContentType', 'vector');
close(fig);
end

function assert_checkpoint_compatible(saved, config, population, design)
assert(saved.config.schema_version == config.schema_version);
assert(strcmp(saved.config.mode, config.mode));
assert(saved.config.NumSims == config.NumSims);
assert(saved.config.replication_seed_offset == config.replication_seed_offset);
assert(isequal(saved.population.mu, population.mu));
assert(isequal(saved.population.Sigma, population.Sigma));
assert(strcmp(saved.design.name, design.name));
end

function atomic_checkpoint(target, arrays, completed, config, population, design)
temporary = [tempname(fileparts(target)), '.mat'];
save(temporary, 'arrays', 'completed', 'config', 'population', 'design', '-v7.3');
movefile(temporary, target, 'f');
end

function ensure_directory(directory)
if ~exist(directory, 'dir')
    mkdir(directory);
end
end
