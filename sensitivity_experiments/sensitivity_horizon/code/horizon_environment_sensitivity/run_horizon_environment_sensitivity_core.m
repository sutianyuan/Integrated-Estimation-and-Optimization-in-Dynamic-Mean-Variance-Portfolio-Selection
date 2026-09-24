function outputs = run_horizon_environment_sensitivity_core(mode, varargin)
%RUN_HORIZON_ENVIRONMENT_SENSITIVITY_CORE Reproduce the two horizon panels.
%   DRY performs a small real simulation for both published environments.
%   FULL performs the formal S=2000 simulation for both environments.
%   PLOT_ONLY redraws the saved summary and does not run simulations.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
mode = validatestring(lower(char(mode)), {'dry','full','plot_only'});
config = locked_config(mode, varargin{:});

if strcmp(mode, 'plot_only')
    validate_packaged_summary(config.saved_summary_file, config);
    figure_files = plot_horizon_regret_full( ...
        'SummaryFile', config.saved_summary_file, ...
        'OutputDir', config.figures_dir, ...
        'StemName', 'horizon_mainDGP_aligned_deltaR_plot_only');
    outputs = struct('mode', mode, 'simulation_executed', false, ...
        'summary', config.saved_summary_file, 'figures', {figure_files});
    fprintf('PLOT_ONLY completed from packaged summary; no simulation was run.\n');
    return
end

ensure_directory(config.results_dir);
ensure_directory(config.figures_dir);
ensure_directory(config.checkpoint_dir);

original_path = path;
path_cleanup = onCleanup(@() path(original_path));
addpath(config.dependency_dir, '-begin');
assert_method_path(config);

[regular_mu, regular_Sigma] = build_regular_main_exp1_dgp(config);
[ill_mu, ill_Sigma] = build_ill_conditioned_dgp(config);
regular_population = build_population(regular_mu, regular_Sigma, config);
ill_population = build_population(ill_mu, ill_Sigma, config);
assert_dgp_fingerprints(regular_population, ill_population, config);

regular = run_horizon_panel_regular(regular_population, config);
ill = run_horizon_panel_ill_conditioned(ill_population, config);
assert(~isempty(regular.summary) && height(regular.summary) == numel(config.T_grid), ...
    'HorizonSensitivity:EmptyRegularPanel', 'Regular horizon panel is empty or incomplete.');
assert(~isempty(ill.summary) && height(ill.summary) == numel(config.T_grid), ...
    'HorizonSensitivity:EmptyIllPanel', 'Ill-conditioned horizon panel is empty or incomplete.');

summary = merge_panels(regular.summary, ill.summary, config);
validation = validate_run(regular, ill, summary, config);
assert(all(validation.passed), 'HorizonSensitivity:ValidationFailed', ...
    'Horizon sensitivity failed an internal consistency check.');

outputs = output_paths(config);
writetable(summary, outputs.summary);
cleanup_runtime_checkpoints(config);

if config.make_plots
    outputs.figures = plot_horizon_regret_full( ...
        'SummaryFile', outputs.summary, 'OutputDir', config.figures_dir, ...
        'StemName', sprintf('horizon_%s_deltaR', config.mode));
else
    outputs.figures = {};
end

outputs.mode = config.mode;
outputs.simulation_executed = true;
outputs.regular_panel_rows = height(regular.summary);
outputs.ill_conditioned_panel_rows = height(ill.summary);
outputs.merged_rows = height(summary);
outputs.full_entry_ready = true;

fprintf('Horizon sensitivity %s completed: regular=%d, ill-conditioned=%d, merged=%d rows.\n', ...
    config.mode, outputs.regular_panel_rows, outputs.ill_conditioned_panel_rows, outputs.merged_rows);
clear path_cleanup
end

function config = locked_config(mode, varargin)
script_dir = fileparts(mfilename('fullpath'));
module_dir = fileparts(fileparts(script_dir));
config = struct();
config.mode = mode;
config.module_dir = module_dir;
config.dependency_dir = fullfile(module_dir, 'code', 'dependencies');
config.saved_summary_file = fullfile(module_dir, 'tables', 'results', ...
    'horizon_mainDGP_aligned_full.csv');
config.n = 10;
config.x0 = 1;
config.Ntr = 60;
config.Nev = 1000;
config.lambda_scalar = 1e-4;
config.T_grid = 10:5:200;
config.base_seed = 42;
config.replication_seed_offset = 620000;
config.bootstrap_seed = 271828;
config.regular_marginal_variance = 8.103335969935516e-5;
config.regular_mu_bar = 1.0002370503416058;
config.regular_condition_target = 20;
config.regular_active_l2_target = 0.05;
config.regular_expected_delta = 0.00065978679633078;
config.ill_target20 = 945859393.549689;
config.ill_mu = [1.00010418542893; 1.00016912992543; 1.00043456604317; ...
    1.00010874203588; 0.999944120596905; 1.00045789856738; ...
    1.00045329976482; 0.999864641813033; 1.00027037817328; ...
    1.00056354106723];
config.negative_regret_tolerance = 1e-10;
config.budget_tolerance = 1e-8;
config.full_NumSims = 2000;
config.dry_NumSims = 3;
config.full_bootstrap_resamples = 5000;
config.dry_bootstrap_resamples = 500;

if strcmp(mode, 'full')
    config.NumSims = config.full_NumSims;
    config.bootstrap_resamples = config.full_bootstrap_resamples;
    config.use_parallel = true;
    config.parallel_workers = 4;
    config.batch_size = 10;
else
    config.NumSims = config.dry_NumSims;
    config.bootstrap_resamples = config.dry_bootstrap_resamples;
    config.use_parallel = false;
    config.parallel_workers = 0;
    config.batch_size = 1;
end

p = inputParser;
addParameter(p, 'OutputRoot', module_dir);
addParameter(p, 'UseParallel', config.use_parallel);
addParameter(p, 'ParallelWorkers', config.parallel_workers);
addParameter(p, 'BatchSize', config.batch_size);
addParameter(p, 'NumSims', config.NumSims);
addParameter(p, 'BootstrapResamples', config.bootstrap_resamples);
addParameter(p, 'MakePlots', true);
addParameter(p, 'CheckpointDir', '');
parse(p, varargin{:});

config.output_root = char(p.Results.OutputRoot);
config.use_parallel = logical(p.Results.UseParallel);
config.parallel_workers = p.Results.ParallelWorkers;
config.batch_size = p.Results.BatchSize;
config.NumSims = p.Results.NumSims;
config.bootstrap_resamples = p.Results.BootstrapResamples;
config.make_plots = logical(p.Results.MakePlots);
config.checkpoint_dir = char(p.Results.CheckpointDir);

if strcmp(mode, 'full')
    assert(config.NumSims == config.full_NumSims, ...
        'HorizonSensitivity:LockedFullReplications', 'FULL mode requires S=2000.');
    assert(config.bootstrap_resamples == config.full_bootstrap_resamples, ...
        'HorizonSensitivity:LockedFullBootstrap', 'FULL mode requires 5000 bootstrap resamples.');
end
validateattributes(config.NumSims, {'numeric'}, {'scalar','integer','positive'});
validateattributes(config.bootstrap_resamples, {'numeric'}, {'scalar','integer','positive'});
validateattributes(config.batch_size, {'numeric'}, {'scalar','integer','positive'});
assert(isequal(config.T_grid, 10:5:200) && numel(config.T_grid) == 39);
assert(config.Ntr == 60 && config.Nev == 1000 && config.lambda_scalar == 1e-4);

mode_dir = strrep(mode, '_', '-');
config.results_dir = fullfile(config.output_root, 'results', mode_dir);
config.figures_dir = fullfile(config.output_root, 'figures', mode_dir);
if isempty(config.checkpoint_dir) && ~strcmp(mode, 'plot_only')
    config.checkpoint_dir = fullfile(tempdir, ...
        sprintf('ieo_horizon_sensitivity_%s_checkpoint', mode_dir));
end
config.lambda = config.lambda_scalar * ones(1, max(config.T_grid));
end

function assert_method_path(config)
assert(isfolder(config.dependency_dir), 'HorizonSensitivity:MissingDependencies', ...
    'Missing horizon dependency directory: %s', config.dependency_dir);
names = {'run_eto.m','run_ieo.m','calculate_oracle.m'};
resolved_names = {'run_eto','run_ieo','calculate_oracle'};
for k = 1:numel(names)
    expected = fullfile(config.dependency_dir, names{k});
    actual = which(resolved_names{k});
    assert(isfile(expected) && strcmp(actual, expected), ...
        'HorizonSensitivity:WrongMethodPath', ...
        'Expected %s but MATLAB resolves %s.', expected, actual);
end
end

function [mu, Sigma] = build_regular_main_exp1_dgp(config)
n = config.n;
covariance_stream = RandStream('mt19937ar', 'Seed', 913701);
[Q, ~] = qr(randn(covariance_stream, n, n), 0);
eigenvalue_shape = logspace(log10(config.regular_condition_target), 0, n)';
eigenvalue_scale = config.regular_marginal_variance * n / sum(eigenvalue_shape);
Sigma = Q * diag(eigenvalue_scale * eigenvalue_shape) * Q';
Sigma = (Sigma + Sigma') / 2;

signal_stream = RandStream('mt19937ar', 'Seed', 913702);
signal_direction = randn(signal_stream, n, 1);
signal_direction = signal_direction - mean(signal_direction);
signal_direction = signal_direction / norm(signal_direction, 2);
local = struct('n', n, 'T', 50, ...
    'lambda', config.lambda_scalar * ones(1, 50), ...
    'mu_bar', config.regular_mu_bar, ...
    'population_active_l2_target', config.regular_active_l2_target);
zero_mean = config.regular_mu_bar * ones(n, 1);
zero_oracle = make_oracle_record(Sigma, zero_mean, local);
zero_policy = zero_oracle.strategy.A_coef(:, 1) + zero_oracle.strategy.B_coef(:, 1);
delta = calibrate_population_signal(Sigma, signal_direction, zero_policy, local);
assert(abs(delta - config.regular_expected_delta) < 1e-14, ...
    'HorizonSensitivity:RegularDGPFingerprint', 'Regular signal calibration changed.');
mu = zero_mean + delta * signal_direction;
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
assert(objective(upper) >= 0, 'Could not bracket regular population signal.');
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
candidate = make_oracle_record(Sigma, mu, config);
policy = candidate.strategy.A_coef(:, 1) + candidate.strategy.B_coef(:, 1);
distance = norm(policy - zero_policy, 2);
end

function [mu, Sigma] = build_ill_conditioned_dgp(config)
ratio = (config.ill_target20 - 1) / 20;
d0 = config.regular_marginal_variance / (1 + ratio);
s2 = ratio * d0;
Sigma = s2 * ones(config.n) + d0 * eye(config.n);
mu = config.ill_mu;
end

function population = build_population(mu, Sigma, config)
Sigma = (Sigma + Sigma') / 2;
L = chol(Sigma, 'lower');
eigenvalues = eig(Sigma, 'vector');
population = repmat(struct(), 1, numel(config.T_grid));
for h = 1:numel(config.T_grid)
    T = config.T_grid(h);
    local = struct('n', config.n, 'T', T, ...
        'lambda', config.lambda_scalar * ones(1, T));
    oracle = make_oracle_record(Sigma, mu, local);
    population(h).mu = mu;
    population(h).Sigma = Sigma;
    population(h).cholesky = L;
    population(h).mu_path = oracle.mu_path;
    population(h).Sigma_path = oracle.Sigma_path;
    population(h).oracle = oracle.strategy;
    population(h).oracle_debug = oracle.debug;
    population(h).condition_number = cond(Sigma);
    population(h).lambda_min = min(eigenvalues);
    population(h).lambda_max = max(eigenvalues);
end
end

function record = make_oracle_record(Sigma, mu, config)
mu_path = repmat(mu(:)', config.T, 1);
Sigma_path = repmat(reshape(Sigma, 1, config.n, config.n), config.T, 1, 1);
[strategy, debug] = calculate_oracle(mu_path, Sigma_path, config.lambda);
record = struct('strategy', strategy, 'debug', debug, ...
    'mu_path', mu_path, 'Sigma_path', Sigma_path);
end

function assert_dgp_fingerprints(regular_population, ill_population, config)
regular_condition = regular_population(1).condition_number;
ill_condition = ill_population(1).condition_number;
assert(abs(regular_condition - 20) < 1e-10, ...
    'HorizonSensitivity:RegularConditionChanged', 'Regular condition number changed.');
assert(abs(mean(diag(regular_population(1).Sigma)) - config.regular_marginal_variance) < 1e-16);
assert(abs(ill_condition - 472929715.157553) / 472929715.157553 < 1e-10, ...
    'HorizonSensitivity:IllConditionChanged', 'Ill-conditioned DGP changed.');
assert(regular_population(1).lambda_min > 0 && ill_population(1).lambda_min > 0);
end

function panel = run_horizon_panel_regular(population, config)
panel = run_horizon_panel('regular', 'Regular stationary Gaussian', ...
    1, population, config);
end

function panel = run_horizon_panel_ill_conditioned(population, config)
panel = run_horizon_panel('ill_conditioned', ...
    'Severely ill-conditioned stationary Gaussian', 2, population, config);
end

function panel = run_horizon_panel(panel_name, panel_label, panel_index, population, config)
H = numel(config.T_grid);
S = config.NumSims;
R0_ETO = nan(H, S);
R0_IEO = nan(H, S);
valid = false(H, S);
min_regret = nan(H, S);
budget = nan(H, S);
err_eto = strings(H, S);
err_ieo = strings(H, S);
checkpoint = fullfile(config.checkpoint_dir, sprintf('%s_checkpoint.mat', panel_name));
metadata = checkpoint_metadata(panel_name, population, config);

if isfile(checkpoint)
    saved = load(checkpoint, 'R0_ETO','R0_IEO','valid','min_regret', ...
        'budget','err_eto','err_ieo','metadata');
    assert(isfield(saved, 'metadata') && isequal(saved.metadata, metadata), ...
        'HorizonSensitivity:IncompatibleCheckpoint', ...
        'Checkpoint is incompatible with the locked horizon configuration: %s', checkpoint);
    R0_ETO = saved.R0_ETO;
    R0_IEO = saved.R0_IEO;
    valid = saved.valid;
    min_regret = saved.min_regret;
    budget = saved.budget;
    err_eto = saved.err_eto;
    err_ieo = saved.err_ieo;
end

remaining = find(~all(valid, 1));
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
        parfor b = 1:numel(ids)
            blocks{b} = one_panel_replication(ids(b), population, config);
        end
    else
        for b = 1:numel(ids)
            blocks{b} = one_panel_replication(ids(b), population, config);
        end
    end
    for b = 1:numel(ids)
        s = ids(b);
        one = blocks{b};
        R0_ETO(:, s) = one.R0_ETO;
        R0_IEO(:, s) = one.R0_IEO;
        valid(:, s) = one.valid;
        min_regret(:, s) = one.min_regret;
        budget(:, s) = one.budget;
        err_eto(:, s) = one.err_eto;
        err_ieo(:, s) = one.err_ieo;
    end
    atomic_checkpoint(checkpoint, R0_ETO, R0_IEO, valid, min_regret, ...
        budget, err_eto, err_ieo, metadata);
    fprintf('Horizon %s panel: %d/%d paired replications complete.\n', ...
        panel_name, sum(all(valid, 1)), S);
end

assert(all(valid, 'all'), 'HorizonSensitivity:InvalidPanelReplication', ...
    '%s panel contains invalid horizon/replication cells.', panel_label);
panel = struct('name', panel_name, 'label', panel_label, ...
    'population', population, 'R0_ETO', R0_ETO, 'R0_IEO', R0_IEO, ...
    'valid', valid, 'min_regret', min_regret, 'budget', budget, ...
    'err_eto', err_eto, 'err_ieo', err_ieo);
panel.summary = summarize_panel(panel, panel_index, config);
end

function one = one_panel_replication(id, population, config)
H = numel(config.T_grid);
Tmax = max(config.T_grid);
one.R0_ETO = nan(H, 1);
one.R0_IEO = nan(H, 1);
one.valid = false(H, 1);
one.min_regret = nan(H, 1);
one.budget = nan(H, 1);
one.err_eto = strings(H, 1);
one.err_ieo = strings(H, 1);
seed = config.base_seed + config.replication_seed_offset + id;
sA = RandStream('mt19937ar', 'Seed', seed + 100);
sB = RandStream('mt19937ar', 'Seed', seed + 200);
zA = randn(sA, config.n, config.Ntr, Tmax);
zB = randn(sB, config.n, config.Nev, Tmax);
pmax = population(end);
RAmax = pmax.mu + pagemtimes(pmax.cholesky, zA);
RBmax = pmax.mu + pagemtimes(pmax.cholesky, zB);
for h = 1:H
    T = config.T_grid(h);
    RA = RAmax(:, :, 1:T);
    RB = RBmax(:, :, 1:T);
    eto = [];
    ieo = [];
    try
        evalc('eto = run_eto(RA, RB, config.lambda(1:T), config.x0);');
    catch ex
        one.err_eto(h) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
    end
    try
        evalc('ieo = run_ieo(RA, RB, config.lambda(1:T), config.x0);');
    catch ex
        one.err_ieo(h) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
    end
    if ~isempty(eto) && ~isempty(ieo)
        result = evaluate_population_stage0(eto, ieo, population(h), T);
        one.R0_ETO(h) = result.R0_ETO;
        one.R0_IEO(h) = result.R0_IEO;
        one.min_regret(h) = result.min_regret;
        one.budget(h) = result.max_budget;
        one.valid(h) = all(isfinite([result.R0_ETO, result.R0_IEO])) && ...
            result.min_regret >= -config.negative_regret_tolerance && ...
            result.max_budget <= config.budget_tolerance;
    end
end
end

function result = evaluate_population_stage0(eto, ieo, population, T)
truth = population.oracle.A_coef + population.oracle.B_coef;
eto_policy = eto.strategy.A_coef + eto.strategy.B_coef;
ieo_policy = ieo.strategy.A_coef + ieo.strategy.B_coef;
eto_regret = zeros(1, T);
ieo_regret = zeros(1, T);
for t = 1:T
    M = population.oracle_debug.Sigma_t1(:, :, t);
    eto_difference = eto_policy(:, t) - truth(:, t);
    ieo_difference = ieo_policy(:, t) - truth(:, t);
    eto_regret(t) = eto_difference' * M * eto_difference;
    ieo_regret(t) = ieo_difference' * M * ieo_difference;
end
result = struct('R0_ETO', eto_regret(1), 'R0_IEO', ieo_regret(1), ...
    'min_regret', min([eto_regret, ieo_regret]), ...
    'max_budget', max(max(abs(sum(eto_policy, 1) - 1)), ...
    max(abs(sum(ieo_policy, 1) - 1))));
end

function summary = summarize_panel(panel, panel_index, config)
rows = struct([]);
for h = 1:numel(config.T_grid)
    valid = panel.valid(h, :);
    eto = panel.R0_ETO(h, valid)';
    ieo = panel.R0_IEO(h, valid)';
    difference = eto - ieo;
    ci = paired_bootstrap_ci(difference, config.bootstrap_resamples, ...
        config.bootstrap_seed + 100 * panel_index + h);
    row = struct('T', config.T_grid(h), ...
        'environment', string(panel.label), ...
        'R0_ETO_pop_mean', mean(eto), ...
        'R0_IEO_pop_mean', mean(ieo), ...
        'Delta_R', mean(difference), ...
        'CI_low', ci(1), 'CI_high', ci(2), ...
        'cond_Sigma', panel.population(h).condition_number, ...
        'lambda', config.lambda_scalar, 'Ntr', config.Ntr, ...
        'Nev', config.Nev, 'S', config.NumSims);
    if isempty(rows)
        rows = row;
    else
        rows(end + 1) = row; %#ok<AGROW>
    end
end
summary = struct2table(rows);
end

function ci = paired_bootstrap_ci(difference, B, seed)
difference = difference(isfinite(difference));
assert(~isempty(difference), 'Cannot bootstrap an empty paired-difference vector.');
saved_rng = rng;
rng_cleanup = onCleanup(@() rng(saved_rng));
rng(seed, 'twister');
N = numel(difference);
bootstrap_means = nan(B, 1);
for first = 1:250:B
    ids = first:min(first + 249, B);
    indices = randi(N, N, numel(ids));
    bootstrap_means(ids) = mean(difference(indices), 1)';
end
ci = quantile(bootstrap_means, [0.025, 0.975]);
clear rng_cleanup
end

function summary = merge_panels(regular, ill, config)
assert(~isempty(regular), 'HorizonSensitivity:EmptyRegularPanel', ...
    'Regular horizon panel is empty.');
assert(~isempty(ill), 'HorizonSensitivity:EmptyIllPanel', ...
    'Ill-conditioned horizon panel is empty.');
assert(height(regular) == 39 && height(ill) == 39);
summary = [regular; ill];
summary.environment = categorical(summary.environment, ...
    {'Regular stationary Gaussian', ...
    'Severely ill-conditioned stationary Gaussian'}, 'Ordinal', true);
summary = sortrows(summary, {'environment','T'});
summary.environment = string(summary.environment);
assert(height(summary) == 2 * numel(config.T_grid));
assert(sum(summary.environment == "Regular stationary Gaussian") == 39);
assert(sum(summary.environment == "Severely ill-conditioned stationary Gaussian") == 39);
assert(max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean))) < 1e-14);
end

function validation = validate_run(regular, ill, summary, config)
checks = strings(0, 1);
values = strings(0, 1);
passed = false(0, 1);
add('regular_panel_nonempty', ~isempty(regular.summary), sprintf('%d rows', height(regular.summary)));
add('ill_panel_nonempty', ~isempty(ill.summary), sprintf('%d rows', height(ill.summary)));
add('regular_panel_39_rows', height(regular.summary) == 39, sprintf('%d', height(regular.summary)));
add('ill_panel_39_rows', height(ill.summary) == 39, sprintf('%d', height(ill.summary)));
add('merged_78_rows', height(summary) == 78, sprintf('%d', height(summary)));
add('horizon_grid_locked', isequal(unique(summary.T)', config.T_grid), mat2str(unique(summary.T)'));
add('both_environments_present', numel(unique(summary.environment)) == 2, strjoin(unique(summary.environment), ' | '));
add('all_regular_cells_valid', all(regular.valid, 'all'), sprintf('%d/%d', nnz(regular.valid), numel(regular.valid)));
add('all_ill_cells_valid', all(ill.valid, 'all'), sprintf('%d/%d', nnz(ill.valid), numel(ill.valid)));
add('delta_identity', max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean))) < 1e-14, 'tolerance 1e-14');
add('ci_ordered', all(summary.CI_low <= summary.CI_high), 'CI_low <= CI_high');
add('point_estimate_inside_ci', all(summary.CI_low <= summary.Delta_R & ...
    summary.Delta_R <= summary.CI_high), 'Delta_R inside interval');
add('formal_settings_locked', all(summary.Ntr == 60) && all(summary.Nev == 1000) && ...
    all(summary.lambda == 1e-4), 'Ntr=60 Nev=1000 lambda=1e-4');
add('replication_count_matches_mode', all(summary.S == config.NumSims), sprintf('S=%d', config.NumSims));
add('regular_condition_number', max(abs(regular.summary.cond_Sigma - 20)) < 1e-8, 'cond=20');
add('ill_condition_number', max(abs(ill.summary.cond_Sigma - 472929715.157553)) / ...
    472929715.157553 < 1e-8, sprintf('cond=%.15g', ill.summary.cond_Sigma(1)));
validation = table(checks, values, passed, ...
    'VariableNames', {'check','value','passed'});
    function add(name, condition, value)
        checks(end + 1, 1) = string(name);
        values(end + 1, 1) = string(value);
        passed(end + 1, 1) = logical(condition);
    end
end

function metadata = checkpoint_metadata(panel_name, population, config)
metadata = struct('schema_version', 1, 'panel_name', panel_name, ...
    'T_grid', config.T_grid, 'NumSims', config.NumSims, ...
    'Ntr', config.Ntr, 'Nev', config.Nev, ...
    'lambda_scalar', config.lambda_scalar, ...
    'base_seed', config.base_seed, ...
    'replication_seed_offset', config.replication_seed_offset, ...
    'mu', population(1).mu, 'Sigma', population(1).Sigma);
end

function atomic_checkpoint(target, R0_ETO, R0_IEO, valid, min_regret, ...
    budget, err_eto, err_ieo, metadata)
temporary = [tempname(fileparts(target)), '.mat'];
save(temporary, 'R0_ETO','R0_IEO','valid','min_regret','budget', ...
    'err_eto','err_ieo','metadata','-v7.3');
movefile(temporary, target, 'f');
end

function outputs = output_paths(config)
if strcmp(config.mode, 'full')
    summary_name = 'horizon_mainDGP_aligned_full.csv';
else
    summary_name = 'horizon_dry_summary.csv';
end
outputs = struct('summary', fullfile(config.results_dir, summary_name));
end

function cleanup_runtime_checkpoints(config)
names = {'regular_checkpoint.mat', 'ill_conditioned_checkpoint.mat'};
for k = 1:numel(names)
    file_name = fullfile(config.checkpoint_dir, names{k});
    if isfile(file_name)
        delete(file_name);
    end
end
if isfolder(config.checkpoint_dir)
    remaining = dir(config.checkpoint_dir);
    remaining = remaining(~ismember({remaining.name}, {'.','..'}));
    if isempty(remaining)
        rmdir(config.checkpoint_dir);
    end
end
end

function validate_packaged_summary(summary_file, config)
assert(isfile(summary_file), 'HorizonSensitivity:MissingSavedSummary', ...
    'Packaged horizon summary is missing: %s', summary_file);
summary = readtable(summary_file, 'TextType', 'string');
required = {'T','environment','R0_ETO_pop_mean','R0_IEO_pop_mean', ...
    'Delta_R','CI_low','CI_high','cond_Sigma','lambda','Ntr','Nev','S'};
assert(all(ismember(required, summary.Properties.VariableNames)));
assert(height(summary) == 78 && numel(unique(summary.T)) == 39);
assert(sum(summary.environment == "Regular stationary Gaussian") == 39);
assert(sum(summary.environment == "Severely ill-conditioned stationary Gaussian") == 39);
delta_from_components = summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean;
assert(max(abs(summary.Delta_R - delta_from_components)) <= 5e-8, ...
    'HorizonSensitivity:SavedDeltaMismatch', ...
    'Packaged Delta_R is inconsistent with ETO - IEO beyond CSV rounding tolerance.');
assert(all(summary.CI_low <= summary.Delta_R & summary.Delta_R <= summary.CI_high));
assert(all(summary.S == 2000) && all(summary.Nev == 1000) && ...
    all(summary.Ntr == 60) && all(summary.lambda == config.lambda_scalar));
end

function ensure_directory(directory)
if ~isfolder(directory)
    mkdir(directory);
end
end
