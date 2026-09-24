function outputs = run_training_sample_sensitivity_core(mode, varargin)
%RUN_TRAINING_SAMPLE_SENSITIVITY_CORE Reproduce the two paper training panels.
%   DRY performs a small real simulation over all 11 training sizes and both
%   published environments. FULL performs the formal S=2000 experiment.
%   PLOT_ONLY redraws the saved 22-row summary without running simulation.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
mode = validatestring(lower(char(mode)), {'dry','full','plot_only'});
config = locked_config(mode, varargin{:});

if strcmp(mode, 'plot_only')
    validation = validate_packaged_summary(config.saved_summary_file, config);
    figure_files = plot_training_sample_full( ...
        'SummaryFile', config.saved_summary_file, ...
        'OutputDir', config.output_dir, ...
        'StemName', 'training_mainDGP_aligned_deltaR');
    outputs = struct('mode', mode, 'simulation_executed', false, ...
        'summary', config.saved_summary_file, ...
        'figures', {figure_files}, 'regular_panel_rows', 11, ...
        'ill_conditioned_panel_rows', 11, 'merged_rows', 22, ...
        'full_entry_ready', true);
    fprintf('PLOT_ONLY completed from the packaged 22-row summary; no simulation was run.\n');
    return
end

ensure_directory(config.output_dir);
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

regularResults = run_training_panel_regular(regular_population, config);
illResults = run_training_panel_ill_conditioned(ill_population, config);
assert(~isempty(regularResults.summary), 'TrainingSensitivity:EmptyRegularPanel', ...
    'Regular training panel is empty.');
assert(~isempty(illResults.summary), 'TrainingSensitivity:EmptyIllPanel', ...
    'Ill-conditioned training panel is empty.');
assert(height(regularResults.summary) == 11, ...
    'TrainingSensitivity:IncompleteRegularPanel', 'Regular training panel must have 11 rows.');
assert(height(illResults.summary) == 11, ...
    'TrainingSensitivity:IncompleteIllPanel', 'Ill-conditioned training panel must have 11 rows.');

allResults = merge_panels(regularResults.summary, illResults.summary, config);
validation = validate_run(regularResults, illResults, allResults, config);
assert(all(validation.passed), 'TrainingSensitivity:ValidationFailed', ...
    'Training-sample sensitivity validation failed.');

outputs = output_paths(config);
writetable(allResults, outputs.summary);
cleanup_runtime_checkpoints(config);

if config.make_plots
    if strcmp(config.mode, 'full')
        figure_stem = 'training_mainDGP_aligned_deltaR';
    else
        figure_stem = 'training_dry_deltaR';
    end
    outputs.figures = plot_training_sample_full( ...
        'SummaryFile', outputs.summary, 'OutputDir', config.output_dir, ...
        'StemName', figure_stem);
else
    outputs.figures = {};
end

outputs.mode = config.mode;
outputs.simulation_executed = true;
outputs.regular_panel_rows = height(regularResults.summary);
outputs.ill_conditioned_panel_rows = height(illResults.summary);
outputs.merged_rows = height(allResults);
outputs.full_entry_ready = true;

fprintf(['Training-sample sensitivity %s completed: regular=%d, ', ...
    'ill-conditioned=%d, merged=%d rows.\n'], config.mode, ...
    outputs.regular_panel_rows, outputs.ill_conditioned_panel_rows, outputs.merged_rows);
clear path_cleanup
end

function config = locked_config(mode, varargin)
script_dir = fileparts(mfilename('fullpath'));
module_dir = fileparts(fileparts(script_dir));
project_root = fileparts(fileparts(module_dir));
config = struct();
config.mode = char(mode);
config.module_dir = module_dir;
config.project_root = project_root;
config.dependency_dir = fullfile(module_dir, 'code', 'dependencies');
config.saved_summary_file = fullfile(module_dir, 'tables', 'results', ...
    'training_mainDGP_aligned_full.csv');
config.output_root = fullfile(project_root, 'outputs', 'sensitivity_training');
config.checkpoint_dir = '';
config.N_tr_grid = [40,50,60,80,100,120,160,200,240,320,480];
config.T = 50;
config.n = 10;
config.x0 = 1;
config.Nev = 1000;
config.lambda_scalar = 1e-4;
config.base_seed = 42;
config.replication_seed_offset = 720000;
config.bootstrap_seed = 314159;
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
elseif strcmp(mode, 'dry')
    config.NumSims = config.dry_NumSims;
    config.bootstrap_resamples = config.dry_bootstrap_resamples;
    config.use_parallel = false;
    config.parallel_workers = 0;
    config.batch_size = 1;
else
    config.NumSims = 0;
    config.bootstrap_resamples = 0;
    config.use_parallel = false;
    config.parallel_workers = 0;
    config.batch_size = 1;
end

p = inputParser;
addParameter(p, 'OutputRoot', config.output_root);
addParameter(p, 'UseParallel', config.use_parallel);
addParameter(p, 'ParallelWorkers', config.parallel_workers);
addParameter(p, 'BatchSize', config.batch_size);
addParameter(p, 'NumSims', config.NumSims);
addParameter(p, 'BootstrapResamples', config.bootstrap_resamples);
addParameter(p, 'MakePlots', true);
addParameter(p, 'CheckpointDir', config.checkpoint_dir);
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
        'TrainingSensitivity:LockedFullReplications', 'FULL mode requires S=2000.');
    assert(config.bootstrap_resamples == config.full_bootstrap_resamples, ...
        'TrainingSensitivity:LockedFullBootstrap', 'FULL mode requires 5000 bootstrap resamples.');
end
if ~strcmp(mode, 'plot_only')
    validateattributes(config.NumSims, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(config.bootstrap_resamples, {'numeric'}, {'scalar','integer','positive'});
end
validateattributes(config.batch_size, {'numeric'}, {'scalar','integer','positive'});
assert(isequal(config.N_tr_grid, [40,50,60,80,100,120,160,200,240,320,480]));
assert(config.T == 50 && config.Nev == 1000 && config.lambda_scalar == 1e-4);
config.max_N_tr = max(config.N_tr_grid);
config.lambda = config.lambda_scalar * ones(1, config.T);
switch mode
    case 'dry'
        mode_dir = 'dry_run';
    case 'full'
        mode_dir = 'full_run_candidate';
    otherwise
        mode_dir = 'plot_only';
end
config.output_dir = fullfile(config.output_root, mode_dir);
if isempty(config.checkpoint_dir) && ~strcmp(mode, 'plot_only')
    config.checkpoint_dir = tempname(tempdir);
end
end

function assert_method_path(config)
assert(isfolder(config.dependency_dir), 'TrainingSensitivity:MissingDependencies', ...
    'Missing training dependency directory: %s', config.dependency_dir);
names = {'run_eto.m','run_ieo.m','calculate_oracle.m'};
resolved = {'run_eto','run_ieo','calculate_oracle'};
for k = 1:numel(names)
    expected = fullfile(config.dependency_dir, names{k});
    actual = which(resolved{k});
    assert(isfile(expected) && strcmp(actual, expected), ...
        'TrainingSensitivity:WrongMethodPath', ...
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
local = struct('n', n, 'T', config.T, 'lambda', config.lambda, ...
    'mu_bar', config.regular_mu_bar, ...
    'population_active_l2_target', config.regular_active_l2_target);
zero_mean = config.regular_mu_bar * ones(n, 1);
zero_oracle = make_oracle_record(Sigma, zero_mean, local);
zero_policy = zero_oracle.strategy.A_coef(:, 1) + zero_oracle.strategy.B_coef(:, 1);
delta = calibrate_population_signal(Sigma, signal_direction, zero_policy, local);
assert(abs(delta - config.regular_expected_delta) < 1e-14, ...
    'TrainingSensitivity:RegularDGPFingerprint', 'Regular signal calibration changed.');
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
eigenvalues = eig(Sigma, 'vector');
oracle = make_oracle_record(Sigma, mu, config);
population = struct('mu', mu, 'Sigma', Sigma, ...
    'cholesky', chol(Sigma, 'lower'), 'mu_path', oracle.mu_path, ...
    'Sigma_path', oracle.Sigma_path, 'oracle', oracle.strategy, ...
    'oracle_debug', oracle.debug, 'condition_number', cond(Sigma), ...
    'lambda_min', min(eigenvalues), 'lambda_max', max(eigenvalues));
end

function record = make_oracle_record(Sigma, mu, config)
mu_path = repmat(mu(:)', config.T, 1);
Sigma_path = repmat(reshape(Sigma, 1, config.n, config.n), config.T, 1, 1);
[strategy, debug] = calculate_oracle(mu_path, Sigma_path, config.lambda);
record = struct('strategy', strategy, 'debug', debug, ...
    'mu_path', mu_path, 'Sigma_path', Sigma_path);
end

function assert_dgp_fingerprints(regular, ill, config)
assert(abs(regular.condition_number - 20) < 1e-10, ...
    'TrainingSensitivity:RegularConditionChanged', 'Regular condition number changed.');
assert(abs(mean(diag(regular.Sigma)) - config.regular_marginal_variance) < 1e-16);
assert(abs(ill.condition_number - 472929715.157553) / 472929715.157553 < 1e-10, ...
    'TrainingSensitivity:IllConditionChanged', 'Ill-conditioned DGP changed.');
assert(regular.lambda_min > 0 && ill.lambda_min > 0);
end

function panel = run_training_panel_regular(population, config)
panel = run_training_panel('regular', 'Regular stationary Gaussian', ...
    1, population, config);
end

function panel = run_training_panel_ill_conditioned(population, config)
panel = run_training_panel('ill_conditioned', ...
    'Severely ill-conditioned stationary Gaussian', 2, population, config);
end

function panel = run_training_panel(panel_name, panel_label, panel_index, population, config)
G = numel(config.N_tr_grid);
S = config.NumSims;
R0_ETO = nan(G, S);
R0_IEO = nan(G, S);
valid = false(G, S);
min_regret = nan(G, S);
budget = nan(G, S);
nested_prefix_ok = false(1, S);
err_eto = strings(G, S);
err_ieo = strings(G, S);
checkpoint = fullfile(config.checkpoint_dir, sprintf('%s_checkpoint.mat', panel_name));
metadata = checkpoint_metadata(panel_name, population, config);

if isfile(checkpoint)
    saved = load(checkpoint, 'R0_ETO','R0_IEO','valid','min_regret', ...
        'budget','nested_prefix_ok','err_eto','err_ieo','metadata');
    assert(isfield(saved, 'metadata') && isequal(saved.metadata, metadata), ...
        'TrainingSensitivity:IncompatibleCheckpoint', ...
        'Checkpoint is incompatible with the locked training configuration: %s', checkpoint);
    R0_ETO = saved.R0_ETO;
    R0_IEO = saved.R0_IEO;
    valid = saved.valid;
    min_regret = saved.min_regret;
    budget = saved.budget;
    nested_prefix_ok = saved.nested_prefix_ok;
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
        nested_prefix_ok(s) = one.nested_prefix_ok;
        err_eto(:, s) = one.err_eto;
        err_ieo(:, s) = one.err_ieo;
    end
    atomic_checkpoint(checkpoint, R0_ETO, R0_IEO, valid, min_regret, ...
        budget, nested_prefix_ok, err_eto, err_ieo, metadata);
    fprintf('Training %s panel: %d/%d paired replications complete.\n', ...
        panel_name, sum(all(valid, 1)), S);
end

assert(all(valid, 'all'), 'TrainingSensitivity:InvalidPanelReplication', ...
    '%s panel contains invalid training-size/replication cells.', panel_label);
assert(all(nested_prefix_ok), 'TrainingSensitivity:NestedPrefixFailure', ...
    '%s panel did not use nested training-sample prefixes.', panel_label);
panel = struct('name', panel_name, 'label', panel_label, ...
    'population', population, 'R0_ETO', R0_ETO, 'R0_IEO', R0_IEO, ...
    'valid', valid, 'min_regret', min_regret, 'budget', budget, ...
    'nested_prefix_ok', nested_prefix_ok, 'err_eto', err_eto, 'err_ieo', err_ieo);
panel.summary = summarize_panel(panel, panel_index, config);
end

function one = one_panel_replication(id, population, config)
G = numel(config.N_tr_grid);
one.R0_ETO = nan(G, 1);
one.R0_IEO = nan(G, 1);
one.valid = false(G, 1);
one.min_regret = nan(G, 1);
one.budget = nan(G, 1);
one.err_eto = strings(G, 1);
one.err_ieo = strings(G, 1);
seed = config.base_seed + config.replication_seed_offset + id;
sA = RandStream('mt19937ar', 'Seed', seed + 100);
sB = RandStream('mt19937ar', 'Seed', seed + 200);
zA = randn(sA, config.n, config.max_N_tr, config.T);
zB = randn(sB, config.n, config.Nev, config.T);
RAmax = population.mu + pagemtimes(population.cholesky, zA);
RB = population.mu + pagemtimes(population.cholesky, zB);
one.nested_prefix_ok = verify_nested_prefixes(RAmax, config.N_tr_grid);
for g = 1:G
    Ntr = config.N_tr_grid(g);
    RA = RAmax(:, 1:Ntr, :);
    eto = [];
    ieo = [];
    try
        evalc('eto = run_eto(RA, RB, config.lambda, config.x0);');
    catch ex
        one.err_eto(g) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
    end
    try
        evalc('ieo = run_ieo(RA, RB, config.lambda, config.x0);');
    catch ex
        one.err_ieo(g) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
    end
    if ~isempty(eto) && ~isempty(ieo)
        result = evaluate_population_stage0(eto, ieo, population, config.T);
        one.R0_ETO(g) = result.R0_ETO;
        one.R0_IEO(g) = result.R0_IEO;
        one.min_regret(g) = result.min_regret;
        one.budget(g) = result.max_budget;
        one.valid(g) = all(isfinite([result.R0_ETO, result.R0_IEO])) && ...
            result.min_regret >= -config.negative_regret_tolerance && ...
            result.max_budget <= config.budget_tolerance;
    end
end
end

function ok = verify_nested_prefixes(RAmax, N_grid)
ok = true;
for g = 2:numel(N_grid)
    small = RAmax(:, 1:N_grid(g - 1), :);
    current = RAmax(:, 1:N_grid(g), :);
    ok = ok && isequaln(small, current(:, 1:N_grid(g - 1), :));
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
for g = 1:numel(config.N_tr_grid)
    mask = panel.valid(g, :);
    eto = panel.R0_ETO(g, mask)';
    ieo = panel.R0_IEO(g, mask)';
    difference = eto - ieo;
    ci = paired_bootstrap_ci(difference, config.bootstrap_resamples, ...
        config.bootstrap_seed + 1000 * panel_index + g);
    row = struct('Ntr', config.N_tr_grid(g), ...
        'environment', string(panel.label), ...
        'R0_ETO_pop_mean', mean(eto), ...
        'R0_IEO_pop_mean', mean(ieo), ...
        'Delta_R', mean(difference), ...
        'CI_low', ci(1), 'CI_high', ci(2), ...
        'cond_Sigma', panel.population.condition_number, ...
        'lambda', config.lambda_scalar, 'T', config.T, ...
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

function summary = merge_panels(regularResults, illResults, config)
assert(~isempty(regularResults), 'TrainingSensitivity:EmptyRegularPanel', ...
    'Regular training panel is empty.');
assert(~isempty(illResults), 'TrainingSensitivity:EmptyIllPanel', ...
    'Ill-conditioned training panel is empty.');
assert(height(regularResults) == 11 && height(illResults) == 11);
summary = [regularResults; illResults];
summary.environment = categorical(summary.environment, ...
    {'Regular stationary Gaussian', ...
    'Severely ill-conditioned stationary Gaussian'}, 'Ordinal', true);
summary = sortrows(summary, {'environment','Ntr'});
summary.environment = string(summary.environment);
assert(height(summary) == 2 * numel(config.N_tr_grid));
assert(sum(summary.environment == "Regular stationary Gaussian") == 11);
assert(sum(summary.environment == "Severely ill-conditioned stationary Gaussian") == 11);
assert(max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean))) < 1e-14);
end

function validation = validate_run(regular, ill, summary, config)
checks = strings(0, 1);
values = strings(0, 1);
passed = false(0, 1);
add('regular_panel_nonempty', ~isempty(regular.summary), sprintf('%d rows', height(regular.summary)));
add('ill_panel_nonempty', ~isempty(ill.summary), sprintf('%d rows', height(ill.summary)));
add('regular_panel_11_rows', height(regular.summary) == 11, sprintf('%d', height(regular.summary)));
add('ill_panel_11_rows', height(ill.summary) == 11, sprintf('%d', height(ill.summary)));
add('merged_22_rows', height(summary) == 22, sprintf('%d', height(summary)));
add('training_grid_locked', isequal(unique(summary.Ntr)', config.N_tr_grid), mat2str(unique(summary.Ntr)'));
add('both_environments_present', numel(unique(summary.environment)) == 2, strjoin(unique(summary.environment), ' | '));
add('all_regular_cells_valid', all(regular.valid, 'all'), sprintf('%d/%d', nnz(regular.valid), numel(regular.valid)));
add('all_ill_cells_valid', all(ill.valid, 'all'), sprintf('%d/%d', nnz(ill.valid), numel(ill.valid)));
add('regular_nested_prefixes', all(regular.nested_prefix_ok), sprintf('%d/%d', nnz(regular.nested_prefix_ok), numel(regular.nested_prefix_ok)));
add('ill_nested_prefixes', all(ill.nested_prefix_ok), sprintf('%d/%d', nnz(ill.nested_prefix_ok), numel(ill.nested_prefix_ok)));
add('delta_identity', max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean))) < 1e-14, 'tolerance 1e-14');
add('ci_ordered', all(summary.CI_low <= summary.CI_high), 'CI_low <= CI_high');
add('point_estimate_inside_ci', all(summary.CI_low <= summary.Delta_R & ...
    summary.Delta_R <= summary.CI_high), 'Delta_R inside interval');
add('formal_nonreplication_settings_locked', all(summary.T == 50) && ...
    all(summary.Nev == 1000) && all(summary.lambda == 1e-4), 'T=50 Nev=1000 lambda=1e-4');
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
    'N_tr_grid', config.N_tr_grid, 'NumSims', config.NumSims, ...
    'T', config.T, 'Nev', config.Nev, 'lambda_scalar', config.lambda_scalar, ...
    'base_seed', config.base_seed, ...
    'replication_seed_offset', config.replication_seed_offset, ...
    'mu', population.mu, 'Sigma', population.Sigma);
end

function atomic_checkpoint(target, R0_ETO, R0_IEO, valid, min_regret, ...
    budget, nested_prefix_ok, err_eto, err_ieo, metadata)
temporary = [tempname(fileparts(target)), '.mat'];
save(temporary, 'R0_ETO','R0_IEO','valid','min_regret','budget', ...
    'nested_prefix_ok','err_eto','err_ieo','metadata','-v7.3');
movefile(temporary, target, 'f');
end

function outputs = output_paths(config)
if strcmp(config.mode, 'full')
    summary_name = 'training_mainDGP_aligned_full_candidate.csv';
else
    summary_name = 'training_dry_summary.csv';
end
outputs = struct('summary', fullfile(config.output_dir, summary_name), ...
    'output_dir', config.output_dir);
end

function validation = validate_packaged_summary(summary_file, config)
assert(isfile(summary_file), 'TrainingSensitivity:MissingPackagedSummary', ...
    'Packaged training summary is missing: %s', summary_file);
summary = readtable(summary_file, 'TextType', 'string');
required = {'Ntr','environment','R0_ETO_pop_mean','R0_IEO_pop_mean', ...
    'Delta_R','CI_low','CI_high','cond_Sigma','lambda','T','Nev','S'};
assert(all(ismember(required, summary.Properties.VariableNames)), ...
    'TrainingSensitivity:PackagedSchema', 'Packaged summary schema is incomplete.');
checks = strings(0, 1);
values = strings(0, 1);
passed = false(0, 1);
regular_label = "Regular stationary Gaussian";
ill_label = "Severely ill-conditioned stationary Gaussian";
regular = summary(summary.environment == regular_label, :);
ill = summary(summary.environment == ill_label, :);
delta_residual = max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean)));
add('row_count_22', height(summary) == 22, sprintf('%d', height(summary)));
add('environment_count_2', numel(unique(summary.environment)) == 2, sprintf('%d', numel(unique(summary.environment))));
add('exact_environment_labels', height(regular) == 11 && height(ill) == 11, strjoin(unique(summary.environment), ' | '));
add('regular_rows_11', height(regular) == 11, sprintf('%d', height(regular)));
add('ill_rows_11', height(ill) == 11, sprintf('%d', height(ill)));
add('training_grid_locked', isequal(sort(unique(summary.Ntr))', config.N_tr_grid), mat2str(sort(unique(summary.Ntr))'));
add('S_2000', all(summary.S == 2000), sprintf('%g..%g', min(summary.S), max(summary.S)));
add('Nev_1000', all(summary.Nev == 1000), sprintf('%g..%g', min(summary.Nev), max(summary.Nev)));
add('lambda_1e_4', all(abs(summary.lambda - 1e-4) < 1e-15), sprintf('%.15g', unique(summary.lambda)));
add('T_50', all(summary.T == 50), sprintf('%g..%g', min(summary.T), max(summary.T)));
add('ci_ordered', all(summary.CI_low <= summary.CI_high), 'CI_low <= CI_high');
add('point_estimate_inside_ci', all(summary.CI_low <= summary.Delta_R & ...
    summary.Delta_R <= summary.CI_high), 'CI_low <= Delta_R <= CI_high');
add('delta_identity_with_storage_tolerance', delta_residual <= 5e-8, ...
    sprintf('max residual %.15g; tolerance 5e-8', delta_residual));
add('regular_condition_number', ~isempty(regular) && max(abs(regular.cond_Sigma - 20)) < 1e-8, 'cond=20');
add('ill_condition_number', ~isempty(ill) && max(abs(ill.cond_Sigma - 472929715.157553)) / ...
    472929715.157553 < 1e-8, 'cond=472929715.157553');
validation = table(checks, values, passed, ...
    'VariableNames', {'check','value','passed'});
assert(all(validation.passed), 'TrainingSensitivity:PackagedValidationFailed', ...
    'Packaged training summary failed validation.');
    function add(name, condition, value)
        checks(end + 1, 1) = string(name);
        values(end + 1, 1) = string(value);
        passed(end + 1, 1) = logical(condition);
    end
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

function ensure_directory(path_name)
if ~isfolder(path_name)
    mkdir(path_name);
end
end
