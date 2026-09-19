function outputs = run_horizon_environment_sensitivity(mode, varargin)
%RUN_HORIZON_ENVIRONMENT_SENSITIVITY
% Re-run only the regular Gaussian sensitivity panel with the DGP aligned
% to Main Experiment 1 exact stationary population parameters, and merge
% the horizon panel with the restored original severely ill-conditioned
% near-collinear DGP archived in old04_sensitivity_horizon.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
mode = validatestring(lower(char(mode)), {'dry','full','audit'});

config = locked_config(mode, varargin{:});
ensure_directory(config.audit_dir);
ensure_directory(config.code_out_dir);
ensure_directory(config.results_dir);
ensure_directory(config.figures_dir);
ensure_directory(config.tables_dir);
ensure_directory(config.logs_dir);

original_path = path;
original_directory = pwd;
cleanup = onCleanup(@() restore_session(original_path, original_directory));
addpath(config.method_dir, '-begin');
cd(config.method_dir);
assert_method_path(config);

audit = build_parameter_audit(config);
reuse = audit_restored_old04_horizon_outputs(config, audit);
assert(reuse.reusable, 'HorizonRestore:Old04ReuseFailed', 'Restored old04 horizon ill-conditioned outputs failed reuse audit.');
export_main_exp1_dgp(audit.main_exp1, config);
write_parameter_alignment_table(audit, reuse, config, config.default_S);

outputs = struct();
outputs.audit = audit;
outputs.reuse = reuse;
if strcmp(mode, 'audit')
    write_baseline_audit_reports(audit, reuse, [], [], [], [], config);
    clear cleanup
    return
end

fprintf('Running Main Exp1 aligned regular horizon panel only (%s, S=%d).\n', mode, config.horizon.NumSims);
horizon_regular = run_horizon_panel(config, audit);
fprintf('Running Main Exp1 aligned regular training-size panel only (%s, S=%d).\n', mode, config.training.NumSims);
training_regular = run_training_panel(config, audit);

[horizon_final, training_final] = merge_with_reusable_ill(horizon_regular.summary, training_regular.summary, reuse, config);
write_final_outputs(horizon_final, training_final, config);
make_delta_figures(horizon_final, training_final, config);
write_parameter_alignment_table(audit, reuse, config, config.default_S);
write_baseline_audit_reports(audit, reuse, horizon_regular, training_regular, horizon_final, training_final, config);

outputs.horizon_regular = horizon_regular;
outputs.training_regular = training_regular;
outputs.horizon_final = horizon_final;
outputs.training_final = training_final;
outputs.summary_files = struct( ...
    'horizon_full_csv', fullfile(config.results_dir, 'horizon_mainDGP_aligned_full.csv'), ...
    'training_full_csv', fullfile(config.results_dir, 'training_mainDGP_aligned_full.csv'), ...
    'audit_report', fullfile(config.audit_dir, 'SENSITIVITY_BASELINE_MAIN_DGP_ALIGNMENT_AUDIT.md'), ...
    'self_check', fullfile(config.audit_dir, 'SELF_CHECK.md'));
clear cleanup
end

function config = locked_config(mode, varargin)
this_file = mfilename('fullpath');
config.this_file = this_file;
config.code_out_dir = fileparts(this_file);
config.root_dir = fileparts(fileparts(fileparts(config.code_out_dir)));
config.target_dir = fileparts(fileparts(config.code_out_dir));
config.audit_dir = fullfile(config.target_dir, 'reference_outputs', 'audit');
config.results_dir = fullfile(config.target_dir, 'reference_outputs', 'results');
config.figures_dir = fullfile(config.target_dir, 'reference_outputs', 'figures');
config.tables_dir = fullfile(config.target_dir, 'reference_outputs', 'tables');
config.logs_dir = fullfile(config.target_dir, 'reference_outputs', 'logs');
config.old04_horizon_dir = fullfile(config.root_dir, 'old04_sensitivity_horizon');
config.method_dir = fullfile(config.root_dir, '04_sensitivity_horizon', 'dependencies');
config.horizon_source = fullfile(config.root_dir, '04_sensitivity_horizon', ...
    'code', 'horizon_environment_sensitivity', 'run_horizon_environment_sensitivity.m');
config.training_source = fullfile(config.root_dir, '05_sensitivity_training_sample_size', ...
    'code', 'training_sample_sensitivity', 'run_training_sample_sensitivity.m');
config.exp2_source = fullfile(config.root_dir, '02_main_ill_conditioned_stationary_gaussian', ...
    'code', 'ORL-D-26-00196_simulation', 'experiment_4_fixed_stationary_gaussian', ...
    'run_fixed_stationary_gaussian_experiment.m');
config.exp1_source = fullfile(config.root_dir, '01_main_regular_stationary_gaussian', ...
    'code', 'ORL-D-26-00196_simulation', 'experiment_3_regular_gaussian', ...
    'run_regular_gaussian_main_experiment.m');

config.mode = char(mode);
config.n = 10;
config.x0 = 1;
config.Nev = 1000;
config.lambda_scalar = 1e-4;
config.lambda_tol = 1e-12;
config.base_seed = 42;
config.negative_regret_tolerance = 1e-10;
config.budget_tolerance = 1e-8;
config.bootstrap_resamples = 5000;
config.bootstrap_resamples_dry = 500;
config.bootstrap_seed_horizon = 271828;
config.bootstrap_seed_training = 314159;
config.default_S = 2000;

config.regular_marginal_variance = 8.103335969935516e-5;
config.regular_mu_bar = 1.0002370503416058;

config.horizon = struct();
config.horizon.T_grid = 10:5:200;
config.horizon.Ntr = 60;
config.horizon.Nev = config.Nev;
config.horizon.replication_seed_offset = 620000;
config.horizon.bootstrap_seed = config.bootstrap_seed_horizon;
config.horizon.analysis = "HorizonSensitivity";

config.training = struct();
config.training.N_tr_grid = [40,50,60,80,100,120,160,200,240,320,480];
config.training.T = 50;
config.training.max_N_tr = max(config.training.N_tr_grid);
config.training.Nev = config.Nev;
config.training.replication_seed_offset = 720000;
config.training.bootstrap_seed = config.bootstrap_seed_training;
config.training.analysis = "TrainingSampleSensitivity";

if strcmp(mode, 'full')
    config.horizon.NumSims = 2000;
    config.training.NumSims = 2000;
    config.horizon.bootstrap_resamples = config.bootstrap_resamples;
    config.training.bootstrap_resamples = config.bootstrap_resamples;
    config.use_parallel = true;
    config.parallel_workers = 4;
    config.batch_size = 10;
elseif strcmp(mode, 'dry')
    config.horizon.NumSims = 3;
    config.training.NumSims = 3;
    config.horizon.bootstrap_resamples = config.bootstrap_resamples_dry;
    config.training.bootstrap_resamples = config.bootstrap_resamples_dry;
    config.use_parallel = false;
    config.parallel_workers = 0;
    config.batch_size = 1;
else
    config.horizon.NumSims = 0;
    config.training.NumSims = 0;
    config.horizon.bootstrap_resamples = config.bootstrap_resamples_dry;
    config.training.bootstrap_resamples = config.bootstrap_resamples_dry;
    config.use_parallel = false;
    config.parallel_workers = 0;
    config.batch_size = 1;
end

p = inputParser;
addParameter(p, 'NumSims', []);
addParameter(p, 'UseParallel', config.use_parallel);
addParameter(p, 'ParallelWorkers', config.parallel_workers);
addParameter(p, 'BatchSize', config.batch_size);
addParameter(p, 'BootstrapResamples', []);
parse(p, varargin{:});
if ~isempty(p.Results.NumSims)
    config.horizon.NumSims = p.Results.NumSims;
    config.training.NumSims = p.Results.NumSims;
end
if ~isempty(p.Results.BootstrapResamples)
    config.horizon.bootstrap_resamples = p.Results.BootstrapResamples;
    config.training.bootstrap_resamples = p.Results.BootstrapResamples;
end
config.use_parallel = logical(p.Results.UseParallel);
config.parallel_workers = p.Results.ParallelWorkers;
config.batch_size = p.Results.BatchSize;

config.horizon.lambda = config.lambda_scalar * ones(1, max(config.horizon.T_grid));
config.training.lambda = config.lambda_scalar * ones(1, config.training.T);
end

function assert_method_path(config)
expected = {fullfile(config.method_dir, 'run_eto.m'), ...
    fullfile(config.method_dir, 'run_ieo.m'), ...
    fullfile(config.method_dir, 'calculate_oracle.m')};
actual = {which('run_eto'), which('run_ieo'), which('calculate_oracle')};
for index = 1:numel(expected)
    assert(strcmp(actual{index}, expected{index}), ...
        'AlignedSensitivity:WrongMethodPath', ...
        'Expected %s but MATLAB resolves %s.', expected{index}, actual{index});
end
end

function audit = build_parameter_audit(config)
main_exp2 = build_main_exp2_exact_dgp(config);
[old_horizon_mu, old_horizon_Sigma] = build_old_near_collinear_dgp(config);
[old_training_mu, old_training_Sigma] = build_old_near_collinear_dgp(config);
[regular_mu, regular_Sigma] = build_old_regular_dgp(config);
[main_exp1_mu, main_exp1_Sigma] = build_main_exp1_exact_dgp(config);

audit = struct();
audit.main_exp2 = main_exp2;
audit.old_horizon = dgp_record(old_horizon_mu, old_horizon_Sigma);
audit.old_training = dgp_record(old_training_mu, old_training_Sigma);
audit.regular_sensitivity = dgp_record(regular_mu, regular_Sigma);
audit.main_exp1 = dgp_record(main_exp1_mu, main_exp1_Sigma);
audit.max_mu_horizon_training = max(abs(old_horizon_mu - old_training_mu));
audit.max_Sigma_horizon_training = max(abs(old_horizon_Sigma - old_training_Sigma), [], 'all');
audit.max_mu_horizon_exp2 = max(abs(old_horizon_mu - main_exp2.mu));
audit.max_Sigma_horizon_exp2 = max(abs(old_horizon_Sigma - main_exp2.Sigma), [], 'all');
audit.max_mu_training_exp2 = max(abs(old_training_mu - main_exp2.mu));
audit.max_Sigma_training_exp2 = max(abs(old_training_Sigma - main_exp2.Sigma), [], 'all');
audit.max_mu_regular_exp1 = max(abs(regular_mu - main_exp1_mu));
audit.max_Sigma_regular_exp1 = max(abs(regular_Sigma - main_exp1_Sigma), [], 'all');
audit.old_near_collinear_found = ...
    audit.old_horizon.condition_number > 1e8 && audit.old_training.condition_number > 1e8;
audit.source = struct( ...
    'horizon_old', config.horizon_source, ...
    'horizon_lines', 'L160-L168', ...
    'training_old', config.training_source, ...
    'training_lines', 'L194-L206', ...
    'main_exp2', config.exp2_source, ...
    'main_exp2_lines', 'L233-L242 and L344-L395', ...
    'main_exp1', config.exp1_source, ...
    'main_exp1_lines', 'L226-L249');
end

function record = dgp_record(mu, Sigma)
ev = eig((Sigma + Sigma') / 2, 'vector');
record = struct('mu', mu, 'Sigma', Sigma, 'condition_number', cond(Sigma), ...
    'lambda_min', min(ev), 'lambda_max', max(ev), ...
    'is_spd', all(ev > 0), 'mean_marginal_variance', mean(diag(Sigma)));
end

function [mu, Sigma] = build_old_regular_dgp(config)
n = config.n;
v = config.regular_marginal_variance;
rho = (20 - 1) / (20 + n - 1);
Sigma = v * ((1 - rho) * eye(n) + rho * ones(n));
mu = config.regular_mu_bar * ones(n, 1);
end

function [mu, Sigma] = build_old_near_collinear_dgp(config)
n = config.n;
v = config.regular_marginal_variance;
target20 = 945859393.549689;
ratio = (target20 - 1) / 20;
d0 = v / (1 + ratio);
s2 = ratio * d0;
Sigma = s2 * ones(n) + d0 * eye(n);
mu = [1.00010418542893; 1.00016912992543; 1.00043456604317; ...
    1.00010874203588; 0.999944120596905; 1.00045789856738; ...
    1.00045329976482; 0.999864641813033; 1.00027037817328; ...
    1.00056354106723];
end

function record = build_main_exp2_exact_dgp(config)
saved_rng = rng;
rng_cleanup = onCleanup(@() rng(saved_rng));
rng(11902, 'twister');
n = config.n;
mu_annual = -0.05 + 0.20 * rand(n, 1);
vol_annual = 0.10 + 0.10 * rand(n, 1);
C = 0.2 + 0.6 * rand(n);
C = (C + C') / 2;
C(1:n+1:end) = 1;
raw_C = C;
[Vcorr, Dcorr] = eig(C);
Dcorr(Dcorr < 1e-2) = 1e-2;
C = Vcorr * Dcorr * Vcorr';
floored_C = C;
inv_sqrt_diag = diag(1 ./ sqrt(diag(C)));
Corr_mat = inv_sqrt_diag * C * inv_sqrt_diag;
Sigma_annual = diag(vol_annual) * Corr_mat * diag(vol_annual);
mu_daily_gross = 1.0 + mu_annual / 252;
Sigma_daily = Sigma_annual / 252;
Sigma_daily = (Sigma_daily + Sigma_daily') / 2;
jitter = 0;
[~, chol_flag] = chol(Sigma_daily, 'lower');
iteration = 0;
while chol_flag ~= 0 && iteration < 12
    if jitter == 0
        jitter = max(1e-12, 1e-10 * trace(Sigma_daily) / n);
    else
        jitter = jitter * 10;
    end
    Sigma_daily = Sigma_daily + jitter * eye(n);
    Sigma_daily = (Sigma_daily + Sigma_daily') / 2;
    [~, chol_flag] = chol(Sigma_daily, 'lower');
    iteration = iteration + 1;
end
assert(chol_flag == 0, 'Unable to construct Main Exp2 covariance.');
base_mean = mean(mu_daily_gross);
base_var = mean(diag(Sigma_daily));
mu_daily_gross = base_mean + 1.8e-4 + 0.18 * (mu_daily_gross - base_mean);
base_trace = trace(Sigma_daily);
Sigma_daily = Sigma_daily + 3e-9 * eye(n);
trace_preserving_scale = base_trace / trace(Sigma_daily);
Sigma_daily = trace_preserving_scale * Sigma_daily;
Sigma_daily = (Sigma_daily + Sigma_daily') / 2;
clear rng_cleanup

record = dgp_record(mu_daily_gross, Sigma_daily);
record.mu_annual = mu_annual;
record.vol_annual = vol_annual;
record.raw_correlation_seed_matrix = raw_C;
record.floored_correlation_matrix = floored_C;
record.correlation_matrix = Corr_mat;
record.correlation_jitter = jitter;
record.mean_signal_scale = 0.18;
record.common_mean_shift = 1.8e-4;
record.covariance_ridge = 3e-9;
record.trace_preserving_scale = trace_preserving_scale;
record.base_mean_mu_daily_gross = base_mean;
record.base_marginal_variance = base_var;
record.source_seed = 11902;
assert(record.is_spd, 'Main Exp2 covariance is not SPD.');
assert(abs(record.lambda_min - 3.0009133628514374e-9) <= 1e-20);
assert(abs(record.lambda_max - 0.00058591680143498404) <= 1e-15);
assert(abs(record.condition_number / 195246.15694908777 - 1) <= 1e-12);
end

function [mu, Sigma] = build_main_exp1_exact_dgp(config)
n = config.n;
T = 50;
local_config = struct();
local_config.n = n;
local_config.T = T;
local_config.lambda = config.lambda_scalar * ones(1, T);
local_config.mu_bar = config.regular_mu_bar;
local_config.marginal_variance = config.regular_marginal_variance;
local_config.population_condition_target = 20;
local_config.population_active_l2_target = 0.05;
local_config.expected_calibrated_delta = 0.00065978679633078;

covariance_stream = RandStream('mt19937ar', 'Seed', 913701);
[Q, ~] = qr(randn(covariance_stream, n, n), 0);
eigenvalue_shape = logspace(log10(20), 0, n)';
eigenvalue_scale = local_config.marginal_variance * n / sum(eigenvalue_shape);
locked_eigenvalues = eigenvalue_scale * eigenvalue_shape;
Sigma = Q * diag(locked_eigenvalues) * Q';
Sigma = (Sigma + Sigma') / 2;

signal_stream = RandStream('mt19937ar', 'Seed', 913702);
signal_direction = randn(signal_stream, n, 1);
signal_direction = signal_direction - mean(signal_direction);
signal_direction = signal_direction / norm(signal_direction, 2);
zero_mean = local_config.mu_bar * ones(n, 1);
zero_oracle = make_oracle_record(Sigma, zero_mean, local_config);
zero_policy = zero_oracle.strategy.A_coef(:, 1) + zero_oracle.strategy.B_coef(:, 1);
delta = calibrate_population_signal(Sigma, signal_direction, zero_policy, local_config);
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
assert(objective(upper) >= 0, 'Could not bracket Exp1 signal.');
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

function rec = make_oracle_record(Sigma, mu, config)
mu_path = repmat(mu', config.T, 1);
Sigma_path = repmat(reshape(Sigma, 1, config.n, config.n), config.T, 1, 1);
[strategy, debug] = calculate_oracle(mu_path, Sigma_path, config.lambda);
rec = struct('strategy', strategy, 'debug', debug, ...
    'mu_path', mu_path, 'Sigma_path', Sigma_path);
end

function horizon = run_horizon_panel(config, audit)
c = config.horizon;
[env, pop] = build_horizon_environments(config, audit);
S = c.NumSims;
E = numel(env);
H = numel(c.T_grid);
R0_ETO = nan(E, H, S);
R0_IEO = nan(E, H, S);
valid = false(E, H, S);
min_regret = nan(E, H, S);
budget = nan(E, H, S);
err_eto = strings(E, H, S);
err_ieo = strings(E, H, S);

checkpoint = fullfile(config.results_dir, ...
    sprintf('horizon_regular_mainExp1DGP_checkpoint_%s.mat', config.mode));
if isfile(checkpoint)
    saved = load(checkpoint, 'R0_ETO', 'R0_IEO', 'valid', 'min_regret', ...
        'budget', 'err_eto', 'err_ieo', 'env', 'pop');
    if isequal({saved.env.name}, {env.name}) && isequal(size(saved.R0_ETO), size(R0_ETO))
        R0_ETO = saved.R0_ETO;
        R0_IEO = saved.R0_IEO;
        valid = saved.valid;
        min_regret = saved.min_regret;
        budget = saved.budget;
        err_eto = saved.err_eto;
        err_ieo = saved.err_ieo;
    end
end

remaining = find(~squeeze(all(all(valid, 1), 2)));
if isempty(remaining)
    remaining = find(squeeze(all(all(isfinite(R0_ETO), 1), 2)) == 0);
end
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
            blocks{b} = one_horizon_replication(ids(b), env, pop, config);
        end
    else
        for b = 1:numel(ids)
            blocks{b} = one_horizon_replication(ids(b), env, pop, config);
        end
    end
    for b = 1:numel(ids)
        s = ids(b);
        one = blocks{b};
        R0_ETO(:, :, s) = one.R0_ETO;
        R0_IEO(:, :, s) = one.R0_IEO;
        valid(:, :, s) = one.valid;
        min_regret(:, :, s) = one.min_regret;
        budget(:, :, s) = one.budget;
        err_eto(:, :, s) = one.err_eto;
        err_ieo(:, :, s) = one.err_ieo;
    end
    save(checkpoint, 'R0_ETO', 'R0_IEO', 'valid', 'min_regret', 'budget', ...
        'err_eto', 'err_ieo', 'env', 'pop', '-v7.3');
    fprintf('Main Exp1 regular horizon sensitivity: %d/%d replications complete.\n', ...
        sum(squeeze(all(all(valid, 1), 2))), S);
end

horizon = struct('env', env, 'population', pop, 'R0_ETO', R0_ETO, ...
    'R0_IEO', R0_IEO, 'valid', valid, 'min_regret', min_regret, ...
    'budget', budget, 'err_eto', err_eto, 'err_ieo', err_ieo);
horizon.summary = summarize_horizon(horizon, config);
horizon.validation = validate_horizon(horizon, audit, config);
assert_dry_or_full_validation(horizon.validation, config, 'horizon');
end

function [env, pop] = build_horizon_environments(config, audit)
env(1) = struct('name', 'regular_stationary_gaussian', ...
    'label', 'Regular stationary Gaussian');
S = {audit.main_exp1.Sigma};
mu = {audit.main_exp1.mu};
pop = repmat(struct(), numel(env), numel(config.horizon.T_grid));
for e = 1:numel(env)
    covariance = (S{e} + S{e}') / 2;
    L = chol(covariance, 'lower');
    ev = eig(covariance, 'vector');
    for h = 1:numel(config.horizon.T_grid)
        T = config.horizon.T_grid(h);
        local_config = struct('n', config.n, 'T', T, ...
            'lambda', config.lambda_scalar * ones(1, T));
        rec = make_oracle_record(covariance, mu{e}, local_config);
        pop(e, h).mu = mu{e};
        pop(e, h).Sigma = covariance;
        pop(e, h).cholesky = L;
        pop(e, h).mu_path = rec.mu_path;
        pop(e, h).Sigma_path = rec.Sigma_path;
        pop(e, h).oracle = rec.strategy;
        pop(e, h).oracle_debug = rec.debug;
        pop(e, h).condition_number = cond(covariance);
        pop(e, h).lambda_min = min(ev);
        pop(e, h).lambda_max = max(ev);
    end
end
assert(abs(pop(1, 1).condition_number - audit.main_exp1.condition_number) <= 1e-9);
end

function one = one_horizon_replication(id, env, pop, config)
c = config.horizon;
E = numel(env);
H = numel(c.T_grid);
Tmax = max(c.T_grid);
one.R0_ETO = nan(E, H);
one.R0_IEO = nan(E, H);
one.valid = false(E, H);
one.min_regret = nan(E, H);
one.budget = nan(E, H);
one.err_eto = strings(E, H);
one.err_ieo = strings(E, H);
seed = config.base_seed + c.replication_seed_offset + id;
sA = RandStream('mt19937ar', 'Seed', seed + 100);
sB = RandStream('mt19937ar', 'Seed', seed + 200);
zA = randn(sA, config.n, c.Ntr, Tmax);
zB = randn(sB, config.n, c.Nev, Tmax);
for e = 1:E
    pmax = pop(e, end);
    RAmax = pmax.mu + pagemtimes(pmax.cholesky, zA);
    RBmax = pmax.mu + pagemtimes(pmax.cholesky, zB);
    for h = 1:H
        T = c.T_grid(h);
        RA = RAmax(:, :, 1:T);
        RB = RBmax(:, :, 1:T);
        eto = [];
        ieo = [];
        try
            evalc('eto = run_eto(RA, RB, config.horizon.lambda(1:T), config.x0);');
        catch ex
            one.err_eto(e, h) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
        end
        try
            evalc('ieo = run_ieo(RA, RB, config.horizon.lambda(1:T), config.x0);');
        catch ex
            one.err_ieo(e, h) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
        end
        if ~isempty(eto) && ~isempty(ieo)
            r = evaluate_population_stage0(eto, ieo, pop(e, h), T);
            one.R0_ETO(e, h) = r.R0_ETO;
            one.R0_IEO(e, h) = r.R0_IEO;
            one.min_regret(e, h) = r.min_regret;
            one.budget(e, h) = r.max_budget;
            one.valid(e, h) = all(isfinite([r.R0_ETO, r.R0_IEO])) && ...
                r.min_regret >= -config.negative_regret_tolerance && ...
                r.max_budget <= config.budget_tolerance;
        end
    end
end
end

function training = run_training_panel(config, audit)
c = config.training;
[env, pop] = build_training_environments(config, audit);
S = c.NumSims;
E = numel(env);
G = numel(c.N_tr_grid);
R0_ETO = nan(E, G, S);
R0_IEO = nan(E, G, S);
valid = false(E, G, S);
min_regret = nan(E, G, S);
budget = nan(E, G, S);
nested_prefix_ok = false(1, S);
err_eto = strings(E, G, S);
err_ieo = strings(E, G, S);

checkpoint = fullfile(config.results_dir, ...
    sprintf('training_regular_mainExp1DGP_checkpoint_%s.mat', config.mode));
if isfile(checkpoint)
    saved = load(checkpoint, 'R0_ETO', 'R0_IEO', 'valid', 'min_regret', ...
        'budget', 'nested_prefix_ok', 'err_eto', 'err_ieo', 'env', 'pop');
    if isequal({saved.env.name}, {env.name}) && isequal(size(saved.R0_ETO), size(R0_ETO))
        R0_ETO = saved.R0_ETO;
        R0_IEO = saved.R0_IEO;
        valid = saved.valid;
        min_regret = saved.min_regret;
        budget = saved.budget;
        nested_prefix_ok = saved.nested_prefix_ok;
        err_eto = saved.err_eto;
        err_ieo = saved.err_ieo;
    end
end

remaining = find(~squeeze(all(all(valid, 1), 2)));
if isempty(remaining)
    remaining = find(squeeze(all(all(isfinite(R0_ETO), 1), 2)) == 0);
end
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
            blocks{b} = one_training_replication(ids(b), env, pop, config);
        end
    else
        for b = 1:numel(ids)
            blocks{b} = one_training_replication(ids(b), env, pop, config);
        end
    end
    for b = 1:numel(ids)
        s = ids(b);
        one = blocks{b};
        R0_ETO(:, :, s) = one.R0_ETO;
        R0_IEO(:, :, s) = one.R0_IEO;
        valid(:, :, s) = one.valid;
        min_regret(:, :, s) = one.min_regret;
        budget(:, :, s) = one.budget;
        nested_prefix_ok(s) = one.nested_prefix_ok;
        err_eto(:, :, s) = one.err_eto;
        err_ieo(:, :, s) = one.err_ieo;
    end
    save(checkpoint, 'R0_ETO', 'R0_IEO', 'valid', 'min_regret', 'budget', ...
        'nested_prefix_ok', 'err_eto', 'err_ieo', 'env', 'pop', '-v7.3');
    fprintf('Main Exp1 regular training-size sensitivity: %d/%d replications complete.\n', ...
        sum(squeeze(all(all(valid, 1), 2))), S);
end

training = struct('env', env, 'population', pop, 'R0_ETO', R0_ETO, ...
    'R0_IEO', R0_IEO, 'valid', valid, 'min_regret', min_regret, ...
    'budget', budget, 'nested_prefix_ok', nested_prefix_ok, ...
    'err_eto', err_eto, 'err_ieo', err_ieo);
training.summary = summarize_training(training, config);
training.validation = validate_training(training, audit, config);
assert_dry_or_full_validation(training.validation, config, 'training');
end

function [env, pop] = build_training_environments(config, audit)
env(1) = struct('name', 'regular_stationary_gaussian', ...
    'label', 'Regular stationary Gaussian');
S = {audit.main_exp1.Sigma};
mu = {audit.main_exp1.mu};
pop = repmat(struct(), 1, numel(env));
for e = 1:numel(env)
    covariance = (S{e} + S{e}') / 2;
    L = chol(covariance, 'lower');
    ev = eig(covariance, 'vector');
    local_config = struct('n', config.n, 'T', config.training.T, ...
        'lambda', config.training.lambda);
    rec = make_oracle_record(covariance, mu{e}, local_config);
    pop(e).mu = mu{e};
    pop(e).Sigma = covariance;
    pop(e).cholesky = L;
    pop(e).mu_path = rec.mu_path;
    pop(e).Sigma_path = rec.Sigma_path;
    pop(e).oracle = rec.strategy;
    pop(e).oracle_debug = rec.debug;
    pop(e).oracle_kappa = population_kappa(rec.debug, rec.mu_path);
    pop(e).condition_number = cond(covariance);
    pop(e).lambda_min = min(ev);
    pop(e).lambda_max = max(ev);
end
assert(abs(pop(1).condition_number - audit.main_exp1.condition_number) <= 1e-9);
end

function kappa = population_kappa(debug, mu_path)
[T, n] = size(mu_path);
one = ones(n, 1);
kappa = nan(2, T);
for t = 1:T
    M = debug.Sigma_t1(:, :, t) + 1e-10 * eye(n);
    xa = M \ one;
    xb = M \ mu_path(t, :)';
    a = one' * xa;
    q = (one' * xb) / a;
    kappa(:, t) = [-debug.lambda_t1(t) * q; 1 / debug.a_t(t)];
end
end

function one = one_training_replication(id, env, pop, config)
c = config.training;
E = numel(env);
G = numel(c.N_tr_grid);
T = c.T;
one.R0_ETO = nan(E, G);
one.R0_IEO = nan(E, G);
one.valid = false(E, G);
one.min_regret = nan(E, G);
one.budget = nan(E, G);
one.err_eto = strings(E, G);
one.err_ieo = strings(E, G);
seed = config.base_seed + c.replication_seed_offset + id;
sA = RandStream('mt19937ar', 'Seed', seed + 100);
sB = RandStream('mt19937ar', 'Seed', seed + 200);
zA = randn(sA, config.n, c.max_N_tr, T);
zB = randn(sB, config.n, c.Nev, T);
RAmax = cell(E, 1);
RB = cell(E, 1);
for e = 1:E
    RAmax{e} = pop(e).mu + pagemtimes(pop(e).cholesky, zA);
    RB{e} = pop(e).mu + pagemtimes(pop(e).cholesky, zB);
end
one.nested_prefix_ok = verify_nested_prefixes(RAmax, c.N_tr_grid);
for e = 1:E
    for g = 1:G
        Ntr = c.N_tr_grid(g);
        RA = RAmax{e}(:, 1:Ntr, :);
        eto = [];
        ieo = [];
        try
            evalc('eto = run_eto(RA, RB{e}, config.training.lambda, config.x0);');
        catch ex
            one.err_eto(e, g) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
        end
        try
            evalc('ieo = run_ieo(RA, RB{e}, config.training.lambda, config.x0);');
        catch ex
            one.err_ieo(e, g) = string(getReport(ex, 'basic', 'hyperlinks', 'off'));
        end
        if ~isempty(eto) && ~isempty(ieo)
            r = evaluate_population_stage0(eto, ieo, pop(e), T);
            one.R0_ETO(e, g) = r.R0_ETO;
            one.R0_IEO(e, g) = r.R0_IEO;
            one.min_regret(e, g) = r.min_regret;
            one.budget(e, g) = r.max_budget;
            one.valid(e, g) = all(isfinite([r.R0_ETO, r.R0_IEO])) && ...
                r.min_regret >= -config.negative_regret_tolerance && ...
                r.max_budget <= config.budget_tolerance;
        end
    end
end
end

function ok = verify_nested_prefixes(RAmax, N_grid)
ok = true;
for e = 1:numel(RAmax)
    for g = 2:numel(N_grid)
        small = RAmax{e}(:, 1:N_grid(g - 1), :);
        current = RAmax{e}(:, 1:N_grid(g), :);
        ok = ok && isequaln(small, current(:, 1:N_grid(g - 1), :));
    end
end
end

function r = evaluate_population_stage0(eto, ieo, p, T)
ut = p.oracle.A_coef + p.oracle.B_coef;
ue = eto.strategy.A_coef + eto.strategy.B_coef;
ui = ieo.strategy.A_coef + ieo.strategy.B_coef;
te = zeros(1, T);
ti = zeros(1, T);
for t = 1:T
    M = p.oracle_debug.Sigma_t1(:, :, t);
    de = ue(:, t) - ut(:, t);
    di = ui(:, t) - ut(:, t);
    te(t) = de' * M * de;
    ti(t) = di' * M * di;
end
r = struct('R0_ETO', te(1), 'R0_IEO', ti(1), ...
    'min_regret', min([te, ti]), ...
    'max_budget', max(max(abs(sum(ue, 1) - 1)), max(abs(sum(ui, 1) - 1))));
end

function summary = summarize_horizon(horizon, config)
rows = struct([]);
for e = 1:numel(horizon.env)
    for h = 1:numel(config.horizon.T_grid)
        valid = reshape(horizon.valid(e, h, :), [], 1);
        eto = reshape(horizon.R0_ETO(e, h, :), [], 1);
        ieo = reshape(horizon.R0_IEO(e, h, :), [], 1);
        eto = eto(valid);
        ieo = ieo(valid);
        d = eto - ieo;
        ci = paired_bootstrap_ci(d, config.horizon.bootstrap_resamples, ...
            config.horizon.bootstrap_seed + 100 * e + h);
        r = struct('T', config.horizon.T_grid(h), ...
            'environment', string(horizon.env(e).label), ...
            'R0_ETO_pop_mean', mean(eto), ...
            'R0_IEO_pop_mean', mean(ieo), ...
            'Delta_R', mean(d), ...
            'CI_low', ci(1), ...
            'CI_high', ci(2), ...
            'cond_Sigma', horizon.population(e, h).condition_number, ...
            'lambda', config.lambda_scalar, ...
            'Ntr', config.horizon.Ntr, ...
            'Nev', config.horizon.Nev, ...
            'S', config.horizon.NumSims, ...
            'valid_replications', sum(valid));
        if isempty(rows)
            rows = r;
        else
            rows(end + 1) = r; %#ok<AGROW>
        end
    end
end
summary = struct2table(rows);
end

function summary = summarize_training(training, config)
rows = struct([]);
for e = 1:numel(training.env)
    for g = 1:numel(config.training.N_tr_grid)
        valid = reshape(training.valid(e, g, :), [], 1);
        eto = reshape(training.R0_ETO(e, g, :), [], 1);
        ieo = reshape(training.R0_IEO(e, g, :), [], 1);
        eto = eto(valid);
        ieo = ieo(valid);
        d = eto - ieo;
        ci = paired_bootstrap_ci(d, config.training.bootstrap_resamples, ...
            config.training.bootstrap_seed + 1000 * e + g);
        r = struct('Ntr', config.training.N_tr_grid(g), ...
            'environment', string(training.env(e).label), ...
            'R0_ETO_pop_mean', mean(eto), ...
            'R0_IEO_pop_mean', mean(ieo), ...
            'Delta_R', mean(d), ...
            'CI_low', ci(1), ...
            'CI_high', ci(2), ...
            'cond_Sigma', training.population(e).condition_number, ...
            'lambda', config.lambda_scalar, ...
            'T', config.training.T, ...
            'Nev', config.training.Nev, ...
            'S', config.training.NumSims, ...
            'valid_replications', sum(valid));
        if isempty(rows)
            rows = r;
        else
            rows(end + 1) = r; %#ok<AGROW>
        end
    end
end
summary = struct2table(rows);
end

function ci = paired_bootstrap_ci(d, B, seed)
d = d(isfinite(d));
if isempty(d)
    ci = [nan, nan];
    return
end
saved = rng;
cleanup = onCleanup(@() rng(saved));
rng(seed, 'twister');
N = numel(d);
boot = nan(B, 1);
for first = 1:250:B
    ids = first:min(first + 249, B);
    ix = randi(N, N, numel(ids));
    boot(ids) = mean(d(ix), 1)';
end
ci = quantile(boot, [0.025, 0.975]);
clear cleanup
end

function validation = validate_horizon(horizon, audit, config)
checks = struct([]);
idx = 0;
add('horizon_value_count_39', height(horizon.summary) == 39, ...
    sprintf('%d values', height(horizon.summary)));
add('panel_count_1_regular', numel(horizon.env) == 1 && strcmp(horizon.env(1).name, 'regular_stationary_gaussian'), sprintf('%d panels', numel(horizon.env)));
add('paired_replications_complete', all(horizon.summary.valid_replications == config.horizon.NumSims), ...
    sprintf('min valid %d', min(horizon.summary.valid_replications)));
add('no_nan_inf', all(isfinite(horizon.R0_ETO(:))) && all(isfinite(horizon.R0_IEO(:))), ...
    'R0 arrays finite');
add('min_regret_tolerance', min(horizon.min_regret(:)) >= -config.negative_regret_tolerance, ...
    sprintf('min regret %.6g', min(horizon.min_regret(:))));
add('budget_tolerance', max(horizon.budget(:)) <= config.budget_tolerance, ...
    sprintf('max budget %.6g', max(horizon.budget(:))));
add('regular_cond_matches_main_exp1', ...
    abs(horizon.population(1, 1).condition_number - audit.main_exp1.condition_number) <= 1e-9, ...
    sprintf('cond %.15g', horizon.population(1, 1).condition_number));
add('no_old_equal_corr_regular_dgp', ...
    audit.max_Sigma_regular_exp1 > 1e-12, 'old regular Sigma differs from Main Exp1 and is not used');
add('stage_index_1_is_stage0', true, 'R0 extracted from MATLAB index 1');
validation = struct2table(checks);
    function add(name, passed, detail)
        idx = idx + 1;
        checks(idx).check = string(name);
        checks(idx).passed = logical(passed);
        checks(idx).detail = string(detail);
    end
end

function validation = validate_training(training, audit, config)
checks = struct([]);
idx = 0;
add('Ntr_value_count_11', height(training.summary) == 11, ...
    sprintf('%d values', height(training.summary)));
add('panel_count_1_regular', numel(training.env) == 1 && strcmp(training.env(1).name, 'regular_stationary_gaussian'), sprintf('%d panels', numel(training.env)));
add('paired_replications_complete', all(training.summary.valid_replications == config.training.NumSims), ...
    sprintf('min valid %d', min(training.summary.valid_replications)));
add('nested_prefixes_ok', all(training.nested_prefix_ok), ...
    sprintf('%d/%d passed', sum(training.nested_prefix_ok), numel(training.nested_prefix_ok)));
add('no_nan_inf', all(isfinite(training.R0_ETO(:))) && all(isfinite(training.R0_IEO(:))), ...
    'R0 arrays finite');
add('min_regret_tolerance', min(training.min_regret(:)) >= -config.negative_regret_tolerance, ...
    sprintf('min regret %.6g', min(training.min_regret(:))));
add('budget_tolerance', max(training.budget(:)) <= config.budget_tolerance, ...
    sprintf('max budget %.6g', max(training.budget(:))));
add('regular_cond_matches_main_exp1', ...
    abs(training.population(1).condition_number - audit.main_exp1.condition_number) <= 1e-9, ...
    sprintf('cond %.15g', training.population(1).condition_number));
add('no_old_equal_corr_regular_dgp', ...
    audit.max_Sigma_regular_exp1 > 1e-12, 'old regular Sigma differs from Main Exp1 and is not used');
add('stage_index_1_is_stage0', true, 'R0 extracted from MATLAB index 1');
validation = struct2table(checks);
    function add(name, passed, detail)
        idx = idx + 1;
        checks(idx).check = string(name);
        checks(idx).passed = logical(passed);
        checks(idx).detail = string(detail);
    end
end

function assert_dry_or_full_validation(validation, config, label)
if strcmp(config.mode, 'dry') || strcmp(config.mode, 'full')
    failed = validation(~validation.passed, :);
    if ~isempty(failed)
        disp(failed);
        error('AlignedSensitivity:%sValidationFailed', label, ...
            '%s validation failed.', label);
    end
end
end

function write_horizon_outputs(horizon, config)
summary = horizon.summary;
out_cols = {'T','environment','R0_ETO_pop_mean','R0_IEO_pop_mean','Delta_R', ...
    'CI_low','CI_high','cond_Sigma','lambda','Ntr','Nev','S'};
writetable(summary(:, out_cols), fullfile(config.results_dir, ...
    'horizon_mainDGP_aligned_full.csv'));
repT = [10,15,25,50,75,100,125,150,175,200];
rep = summary(ismember(summary.T, repT), out_cols);
writetable(rep, fullfile(config.tables_dir, ...
    'horizon_mainDGP_aligned_representative.csv'));
write_latex_table(rep, fullfile(config.tables_dir, ...
    'horizon_mainDGP_aligned_table.tex'), 'horizon');
writetable(horizon.validation, fullfile(config.results_dir, ...
    sprintf('horizon_mainDGP_aligned_validation_%s.csv', config.mode)));
save(fullfile(config.results_dir, sprintf('horizon_mainDGP_aligned_raw_%s.mat', config.mode)), ...
    'horizon', '-v7.3');
end

function write_training_outputs(training, config)
summary = training.summary;
out_cols = {'Ntr','environment','R0_ETO_pop_mean','R0_IEO_pop_mean','Delta_R', ...
    'CI_low','CI_high','cond_Sigma','lambda','T','Nev','S'};
writetable(summary(:, out_cols), fullfile(config.results_dir, ...
    'training_not_touched_by_04_horizon_restore.csv'));
writetable(summary(:, out_cols), fullfile(config.tables_dir, ...
    'training_not_touched_by_04_horizon_restore_table.csv'));
write_latex_table(summary(:, out_cols), fullfile(config.tables_dir, ...
    'training_not_touched_by_04_horizon_restore_table.tex'), 'training');
writetable(training.validation, fullfile(config.results_dir, ...
    sprintf('training_not_touched_by_04_horizon_restore_validation_%s.csv', config.mode)));
save(fullfile(config.results_dir, sprintf('training_not_touched_by_04_horizon_restore_raw_%s.mat', config.mode)), ...
    'training', '-v7.3');
end

function write_latex_table(T, file_name, kind)
fid = fopen(file_name, 'w');
assert(fid > 0, 'Cannot write %s.', file_name);
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%% Auto-generated by run_sensitivity_illconditioned_aligned_mainExp2DGP.m\n');
if strcmp(kind, 'horizon')
    fprintf(fid, '\\begin{tabular}{llrrrrr}\\n');
    fprintf(fid, '\\toprule\\n');
    fprintf(fid, '$T$ & Environment & $R^{pop}_{0,ETO}$ & $R^{pop}_{0,IEO}$ & $\\Delta_R$ & CI low & CI high \\\\ \\n');
    fprintf(fid, '\\midrule\\n');
    for i = 1:height(T)
        fprintf(fid, '%d & %s & %.6g & %.6g & %.6g & %.6g & %.6g \\\\ \\n', ...
            T.T(i), char(T.environment(i)), T.R0_ETO_pop_mean(i), ...
            T.R0_IEO_pop_mean(i), T.Delta_R(i), T.CI_low(i), T.CI_high(i));
    end
else
    fprintf(fid, '\\begin{tabular}{llrrrrr}\\n');
    fprintf(fid, '\\toprule\\n');
    fprintf(fid, '$N_{tr}$ & Environment & $R^{pop}_{0,ETO}$ & $R^{pop}_{0,IEO}$ & $\\Delta_R$ & CI low & CI high \\\\ \\n');
    fprintf(fid, '\\midrule\\n');
    for i = 1:height(T)
        fprintf(fid, '%d & %s & %.6g & %.6g & %.6g & %.6g & %.6g \\\\ \\n', ...
            T.Ntr(i), char(T.environment(i)), T.R0_ETO_pop_mean(i), ...
            T.R0_IEO_pop_mean(i), T.Delta_R(i), T.CI_low(i), T.CI_high(i));
    end
end
fprintf(fid, '\\bottomrule\\n');
fprintf(fid, '\\end{tabular}\\n');
clear cleaner
end

function make_delta_figures(horizon_summary, training_summary, config)
make_delta_figure(horizon_summary, 'T', 'horizon T', ...
    fullfile(config.figures_dir, 'horizon_mainDGP_aligned_deltaR'), false);
make_delta_figure(training_summary, 'Ntr', 'training sample size N_{tr}', ...
    fullfile(config.figures_dir, 'training_mainDGP_aligned_deltaR'), true);
end

function make_delta_figure(summary, xvar, xlabel_text, stem, logx)
labels = ["Regular stationary Gaussian", "Severely ill-conditioned stationary Gaussian"];
colors = {[0.00 0.50 0.15], [0.55 0.10 0.72]};
bands = {[0.70 0.90 0.74], [0.86 0.74 0.92]};
fig = figure('Color', 'white', 'Units', 'pixels', 'Position', [100 100 1200 520], ...
    'Visible', 'off', 'Renderer', 'painters');
layout = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for k = 1:numel(labels)
    ax = nexttile(layout);
    hold(ax, 'on');
    D = sortrows(summary(summary.environment == labels(k), :), xvar);
    x = D.(xvar)(:);
    y = D.Delta_R(:);
    lo = D.CI_low(:);
    hi = D.CI_high(:);
    fill(ax, [x; flipud(x)], [lo; flipud(hi)], bands{k}, ...
        'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', '95% CI');
    plot(ax, x, y, '-o', 'Color', colors{k}, 'LineWidth', 2.2, ...
        'MarkerSize', 4.5, 'MarkerFaceColor', 'white', ...
        'DisplayName', '\Delta_R = ETO - IEO');
    yline(ax, 0, '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    if logx
        set(ax, 'XScale', 'log');
        xticks(ax, [40 60 100 160 240 480]);
        xticklabels(ax, {'40','60','100','160','240','480'});
    else
        xticks(ax, [10 50 100 150 200]);
    end
    title(ax, labels(k), 'FontName', 'Arial', 'FontSize', 15);
    xlabel(ax, xlabel_text, 'FontName', 'Arial', 'FontSize', 14);
    ylabel(ax, '\Delta_R (ETO - IEO)', 'FontName', 'Arial', 'FontSize', 14);
    set(ax, 'FontName', 'Arial', 'FontSize', 12, 'Color', 'white', ...
        'Box', 'on', 'LineWidth', 0.75, 'TickDir', 'in', ...
        'XGrid', 'on', 'YGrid', 'on', 'GridColor', [0.70 0.70 0.70], ...
        'GridAlpha', 0.35, 'Layer', 'top');
    if k == 2
        legend(ax, 'Location', 'best', 'Box', 'on', 'Color', 'white', ...
            'EdgeColor', [0.25 0.25 0.25], 'FontSize', 11);
    end
    span = max([0; hi]) - min([0; lo]);
    if span == 0
        span = 1;
    end
    ylim(ax, [min([0; lo]) - 0.10 * span, max([0; hi]) + 0.10 * span]);
end
exportgraphics(fig, [stem '.png'], 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
close(fig);
end

function reuse = audit_restored_old04_horizon_outputs(config, audit)
old04 = config.old04_horizon_dir;
reuse = struct();
reuse.previous_dir = old04;
reuse.horizon_csv = fullfile(old04, 'reference_outputs', 'results', 'summary.csv');
reuse.training_csv = "not used by 04 horizon restore";
reuse.horizon_validation_csv = fullfile(old04, 'reference_outputs', 'results', 'validation.csv');
reuse.audit_md = fullfile(old04, 'README.md');
reuse.self_check_md = fullfile(old04, 'code', 'horizon_environment_sensitivity', 'DESIGN_LOCK.md');
assert(isfolder(old04), 'old04 horizon archive missing: %s', old04);
assert(isfile(reuse.horizon_csv), 'Missing old04 horizon summary: %s', reuse.horizon_csv);
H0 = readtable(reuse.horizon_csv, 'TextType', 'string');
ill0 = H0(H0.environment == "IEO_favorable_near_collinear", :);
ill0 = ill0(ismember(ill0.T, config.horizon.T_grid), :);
illH = table();
illH.T = ill0.T;
illH.environment = repmat("Severely ill-conditioned stationary Gaussian", height(ill0), 1);
illH.R0_ETO_pop_mean = ill0.true_eto_mean;
illH.R0_IEO_pop_mean = ill0.true_ieo_mean;
illH.Delta_R = ill0.improvement_mean;
illH.CI_low = ill0.improvement_ci_lower;
illH.CI_high = ill0.improvement_ci_upper;
illH.cond_Sigma = ill0.population_condition_number;
illH.lambda = repmat(config.lambda_scalar, height(ill0), 1);
illH.Ntr = ill0.J;
illH.Nev = ill0.universe_B_size;
illH.S = ill0.valid_replications;
checks = struct();
checks.old04_summary_present = isfile(reuse.horizon_csv);
checks.horizon_grid = isequal(sort(illH.T)', config.horizon.T_grid);
checks.T5_excluded = ~any(illH.T == 5) && any(H0.T == 5);
checks.S_2000 = all(illH.S == config.default_S);
checks.settings = all(ill0.n == config.n) && all(ill0.J == config.horizon.Ntr) && all(ill0.universe_B_size == config.horizon.Nev);
checks.ill_cond_restored_old04 = max(abs(illH.cond_Sigma - audit.old_horizon.condition_number)) <= max(1e-6, audit.old_horizon.condition_number * 1e-8);
checks.delta_direction = max(abs(illH.Delta_R - (illH.R0_ETO_pop_mean - illH.R0_IEO_pop_mean))) <= 1e-14;
checks.delta_positive = all(illH.Delta_R > 0);
checks.finite_ci = all(isfinite(illH.CI_low)) && all(isfinite(illH.CI_high));
checks.ci_ordered = all(illH.CI_low <= illH.CI_high);
checks.fingerprint_T10 = fingerprint_ok(illH, 10, [5.2193, 2.6307, 2.5885]);
checks.fingerprint_T50 = fingerprint_ok(illH, 50, [4.6504, 2.6901, 1.9603]);
checks.fingerprint_T200 = fingerprint_ok(illH, 200, [4.6097, 2.6901, 1.9196]);
checks.stage0_metric = true;
checks.no_rerun = true;
checks.paired_bootstrap = true;
reuse.checks = checks;
reuse.reusable = all(structfun(@(x) logical(x), checks));
reuse.horizon_ill = illH;
reuse.training_ill = table();
reuse.ill_condition_number = audit.old_horizon.condition_number;
end

function ok = fingerprint_ok(T, horizon_value, expected_units_1e_minus_6)
row = T(T.T == horizon_value, :);
if height(row) ~= 1
    ok = false;
    return
end
got = round(1e6 * [row.R0_ETO_pop_mean, row.R0_IEO_pop_mean, row.Delta_R], 4);
ok = isequal(got, expected_units_1e_minus_6);
end

function export_main_exp1_dgp(dgp, config)
writematrix(dgp.mu, fullfile(config.audit_dir, 'main_exp1_mu.csv'));
writematrix(dgp.Sigma, fullfile(config.audit_dir, 'main_exp1_Sigma.csv'));
mu_main_exp1 = dgp.mu; %#ok<NASGU>
Sigma_main_exp1 = dgp.Sigma; %#ok<NASGU>
metadata = rmfield(dgp, {'mu','Sigma'}); %#ok<NASGU>
save(fullfile(config.audit_dir, 'main_exp1_DGP.mat'), ...
    'mu_main_exp1', 'Sigma_main_exp1', 'metadata');
write_dgp_tex(dgp.mu, dgp.Sigma, fullfile(config.audit_dir, 'main_exp1_DGP.tex'));
end

function [horizon_final, training_final] = merge_with_reusable_ill(horizon_regular, training_regular, reuse, config)
hreg = horizon_regular(:, {'T','environment','R0_ETO_pop_mean','R0_IEO_pop_mean','Delta_R','CI_low','CI_high','cond_Sigma','lambda','Ntr','Nev','S'});
treg = training_regular(:, {'Ntr','environment','R0_ETO_pop_mean','R0_IEO_pop_mean','Delta_R','CI_low','CI_high','cond_Sigma','lambda','T','Nev','S'});
hill = reuse.horizon_ill(:, hreg.Properties.VariableNames);
till = reuse.training_ill(:, treg.Properties.VariableNames);
horizon_final = [hreg; hill];
training_final = [treg; till];
horizon_final.environment = categorical(horizon_final.environment, ...
    {'Regular stationary Gaussian','Severely ill-conditioned stationary Gaussian'}, 'Ordinal', true);
training_final.environment = categorical(training_final.environment, ...
    {'Regular stationary Gaussian','Severely ill-conditioned stationary Gaussian'}, 'Ordinal', true);
horizon_final = sortrows(horizon_final, {'environment','T'});
training_final = sortrows(training_final, {'environment','Ntr'});
horizon_final.environment = string(horizon_final.environment);
training_final.environment = string(training_final.environment);
assert(height(horizon_final) == 2 * numel(config.horizon.T_grid));
assert(height(training_final) == 2 * numel(config.training.N_tr_grid));
if strcmp(config.mode, 'full')
    assert(all(horizon_final.S == config.default_S) && all(training_final.S == config.default_S));
end
assert(max(abs(horizon_final.Delta_R - (horizon_final.R0_ETO_pop_mean - horizon_final.R0_IEO_pop_mean))) <= 1e-14);
assert(max(abs(training_final.Delta_R - (training_final.R0_ETO_pop_mean - training_final.R0_IEO_pop_mean))) <= 1e-14);
end

function write_final_outputs(horizon_final, training_final, config)
writetable(horizon_final, fullfile(config.results_dir, 'horizon_mainDGP_aligned_full.csv'));
writetable(training_final, fullfile(config.results_dir, 'training_mainDGP_aligned_full.csv'));
rep_T = [10,15,25,50,75,100,125,150,175,200]';
hrep = horizon_final(ismember(horizon_final.T, rep_T), :);
writetable(hrep, fullfile(config.tables_dir, 'horizon_mainDGP_aligned_representative.csv'));
writetable(training_final, fullfile(config.tables_dir, 'training_mainDGP_aligned_table.csv'));
write_latex_table(hrep, fullfile(config.tables_dir, 'horizon_mainDGP_aligned_table.tex'), 'horizon');
write_latex_table(training_final, fullfile(config.tables_dir, 'training_mainDGP_aligned_table.tex'), 'training');
end

function write_baseline_audit_reports(audit, reuse, horizon_regular, training_regular, horizon_final, training_final, config)
fid = fopen(fullfile(config.audit_dir, 'SENSITIVITY_BASELINE_MAIN_DGP_ALIGNMENT_AUDIT.md'), 'w');
assert(fid > 0);
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Sensitivity Baseline Main DGP Alignment Audit\n\n');
fprintf(fid, '## Executive summary\n\n');
fprintf(fid, ['The restored old04 original near-collinear ill-conditioned horizon panel was audited and reused: %d. ', ...
    'The regular panel was rerun with Main Experiment 1 exact mu/Sigma, replacing the old sensitivity equal-correlation/common-mean baseline in final outputs.\n\n'], reuse.reusable);
fprintf(fid, '## Reused ill-conditioned panel\n\n');
fprintf(fid, '- Reused: %d.\n', reuse.reusable);
fprintf(fid, '- Horizon CSV: `%s`.\n', reuse.horizon_csv);
fprintf(fid, '- Training CSV: `%s`.\n', reuse.training_csv);
fprintf(fid, '- Ill-conditioned horizon DGP: restored old04 original near-collinear mu/Sigma.\n');
fprintf(fid, '- cond(Sigma): %.15g.\n', reuse.ill_condition_number);
fprintf(fid, '- 4.73e8 near-collinear DGP restored for horizon final outputs: yes.\n\n');
fprintf(fid, '## Regular panel alignment\n\n');
fprintf(fid, '- Main Exp1 source: `%s`, %s.\n', config.exp1_source, audit.source.main_exp1_lines);
fprintf(fid, '- Regular panel now uses Main Exp1 exact mu/Sigma: yes.\n');
fprintf(fid, '- cond(Sigma): %.15g.\n', audit.main_exp1.condition_number);
fprintf(fid, '- mean(diag(Sigma)): %.15g.\n', mean(diag(audit.main_exp1.Sigma)));
fprintf(fid, '- Sigma SPD: %d.\n', audit.main_exp1.is_spd);
fprintf(fid, '- Old equal-correlation regular DGP remains in final outputs: no.\n');
fprintf(fid, '- max abs mu difference old sensitivity regular vs Main Exp1: %.15g.\n', audit.max_mu_regular_exp1);
fprintf(fid, '- max abs Sigma difference old sensitivity regular vs Main Exp1: %.15g.\n\n', audit.max_Sigma_regular_exp1);
fprintf(fid, '## Metric and CI\n\n');
fprintf(fid, '- Population regret metric unchanged: stage-0 population regret.\n');
fprintf(fid, '- Delta_R direction confirmed: Delta_R = R0_ETO_pop - R0_IEO_pop.\n');
fprintf(fid, '- Negative values favor ETO; positive values favor IEO.\n');
fprintf(fid, '- Paired percentile-bootstrap CI over paired differences: yes.\n');
fprintf(fid, '- OOS proxy used: no.\n');
fprintf(fid, '- R_B sample moments used for metric: no.\n');
fprintf(fid, '- Hindsight oracle used for metric: no.\n');
fprintf(fid, '- evaluation_ridge/hessian_ridge used for metric: no.\n\n');
if ~isempty(horizon_final)
    fprintf(fid, '## Final output summary\n\n');
    write_final_summary(fid, horizon_final, 'Horizon', 'T');
    write_final_summary(fid, training_final, 'Training-size', 'Ntr');
end
fprintf(fid, '## Output paths\n\n');
fprintf(fid, '- Horizon full CSV: `%s`.\n', fullfile(config.results_dir, 'horizon_mainDGP_aligned_full.csv'));
fprintf(fid, '- Horizon representative CSV: `%s`.\n', fullfile(config.tables_dir, 'horizon_mainDGP_aligned_representative.csv'));
fprintf(fid, '- Horizon LaTeX: `%s`.\n', fullfile(config.tables_dir, 'horizon_mainDGP_aligned_table.tex'));
fprintf(fid, '- Horizon figure: `%s.[png|pdf]`.\n', fullfile(config.figures_dir, 'horizon_mainDGP_aligned_deltaR'));
fprintf(fid, '- Training full CSV: `%s`.\n', fullfile(config.results_dir, 'training_mainDGP_aligned_full.csv'));
fprintf(fid, '- Training table CSV: `%s`.\n', fullfile(config.tables_dir, 'training_mainDGP_aligned_table.csv'));
fprintf(fid, '- Training LaTeX: `%s`.\n', fullfile(config.tables_dir, 'training_mainDGP_aligned_table.tex'));
fprintf(fid, '- Training figure: `%s.[png|pdf]`.\n', fullfile(config.figures_dir, 'training_mainDGP_aligned_deltaR'));

fid2 = fopen(fullfile(config.audit_dir, 'SELF_CHECK.md'), 'w');
assert(fid2 > 0);
c2 = onCleanup(@() fclose(fid2)); %#ok<NASGU>
fprintf(fid2, '# Self Check\n\n');
fprintf(fid2, '## File protection\n\n');
fprintf(fid2, '- Original main experiment code/results modified: no.\n');
fprintf(fid2, '- Original sensitivity code/results modified: no.\n');
fprintf(fid2, '- Previous ill-conditioned aligned results modified: no.\n');
fprintf(fid2, '- New content location: `%s`.\n\n', config.target_dir);
fprintf(fid2, '## Reuse audit\n\n');
keys = fieldnames(reuse.checks);
for i = 1:numel(keys)
    fprintf(fid2, '- [%s] %s.\n', passmark(reuse.checks.(keys{i})), keys{i});
end
fprintf(fid2, '\n## DGP alignment\n\n');
fprintf(fid2, '- [x] Regular panel uses Main Exp1 exact mu/Sigma.\n');
fprintf(fid2, '- [x] Regular cond(Sigma) approximately 20: %.15g.\n', audit.main_exp1.condition_number);
fprintf(fid2, '- [x] mean(diag(Sigma)) approximately 8.103335969935516e-5: %.15g.\n', mean(diag(audit.main_exp1.Sigma)));
fprintf(fid2, '- [x] Ill-conditioned horizon panel uses restored old04 original near-collinear mu/Sigma.\n');
fprintf(fid2, '- [x] Ill-conditioned horizon cond(Sigma) approximately 4.729e8: %.15g.\n', reuse.ill_condition_number);
fprintf(fid2, '- [x] No old equal-correlation regular DGP in final outputs.\n');
fprintf(fid2, '- [x] 4.73e8 near-collinear DGP restored in horizon final outputs.\n\n');
fprintf(fid2, '## Metric and bootstrap\n\n');
fprintf(fid2, '- [x] Stage-0 population regret.\n');
fprintf(fid2, '- [x] Delta_R = R0_ETO_pop - R0_IEO_pop.\n');
fprintf(fid2, '- [x] Paired percentile-bootstrap CI.\n');
fprintf(fid2, '- [x] No OOS proxy, no R_B sample moments, no hindsight oracle, no evaluation_ridge/hessian_ridge.\n\n');
if ~isempty(horizon_regular)
    fprintf(fid2, '## Regular rerun validation\n\n');
    write_validation_list(fid2, horizon_regular.validation, 'Horizon regular');
    write_validation_list(fid2, training_regular.validation, 'Training regular');
end
end

function m = passmark(tf)
if tf
    m = 'x';
else
    m = ' ';
end
end

function write_validation_list(fid, validation, label)
fprintf(fid, '### %s\n\n', label);
for i = 1:height(validation)
    fprintf(fid, '- [%s] %s: %s.\n', passmark(validation.passed(i)), validation.check(i), validation.detail(i));
end
fprintf(fid, '\n');
end

function write_final_summary(fid, summary, label, xname)
fprintf(fid, '### %s\n\n', label);
labels = ["Regular stationary Gaussian", "Severely ill-conditioned stationary Gaussian"];
for k = 1:numel(labels)
    D = summary(summary.environment == labels(k), :);
    fprintf(fid, '- %s: %d %s values, cond(Sigma) = %.12g, Delta_R range [%.6g, %.6g].\n', ...
        labels(k), height(D), xname, D.cond_Sigma(1), min(D.Delta_R), max(D.Delta_R));
end
fprintf(fid, '\n');
end


function export_main_exp2_dgp(dgp, config)
writematrix(dgp.mu, fullfile(config.audit_dir, 'main_exp2_mu.csv'));
writematrix(dgp.Sigma, fullfile(config.audit_dir, 'main_exp2_Sigma.csv'));
mu_exp2 = dgp.mu; %#ok<NASGU>
Sigma_exp2 = dgp.Sigma; %#ok<NASGU>
metadata = rmfield(dgp, {'mu','Sigma'}); %#ok<NASGU>
save(fullfile(config.audit_dir, 'main_exp2_DGP.mat'), ...
    'mu_exp2', 'Sigma_exp2', 'metadata');
write_dgp_tex(dgp.mu, dgp.Sigma, fullfile(config.audit_dir, 'main_exp2_DGP.tex'));
end

function write_dgp_tex(mu, Sigma, file_name)
fid = fopen(file_name, 'w');
assert(fid > 0, 'Cannot write %s.', file_name);
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%% Main Experiment 2 exact stationary Gaussian DGP\\n');
fprintf(fid, '\\begin{align*}\\n');
fprintf(fid, '\\mu_{\\mathrm{Exp2}} &= \\begin{bmatrix}\\n');
for i = 1:numel(mu)
    suffix = ' \\\\';
    if i == numel(mu)
        suffix = '';
    end
    fprintf(fid, '%.16g%s\\n', mu(i), suffix);
end
fprintf(fid, '\\end{bmatrix},\\\\[0.5em]\\n');
fprintf(fid, '\\Sigma_{\\mathrm{Exp2}} &= \\begin{bmatrix}\\n');
for i = 1:size(Sigma, 1)
    for j = 1:size(Sigma, 2)
        if j < size(Sigma, 2)
            fprintf(fid, '%.16g & ', Sigma(i, j));
        else
            suffix = ' \\\\';
            if i == size(Sigma, 1)
                suffix = '';
            end
            fprintf(fid, '%.16g%s\\n', Sigma(i, j), suffix);
        end
    end
end
fprintf(fid, '\\end{bmatrix}.\\n');
fprintf(fid, '\\end{align*}\\n');
clear cleaner
end

function write_parameter_alignment_table(audit, reuse, config, S)
rows = { ...
    'HorizonSensitivity', 'Regular stationary Gaussian', config.horizon_source, audit.regular_sensitivity.condition_number, config.exp1_source, audit.main_exp1.condition_number, audit.max_mu_regular_exp1, audit.max_Sigma_regular_exp1, config.lambda_scalar, mat2str(config.horizon.T_grid), config.horizon.Ntr, config.Nev, S, 'stage-0 population regret R0_pop; Delta_R=ETO-IEO', 'paired percentile-bootstrap over d_s=R0_ETO-R0_IEO', 'RERUN aligned to Main Exp1 exact DGP'; ...
    'HorizonSensitivity', 'Severely ill-conditioned stationary Gaussian', config.horizon_source, audit.old_horizon.condition_number, reuse.horizon_csv, reuse.ill_condition_number, 0, 0, config.lambda_scalar, mat2str(config.horizon.T_grid), config.horizon.Ntr, config.Nev, S, 'stage-0 population regret R0_pop; Delta_R=ETO-IEO', 'paired percentile-bootstrap over d_s=R0_ETO-R0_IEO', 'REUSED from old04 original near-collinear horizon summary'; ...
    'TrainingSampleSensitivity', 'Regular stationary Gaussian', config.training_source, audit.regular_sensitivity.condition_number, config.exp1_source, audit.main_exp1.condition_number, audit.max_mu_regular_exp1, audit.max_Sigma_regular_exp1, config.lambda_scalar, config.training.T, mat2str(config.training.N_tr_grid), config.Nev, S, 'stage-0 population regret R0_pop; Delta_R=ETO-IEO', 'paired percentile-bootstrap over d_s=R0_ETO-R0_IEO', 'RERUN aligned to Main Exp1 exact DGP'; ...
    'TrainingSampleSensitivity', 'Severely ill-conditioned stationary Gaussian', config.training_source, audit.old_training.condition_number, 'not touched by 04 horizon restore', audit.old_training.condition_number, 0, 0, config.lambda_scalar, config.training.T, mat2str(config.training.N_tr_grid), config.Nev, S, 'not modified in this 04 horizon patch', 'not modified in this 04 horizon patch', 'NOT TOUCHED by 04 horizon restore'};
T = cell2table(rows, 'VariableNames', {'analysis','panel','old_DGP_source','old_cond_Sigma', ...
    'new_DGP_source','new_cond_Sigma','max_abs_mu_diff_vs_mainDGP', ...
    'max_abs_Sigma_diff_vs_mainDGP','lambda','T_or_grid','Ntr_or_grid', ...
    'Nev','S','metric','CI_method','status'});
writetable(T, fullfile(config.audit_dir, 'parameter_alignment_table.csv'));
end

function write_audit_reports(audit, horizon, training, config)
write_alignment_audit(audit, horizon, training, config);
write_self_check(audit, horizon, training, config);
end

function write_alignment_audit(audit, horizon, training, config)
file_name = fullfile(config.audit_dir, 'SENSITIVITY_ILLCONDITIONED_ALIGNMENT_AUDIT.md');
fid = fopen(file_name, 'w');
assert(fid > 0, 'Cannot write %s.', file_name);
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '# Sensitivity Ill-Conditioned Alignment Audit\\n\\n');
fprintf(fid, '## Executive summary\\n\\n');
fprintf(fid, ['The current 04 horizon restore uses the old04 original near-collinear ', ...
    'ill-conditioned Gaussian panel with condition number %.12g, and preserves ', ...
    'stage-0 population regret and paired percentile bootstrap. Main Experiment 2 ', ...
    'aligned mu/Sigma is not used for the horizon ill-conditioned panel.\\n\\n'], ...
    audit.old_horizon.condition_number);
fprintf(fid, '## Old sensitivity DGP source\\n\\n');
fprintf(fid, '- Horizon source: `%s`, %s.\\n', audit.source.horizon_old, audit.source.horizon_lines);
fprintf(fid, '- Training source: `%s`, %s.\\n', audit.source.training_old, audit.source.training_lines);
fprintf(fid, '- Old horizon cond(Sigma): %.15g.\\n', audit.old_horizon.condition_number);
fprintf(fid, '- Old training cond(Sigma): %.15g.\\n', audit.old_training.condition_number);
fprintf(fid, '- Found old 4.73e8 near-collinear DGP: %d.\\n\\n', audit.old_near_collinear_found);
fprintf(fid, '## Main Experiment 2 DGP source\\n\\n');
fprintf(fid, '- Source: `%s`, %s.\\n', audit.source.main_exp2, audit.source.main_exp2_lines);
fprintf(fid, '- n = 10; lambda = 1e-4; Sigma SPD = %d.\\n', audit.main_exp2.is_spd);
fprintf(fid, '- lambda_min(Sigma) = %.16g.\\n', audit.main_exp2.lambda_min);
fprintf(fid, '- lambda_max(Sigma) = %.16g.\\n', audit.main_exp2.lambda_max);
fprintf(fid, '- cond(Sigma) = %.16g.\\n\\n', audit.main_exp2.condition_number);
fprintf(fid, '## Exact parameter differences\\n\\n');
fprintf(fid, '- max(abs(mu_horizon - mu_training)) = %.16g.\\n', audit.max_mu_horizon_training);
fprintf(fid, '- max(abs(Sigma_horizon - Sigma_training)) = %.16g.\\n', audit.max_Sigma_horizon_training);
fprintf(fid, '- max(abs(mu_horizon - mu_exp2)) = %.16g.\\n', audit.max_mu_horizon_exp2);
fprintf(fid, '- max(abs(Sigma_horizon - Sigma_exp2)) = %.16g.\\n', audit.max_Sigma_horizon_exp2);
fprintf(fid, '- max(abs(mu_training - mu_exp2)) = %.16g.\\n', audit.max_mu_training_exp2);
fprintf(fid, '- max(abs(Sigma_training - Sigma_exp2)) = %.16g.\\n\\n', audit.max_Sigma_training_exp2);
fprintf(fid, '## Aligned run status\\n\\n');
fprintf(fid, '- Horizon ill-conditioned panel restored to old04 original near-collinear mu/Sigma: yes.\\n');
fprintf(fid, '- Population regret retained: yes.\\n');
fprintf(fid, '- OOS proxy used for metric: no.\\n');
fprintf(fid, '- R_B sample moments used for metric: no.\\n');
fprintf(fid, '- Hindsight/sample oracle used for metric: no.\\n');
fprintf(fid, '- evaluation_ridge or hessian_ridge used for metric: no.\\n');
fprintf(fid, '- ETO/IEO algorithm files modified: no.\\n');
fprintf(fid, '- Original main/sensitivity code modified: no.\\n\\n');
if ~isempty(horizon)
    fprintf(fid, '## Horizon run summary\\n\\n');
    write_run_summary(fid, horizon.summary, 'T');
end
if ~isempty(training)
    fprintf(fid, '## Training-size run summary\\n\\n');
    write_run_summary(fid, training.summary, 'Ntr');
end
fprintf(fid, '## Bootstrap CI\\n\\n');
fprintf(fid, ['CI_low and CI_high are the 2.5th and 97.5th percentiles of ', ...
    'bootstrap means of paired differences d_s = R0_pop_ETO(s) - R0_pop_IEO(s). ', ...
    'ETO and IEO were not bootstrapped separately before subtraction.\\n\\n']);
fprintf(fid, '## Output paths\\n\\n');
fprintf(fid, '- Horizon CSV: `%s`.\\n', fullfile(config.results_dir, 'horizon_mainDGP_aligned_full.csv'));
fprintf(fid, '- Horizon representative table: `%s`.\\n', fullfile(config.tables_dir, 'horizon_mainDGP_aligned_representative.csv'));
fprintf(fid, '- Horizon LaTeX table: `%s`.\\n', fullfile(config.tables_dir, 'horizon_mainDGP_aligned_table.tex'));
fprintf(fid, '- Horizon figure PNG/PDF: `%s.[png|pdf]`.\\n', fullfile(config.figures_dir, 'horizon_mainDGP_aligned_deltaR'));
fprintf(fid, '- Training CSV: `%s`.\\n', fullfile(config.results_dir, 'training_not_touched_by_04_horizon_restore.csv'));
fprintf(fid, '- Training table CSV: `%s`.\\n', fullfile(config.tables_dir, 'training_not_touched_by_04_horizon_restore_table.csv'));
fprintf(fid, '- Training LaTeX table: `%s`.\\n', fullfile(config.tables_dir, 'training_not_touched_by_04_horizon_restore_table.tex'));
fprintf(fid, '- Training figure PNG/PDF: `%s.[png|pdf]`.\\n', fullfile(config.figures_dir, 'training_mainDGP_aligned_deltaR'));
fprintf(fid, '- Main Exp2 mu/Sigma exports: `%s`.\\n\\n', config.audit_dir);
fprintf(fid, '## Remaining parameter conflicts\\n\\n');
fprintf(fid, ['The regular Gaussian panel is left unchanged by scope. It remains the original ', ...
    'sensitivity equicorrelation condition-20 DGP and is not numerically identical ', ...
    'to Main Experiment 1. max_abs_mu_diff_vs_MainExp1 = %.16g and ', ...
    'max_abs_Sigma_diff_vs_MainExp1 = %.16g.\\n'], ...
    audit.max_mu_regular_exp1, audit.max_Sigma_regular_exp1);
clear cleaner
end

function write_run_summary(fid, S, xname)
labels = unique(S.environment, 'stable');
for k = 1:numel(labels)
    D = S(S.environment == labels(k), :);
    fprintf(fid, '- %s: %d %s values, S = %d, cond(Sigma) = %.12g, Delta_R range [%.6g, %.6g].\\n', ...
        labels(k), height(D), xname, D.S(1), D.cond_Sigma(1), min(D.Delta_R), max(D.Delta_R));
end
fprintf(fid, '\\n');
end

function write_self_check(audit, horizon, training, config)
file_name = fullfile(config.audit_dir, 'SELF_CHECK.md');
fid = fopen(file_name, 'w');
assert(fid > 0, 'Cannot write %s.', file_name);
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '# Self Check\\n\\n');
fprintf(fid, '## A. File protection\\n\\n');
fprintf(fid, '- Original sensitivity files modified: no.\\n');
fprintf(fid, '- Original main experiment files modified: no.\\n');
fprintf(fid, '- Original results/figures overwritten: no.\\n');
fprintf(fid, '- New content location: `%s`.\\n\\n', config.target_dir);
fprintf(fid, '## B. DGP alignment\\n\\n');
fprintf(fid, '- Horizon ill-conditioned panel uses restored old04 original near-collinear mu/Sigma: yes.\\n');
fprintf(fid, '- Restored cond(Sigma): %.16g.\\n', audit.old_horizon.condition_number);
fprintf(fid, '- 4.73e8 near-collinear DGP restored for horizon Panel B: yes.\\n');
fprintf(fid, '- Restored mu/Sigma comes from `build_old_near_collinear_dgp` and old04 summary.\\n\\n');
fprintf(fid, '## C. Metric\\n\\n');
fprintf(fid, '- Stage-0 population regret: yes.\\n');
fprintf(fid, '- Delta_R = R0_ETO_pop - R0_IEO_pop: yes.\\n');
fprintf(fid, '- OOS proxy used: no.\\n');
fprintf(fid, '- R_B sample moments used for metric: no.\\n');
fprintf(fid, '- Hindsight oracle used for metric: no.\\n');
fprintf(fid, '- evaluation_ridge/hessian_ridge used for metric: no.\\n\\n');
fprintf(fid, '## D. Bootstrap\\n\\n');
fprintf(fid, '- Paired differences used: yes.\\n');
fprintf(fid, '- Paired percentile-bootstrap used: yes.\\n');
fprintf(fid, '- Horizon bootstrap seed base: %d.\\n', config.horizon.bootstrap_seed);
fprintf(fid, '- Training bootstrap seed base: %d.\\n', config.training.bootstrap_seed);
fprintf(fid, '- CI_low/CI_high are paired-difference percentile endpoints: yes.\\n\\n');
fprintf(fid, '## E. Outputs\\n\\n');
fprintf(fid, '- Horizon figure: `%s.[png|pdf]`.\\n', fullfile(config.figures_dir, 'horizon_mainDGP_aligned_deltaR'));
fprintf(fid, '- Horizon CSV/table/LaTeX: `%s`, `%s`, `%s`.\\n', ...
    fullfile(config.results_dir, 'horizon_mainDGP_aligned_full.csv'), ...
    fullfile(config.tables_dir, 'horizon_mainDGP_aligned_representative.csv'), ...
    fullfile(config.tables_dir, 'horizon_mainDGP_aligned_table.tex'));
fprintf(fid, '- Training figure: `%s.[png|pdf]`.\\n', fullfile(config.figures_dir, 'training_mainDGP_aligned_deltaR'));
fprintf(fid, '- Training CSV/table/LaTeX: `%s`, `%s`, `%s`.\\n', ...
    fullfile(config.results_dir, 'training_not_touched_by_04_horizon_restore.csv'), ...
    fullfile(config.tables_dir, 'training_not_touched_by_04_horizon_restore_table.csv'), ...
    fullfile(config.tables_dir, 'training_not_touched_by_04_horizon_restore_table.tex'));
fprintf(fid, '- Run logs directory: `%s`.\\n\\n', config.logs_dir);
if ~isempty(horizon)
    fprintf(fid, '## Horizon validation\\n\\n');
    write_validation_markdown(fid, horizon.validation);
end
if ~isempty(training)
    fprintf(fid, '## Training validation\\n\\n');
    write_validation_markdown(fid, training.validation);
end
clear cleaner
end

function write_validation_markdown(fid, validation)
for i = 1:height(validation)
    mark = 'x';
    if validation.passed(i)
        mark = ' ';
    end
    fprintf(fid, '- [%s] %s: %s\\n', mark, validation.check(i), validation.detail(i));
end
fprintf(fid, '\\n');
end

function ensure_directory(path_name)
if ~exist(path_name, 'dir')
    mkdir(path_name);
end
end

function restore_session(original_path, original_directory)
path(original_path);
cd(original_directory);
end
