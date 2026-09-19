function results_file = run_nonstationary_simulation(varargin)
%RUN_NONSTATIONARY_SIMULATION Reproduce the nonstationary GARCH experiment.
%
% The default configuration matches the formal Experiment 2 run saved in the
% workspace. Name-value arguments can be used for a shorter diagnostic run:
%
%   run_nonstationary_simulation('NumSims', 2, 'Horizon', 4, ...
%       'TestSize', 50, 'MakePlots', false, 'Verbose', false)

code_dir = fileparts(mfilename('fullpath'));
package_dir = fileparts(code_dir);

parser = inputParser;
addParameter(parser, 'Seed', 42);
addParameter(parser, 'Horizon', 50);
addParameter(parser, 'NumAssets', 10);
addParameter(parser, 'TrainingSize', 60);
addParameter(parser, 'TestSize', 1000);
addParameter(parser, 'NumSims', 2000);
addParameter(parser, 'InitialWealth', 1.0);
addParameter(parser, 'Lambda', 5e-4);
addParameter(parser, 'OutputDir', fullfile(package_dir, 'results', 'generated', ...
    'experiment_2_nonstationary'));
addParameter(parser, 'FigureDir', fullfile(package_dir, 'figures', 'generated', ...
    'experiment_2_nonstationary'));
addParameter(parser, 'MakePlots', true);
addParameter(parser, 'ShowFigures', false);
addParameter(parser, 'Verbose', true);
parse(parser, varargin{:});
opts = parser.Results;

validateattributes(opts.Seed, {'numeric'}, {'scalar', 'integer', 'nonnegative'});
validateattributes(opts.Horizon, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(opts.NumAssets, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(opts.TrainingSize, {'numeric'}, {'scalar', 'integer', '>=', 2});
validateattributes(opts.TestSize, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(opts.NumSims, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(opts.InitialWealth, {'numeric'}, {'scalar', 'finite'});
validateattributes(opts.Lambda, {'numeric'}, {'scalar', 'finite', 'nonnegative'});

output_dir = char(opts.OutputDir);
figure_dir = char(opts.FigureDir);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if opts.MakePlots && ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

config = struct();
config.seed = opts.Seed;
config.T = opts.Horizon;
config.n = opts.NumAssets;
config.N_A = opts.TrainingSize;
config.N_B = opts.TestSize;
config.NumSims = opts.NumSims;
config.x0 = opts.InitialWealth;
config.lambda_scalar = opts.Lambda;
config.x_eval = 1.0;
config.hessian_ridge = 1e-4;
config.experiment_seed_offset = 20000;

lambda = config.lambda_scalar * ones(1, config.T);
Regret_ETO_sims = nan(config.NumSims, config.T);
Regret_IEO_sims = nan(config.NumSims, config.T);
Wealth_Paths_ETO = nan(config.NumSims, config.T + 1);
Wealth_Paths_IEO = nan(config.NumSims, config.T + 1);

for sim = 1:config.NumSims
    current_seed = config.seed + sim + config.experiment_seed_offset;
    rng(current_seed, 'twister');

    [mu_true, Sigma_true] = generate_nonstationary_parameters(config.T, config.n);
    seedA = current_seed + 100;
    seedB = current_seed + 200;
    [R_A, R_B] = generate_data(config.T, config.n, mu_true, Sigma_true, ...
        config.N_A, config.N_B, seedA, seedB);

    if opts.Verbose
        results_eto = run_eto(R_A, R_B, lambda, config.x0);
        results_ieo = run_ieo(R_A, R_B, lambda, config.x0);
    else
        evalc('results_eto = run_eto(R_A, R_B, lambda, config.x0);');
        evalc('results_ieo = run_ieo(R_A, R_B, lambda, config.x0);');
    end

    Wealth_Paths_ETO(sim, :) = results_eto.wealth.X_mean(:)';
    Wealth_Paths_IEO(sim, :) = results_ieo.evaluation.X_mean(:)';

    mu_B = zeros(config.T, config.n);
    Sigma_B = zeros(config.T, config.n, config.n);
    for t = 1:config.T
        mu_B(t, :) = mean(R_B(:, :, t), 2)';
        Sigma_B(t, :, :) = cov(R_B(:, :, t)');
    end

    [oracle_B, oracle_debug] = calculate_oracle(mu_B, Sigma_B, lambda);
    A_eto = results_eto.strategy.A_coef;
    B_eto = results_eto.strategy.B_coef;
    A_ieo = results_ieo.strategy.A_coef;
    B_ieo = results_ieo.strategy.B_coef;

    for t = 1:config.T
        H_t_B = 2 * squeeze(oracle_debug.Sigma_t1(:, :, t)) ...
            + config.hessian_ridge * eye(config.n);
        u_star_B = oracle_B.A_coef(:, t) * config.x_eval ...
            + oracle_B.B_coef(:, t);
        u_eto = A_eto(:, t) * config.x_eval + B_eto(:, t);
        u_ieo = A_ieo(:, t) * config.x_eval + B_ieo(:, t);

        Regret_ETO_sims(sim, t) = 0.5 * (u_eto - u_star_B)' ...
            * H_t_B * (u_eto - u_star_B);
        Regret_IEO_sims(sim, t) = 0.5 * (u_ieo - u_star_B)' ...
            * H_t_B * (u_ieo - u_star_B);
    end

    if opts.Verbose && (sim == 1 || mod(sim, 10) == 0 || sim == config.NumSims)
        fprintf('Nonstationary simulation %d/%d complete.\n', sim, config.NumSims);
    end
end

CumRegret_ETO_sims = fliplr(cumsum(fliplr(Regret_ETO_sims), 2));
CumRegret_IEO_sims = fliplr(cumsum(fliplr(Regret_IEO_sims), 2));

stagewise_regret_mean_ETO = mean(Regret_ETO_sims, 1, 'omitnan');
stagewise_regret_mean_IEO = mean(Regret_IEO_sims, 1, 'omitnan');
wealth_mean_ETO = mean(Wealth_Paths_ETO, 1, 'omitnan');
wealth_mean_IEO = mean(Wealth_Paths_IEO, 1, 'omitnan');

regret_definition = [ ...
    'Out-of-sample approximation to stagewise population regret at x=1. ' ...
    'The benchmark uses Universe-B sample moments and a 1e-4 Hessian ridge, ' ...
    'matching the formal nonstationary Experiment 2 convention.'];

file_name = sprintf('results_nonstationary_T%d_n%d_NA%d_NB%d_seed%d_NumSims%d.mat', ...
    config.T, config.n, config.N_A, config.N_B, config.seed, config.NumSims);
results_file = fullfile(output_dir, file_name);
save(results_file, 'config', 'regret_definition', ...
    'Regret_ETO_sims', 'Regret_IEO_sims', ...
    'CumRegret_ETO_sims', 'CumRegret_IEO_sims', ...
    'Wealth_Paths_ETO', 'Wealth_Paths_IEO', ...
    'stagewise_regret_mean_ETO', 'stagewise_regret_mean_IEO', ...
    'wealth_mean_ETO', 'wealth_mean_IEO');

if opts.MakePlots
    plot_config = struct( ...
        'figure_base_name', 'Experiment_2_Nonstationary', ...
        'regret_csv_name', 'Experiment_2_Nonstationary_stagewise_regret.csv', ...
        'wealth_csv_name', 'Experiment_2_Nonstationary_wealth.csv', ...
        'regret_title', '(a) Stagewise population regret (nonstationary)', ...
        'wealth_title', '(b) Out-of-sample wealth (nonstationary)');
    plot_simulation_results(results_file, figure_dir, opts.ShowFigures, plot_config);
end

if opts.Verbose
    fprintf('Saved simulation results: %s\n', results_file);
end
end

function [mu_true, Sigma_true] = generate_nonstationary_parameters(T, n)
C = 0.2 + 0.6 * rand(n);
C = (C + C') / 2;
C(1:n+1:end) = 1;
[Vcorr, Dcorr] = eig(C);
Dcorr(Dcorr < 1e-2) = 1e-2;
C = Vcorr * Dcorr * Vcorr';
inv_sqrt_diag = diag(1 ./ sqrt(diag(C)));
Corr_mat = inv_sqrt_diag * C * inv_sqrt_diag;

omega = 1e-6;
alpha = 0.05;
beta = 0.90;
long_run_var = omega / (1 - alpha - beta);
vol_t = sqrt(long_run_var) * ones(n, 1);

mu_true = zeros(T, n);
Sigma_true = zeros(T, n, n);

for t = 1:T
    vol_t = max(0.005, vol_t);
    noise_mu = 2e-4 * randn(n, 1);
    mu_daily_t = 1.0001 + 0.05 * vol_t + noise_mu;

    Sigma_daily_t = diag(vol_t) * Corr_mat * diag(vol_t);
    Sigma_daily_t = (Sigma_daily_t + Sigma_daily_t') / 2;

    jitter = 0;
    [~, chol_flag] = chol(Sigma_daily_t, 'lower');
    iteration = 0;
    while chol_flag ~= 0 && iteration < 12
        if jitter == 0
            jitter = max(1e-12, 1e-10 * trace(Sigma_daily_t) / n);
        else
            jitter = jitter * 10;
        end
        Sigma_daily_t = Sigma_daily_t + jitter * eye(n);
        Sigma_daily_t = (Sigma_daily_t + Sigma_daily_t') / 2;
        [~, chol_flag] = chol(Sigma_daily_t, 'lower');
        iteration = iteration + 1;
    end
    if chol_flag ~= 0
        error('Unable to construct a positive-definite nonstationary covariance matrix.');
    end

    mu_true(t, :) = mu_daily_t';
    Sigma_true(t, :, :) = Sigma_daily_t;

    if t < T
        z = randn(n, 1);
        var_next = omega + alpha * (vol_t .* z).^2 + beta * (vol_t.^2);
        vol_t = sqrt(var_next);
    end
end
end
