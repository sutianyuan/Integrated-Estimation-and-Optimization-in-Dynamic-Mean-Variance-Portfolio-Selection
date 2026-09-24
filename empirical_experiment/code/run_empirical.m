%% Reproduce the fixed-seed empirical illustration reported in Figure 4
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
package_root = fileparts(script_dir);
data_dir = fullfile(package_root, 'data');
results_dir = fullfile(package_root, 'results');
figures_dir = fullfile(package_root, 'figures');
tables_dir = fullfile(package_root, 'tables');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end
if ~exist(tables_dir, 'dir'), mkdir(tables_dir); end
addpath(script_dir, '-begin');
clear run_eto run_ieo;

FIXED_SEED = 19;
T_HORIZON = 30;
LOOKBACK_DAYS = 500;
TRAIN_WINDOW_DAYS = 501;
N_A = 100;
STEP_SIZE = 10;
LAMBDA_VAL = 0.001;
VALUE_GRID_SIZE = 201;
x0 = 1.0;

data_file = fullfile(data_dir, ...
    'Fama-French_20行业指数_日度_2017-2025.csv');
raw_table = readtable(data_file, 'VariableNamingRule', 'preserve');
ret_indices = 3:4:39;
selected_asset_names = raw_table.Properties.VariableNames(ret_indices);
R_daily = table2array(raw_table(:, ret_indices));
R_gross = 1 + R_daily;
[total_days, n_assets] = size(R_gross);
assert(total_days == 2240 && n_assets == 10);

rolling_origins = TRAIN_WINDOW_DAYS:STEP_SIZE: ...
    (total_days - T_HORIZON);
num_windows = numel(rolling_origins);
block_pool_size = TRAIN_WINDOW_DAYS - T_HORIZON + 1;
assert(num_windows == 171 && block_pool_size == 472);
lambda_vec = LAMBDA_VAL * ones(1, T_HORIZON);

starts_file = fullfile(data_dir, 'empirical_block_starts.mat');
starts = load(starts_file, 'FIXED_SEED_SAVED', 'sampled_block_starts');
assert(starts.FIXED_SEED_SAVED == FIXED_SEED);
sampled_block_starts = double(starts.sampled_block_starts);
assert(isequal(size(sampled_block_starts), [N_A, num_windows]));
assert(all(sampled_block_starts(:) >= 1 ...
    & sampled_block_starts(:) <= block_pool_size));

Store_Wealth_ETO = zeros(num_windows, T_HORIZON + 1);
Store_Wealth_IEO = zeros(num_windows, T_HORIZON + 1);
Store_Kappa2_ETO = zeros(num_windows, T_HORIZON);
Store_Kappa1_ETO = zeros(num_windows, T_HORIZON);
Store_Kappa0_ETO = zeros(num_windows, T_HORIZON);
Store_Kappa2_IEO = zeros(num_windows, T_HORIZON);
Store_Kappa1_IEO = zeros(num_windows, T_HORIZON);
Store_Kappa0_IEO = zeros(num_windows, T_HORIZON);

fprintf('Running 171 rolling empirical evaluations.\n');
for k = 1:num_windows
    current_idx = rolling_origins(k);
    training_window = R_gross( ...
        (current_idx - LOOKBACK_DAYS):current_idx, :);
    holdout_block = R_gross( ...
        (current_idx + 1):(current_idx + T_HORIZON), :);
    R_A = sampled_blocks_from_starts( ...
        training_window, sampled_block_starts(:, k), T_HORIZON);
    R_B = permute(holdout_block, [2, 3, 1]);

    res_eto = run_eto(R_A, R_B, lambda_vec, x0);
    res_ieo = run_ieo(R_A, R_B, lambda_vec, x0);

    Store_Wealth_ETO(k, :) = res_eto.wealth.X_mean;
    Store_Wealth_IEO(k, :) = res_ieo.evaluation.X_mean;
    Store_Kappa2_ETO(k, :) = (1 ./ res_eto.recursion.a_t(:))';
    Store_Kappa1_ETO(k, :) = (-res_eto.recursion.lambda_t1(:) ...
        .* res_eto.recursion.q_t(:))';
    Store_Kappa0_ETO(k, :) = (-res_eto.cost_to_go.S_t(:))';
    Store_Kappa2_IEO(k, :) = res_ieo.value_function.kappa_2(:)';
    Store_Kappa1_IEO(k, :) = res_ieo.value_function.kappa_1(:)';
    Store_Kappa0_IEO(k, :) = res_ieo.value_function.kappa_0(:)';

    if mod(k, 25) == 0 || k == num_windows
        fprintf('Completed %d of %d windows.\n', k, num_windows);
    end
end

assert(all(isfinite(Store_Wealth_ETO(:))));
assert(all(isfinite(Store_Wealth_IEO(:))));
Mean_Wealth_ETO = mean(Store_Wealth_ETO, 1);
Mean_Wealth_IEO = mean(Store_Wealth_IEO, 1);

Store_Value_Function_Discrepancy = zeros(num_windows, T_HORIZON);
for t = 1:T_HORIZON
    pooled_states = [Store_Wealth_ETO(:, t); Store_Wealth_IEO(:, t)];
    x_grid = linspace(min(pooled_states), max(pooled_states), ...
        VALUE_GRID_SIZE);
    for k = 1:num_windows
        V_eto = Store_Kappa2_ETO(k, t) .* x_grid.^2 ...
            + Store_Kappa1_ETO(k, t) .* x_grid ...
            + Store_Kappa0_ETO(k, t);
        V_ieo = Store_Kappa2_IEO(k, t) .* x_grid.^2 ...
            + Store_Kappa1_IEO(k, t) .* x_grid ...
            + Store_Kappa0_IEO(k, t);
        Store_Value_Function_Discrepancy(k, t) = ...
            max(abs(V_ieo - V_eto));
    end
end
Mean_Value_Function_Discrepancy = ...
    mean(Store_Value_Function_Discrepancy, 1);
wealth_gap = Mean_Wealth_IEO - Mean_Wealth_ETO;

wealth_table = table((0:T_HORIZON)', Mean_Wealth_ETO', ...
    Mean_Wealth_IEO', wealth_gap', ...
    'VariableNames', {'paper_stage_t', 'ETO_mean', 'IEO_mean', ...
    'IEO_minus_ETO'});
writetable(wealth_table, fullfile(tables_dir, 'wealth.csv'));
value_table = table((0:(T_HORIZON - 1))', ...
    Mean_Value_Function_Discrepancy', ...
    'VariableNames', {'paper_stage_t', ...
    'estimated_value_function_discrepancy'});
writetable(value_table, fullfile(tables_dir, ...
    'value_function_discrepancy.csv'));

result_file = fullfile(results_dir, 'Empirical_Results.mat');
save(result_file, 'FIXED_SEED', 'T_HORIZON', 'LOOKBACK_DAYS', ...
    'TRAIN_WINDOW_DAYS', 'N_A', 'STEP_SIZE', 'LAMBDA_VAL', ...
    'VALUE_GRID_SIZE', 'x0', 'rolling_origins', ...
    'selected_asset_names', 'Mean_Wealth_ETO', 'Mean_Wealth_IEO', ...
    'Mean_Value_Function_Discrepancy', '-v7.3');

fig_value = fixed_canvas([7, 5.5]);
axes('Parent', fig_value, 'Position', [0.15 0.15 0.81 0.79]);
plot(0:(T_HORIZON - 1), Mean_Value_Function_Discrepancy, ...
    'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880], ...
    'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w');
grid on; box on; xlim([0, T_HORIZON - 1]);
xlabel('Stage t');
ylabel('Estimated value-function discrepancy');
save_figure_set(fig_value, figures_dir, ...
    'Figure_A4a_Estimated_Value_Function_Discrepancy');
close(fig_value);

t = 0:T_HORIZON;
fig_wealth = fixed_canvas([7, 6.2]);
ax1 = axes('Parent', fig_wealth, 'Position', [0.15 0.37 0.81 0.57]);
hold(ax1, 'on');
h_eto = plot(ax1, t, Mean_Wealth_ETO, 'b--', 'LineWidth', 2.2, ...
    'Marker', 's', 'MarkerIndices', 2:3:numel(t), ...
    'MarkerSize', 4, 'MarkerFaceColor', 'w');
h_ieo = plot(ax1, t, Mean_Wealth_IEO, 'r-', 'LineWidth', 2.2, ...
    'Marker', 'o', 'MarkerIndices', 1:3:numel(t), ...
    'MarkerSize', 4, 'MarkerFaceColor', 'w');
grid(ax1, 'on'); box(ax1, 'on'); xlim(ax1, [0 T_HORIZON]);
ylabel(ax1, 'Wealth');
set(ax1, 'XTickLabel', []);
legend(ax1, [h_ieo h_eto], {'IEO', 'ETO'}, 'Location', 'best');

ax2 = axes('Parent', fig_wealth, 'Position', [0.15 0.12 0.81 0.17]);
area(ax2, t, wealth_gap, 'FaceColor', [0.85 0.33 0.10], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none');
hold(ax2, 'on');
plot(ax2, t, wealth_gap, '-', 'Color', [0.85 0.20 0.10], ...
    'LineWidth', 1.8, 'Marker', 'o', 'MarkerSize', 3, ...
    'MarkerFaceColor', 'w');
yline(ax2, 0, 'k-', 'LineWidth', 0.8);
grid(ax2, 'on'); box(ax2, 'on'); xlim(ax2, [0 T_HORIZON]);
xlabel(ax2, 'Time t');
ylabel(ax2, 'IEO - ETO');
save_figure_set(fig_wealth, figures_dir, ...
    'Figure_A4b_Out_of_Sample_Wealth_WithGapPanel');
close(fig_wealth);

fprintf('Empirical reproduction completed.\n');

function R_out = sampled_blocks_from_starts(R_src, starts, T)
    [~, n] = size(R_src);
    N = numel(starts);
    R_out = zeros(n, N, T);
    for i = 1:N
        s = starts(i);
        R_out(:, i, :) = R_src(s:(s + T - 1), :)';
    end
end

function fig = fixed_canvas(canvas_size)
    fig = figure('Color', 'w', 'Visible', 'off', 'Units', 'inches', ...
        'Position', [1 1 canvas_size], 'PaperUnits', 'inches', ...
        'PaperPosition', [0 0 canvas_size], 'PaperSize', canvas_size);
    set(fig, 'InvertHardcopy', 'off');
end

function save_figure_set(fig, out_dir, base_name)
    savefig(fig, fullfile(out_dir, [base_name '.fig']));
    print(fig, fullfile(out_dir, [base_name '.png']), '-dpng', '-r300');
    print(fig, fullfile(out_dir, [base_name '.pdf']), '-dpdf', ...
        '-painters', '-bestfit');
end
