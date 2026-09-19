%% data_analysis.m
% Academic Data Description for "Time-Consistent Robust Portfolio Optimization"
% Focus: Selected Sub-Universe (Systematic Downsampling)
% =========================================================================
clc; clear; close all;

% 1. Data Preprocessing (Strict Filtering)
% =========================================================================
data_file = 'Fama-French_20行业指数_日度_2017-2025.csv';
if ~exist(data_file, 'file')
    error('Data file not found: %s', data_file);
end

raw_table = readtable(data_file);

% Apply Downsampling Logic
num_cols = width(raw_table);
% User requested 10 assets out of 20. 
% File structure is Date, Name1, Ret1, Name2, Ret2...
% Returns are at 3, 5, 7... (Total 20).
% To select 10, we need every *other* return.
% Indices: 3, 7, 11, ... (Step 4)
ret_indices = 3:4:num_cols; 
R_daily = table2array(raw_table(:, ret_indices));
Selected_Names = raw_table.Properties.VariableNames(ret_indices);

[T, N] = size(R_daily);
fprintf('Selected Sub-Universe Dimensions: %d Days x %d Assets\n', T, N);
fprintf('Selected Assets:\n');
disp(Selected_Names');

% 2. Descriptive Statistics
% =========================================================================
% Assumptions: Daily data, 252 trading days per year
ann_factor = 252;

% Annualized Return
mu_daily = mean(R_daily);
Ann_Ret = mu_daily * ann_factor;

% Annualized Volatility
sigma_daily = std(R_daily);
Ann_Vol = sigma_daily * sqrt(ann_factor);

% Max Drawdown calculation
Max_DD = zeros(1, N);
Cumulative_Returns = cumprod(1 + R_daily);
for i = 1:N
    prices = Cumulative_Returns(:, i);
    peak = cummax(prices);
    drawdown = (prices - peak) ./ peak;
    Max_DD(i) = min(drawdown);
end

% Display Statistics Table
fprintf('\n%-20s | %-12s | %-12s | %-12s\n', 'Asset Name', 'Ann Return', 'Ann Vol', 'Max Drawdown');
fprintf('%s\n', repmat('-', 1, 65));
for i = 1:N
    fprintf('%-20s | %10.2f%% | %10.2f%% | %10.2f%%\n', ...
        Selected_Names{i}, Ann_Ret(i)*100, Ann_Vol(i)*100, Max_DD(i)*100);
end

% 3. Visualization
% =========================================================================
out_dir = fullfile(pwd, 'data_analysis_plots');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Fig 1: Cumulative Wealth Paths
fig1 = figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
plot(Cumulative_Returns, 'LineWidth', 1.5);
title('Cumulative Wealth Paths (Selected Sub-Universe)');
xlabel('Trading Days');
ylabel('Wealth (Initial = 1)');
legend(Selected_Names, 'Location', 'eastoutside', 'Interpreter', 'none');
grid on;
saveas(fig1, fullfile(out_dir, 'Fig1_WealthPaths.png'));

% Fig 2: Correlation Heatmap
fig2 = figure('Color', 'w', 'Position', [150, 150, 900, 800]);
corr_matrix = corr(R_daily);
h = heatmap(Selected_Names, Selected_Names, corr_matrix);
h.Title = 'Correlation Matrix (Selected Sub-Universe)';
h.Colormap = parula;
h.ColorLimits = [0, 1]; % Correlations are usually positive for industry indices
saveas(fig2, fullfile(out_dir, 'Fig2_CorrelationHeatmap.png'));

% Fig 3: Mean-Variance Scatter Plot
fig3 = figure('Color', 'w', 'Position', [200, 200, 800, 600]);
scatter(Ann_Vol*100, Ann_Ret*100, 100, 'filled', 'b');
text(Ann_Vol*100 + 0.5, Ann_Ret*100, Selected_Names, 'Interpreter', 'none', 'FontSize', 8);
title('Risk-Return Profile (Annualized)');
xlabel('Annualized Volatility (%)');
ylabel('Annualized Return (%)');
grid on;
saveas(fig3, fullfile(out_dir, 'Fig3_RiskReturnScatter.png'));

fprintf('\nFigures saved to: %s\n', out_dir);
