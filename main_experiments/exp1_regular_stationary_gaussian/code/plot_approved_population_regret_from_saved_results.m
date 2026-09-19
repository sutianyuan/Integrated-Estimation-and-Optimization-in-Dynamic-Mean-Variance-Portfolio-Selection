% Plot only: create an editable population-regret .fig from approved CSV.
% No simulation, ETO/IEO estimation, or DGP generation is performed.
exp_root = fileparts(fileparts(mfilename('fullpath')));
csv_path = fullfile(exp_root, 'reference_outputs', 'results', ...
    'population_regret_stagewise.csv');
figure_path = fullfile(exp_root, 'reference_outputs', 'figures', ...
    'Experiment_3_Regular_Gaussian_Regret.fig');
d = readtable(csv_path);
assert(height(d) == 50 && isequal(d.stage_t, (0:49)'));
assert(all(isfinite(d.regret_pop_eto_mean)) && ...
    all(isfinite(d.regret_pop_ieo_mean)));
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1200, 780]);
ax = axes('Parent', fig); hold(ax, 'on');
plot(ax, d.stage_t, d.regret_pop_eto_mean, '--', ...
    'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
plot(ax, d.stage_t, d.regret_pop_ieo_mean, '-', ...
    'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
grid(ax, 'on'); box(ax, 'on'); xlim(ax, [0, 49]);
xlabel(ax, 'stage t'); ylabel(ax, 'stagewise regret');
title(ax, 'Experiment 1 · Regular stationary Gaussian · Stagewise population regret');
legend(ax, {'ETO', 'IEO'}, 'Location', 'northeast');
savefig(fig, figure_path);
close(fig);
