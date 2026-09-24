function outputs = plot_training_sample_full(varargin)
%PLOT_TRAINING_SAMPLE_FULL Plot a validated 22-row training-size summary.
% The shaded confidence band is the paired percentile-bootstrap interval for
% Delta_R = R0_ETO_pop - R0_IEO_pop. OutputDir selects the figure directory.

script_dir = fileparts(mfilename('fullpath'));
module_dir = fileparts(fileparts(script_dir));
project_root = fileparts(fileparts(module_dir));
default_summary = fullfile(module_dir, 'tables', 'results', ...
    'training_mainDGP_aligned_full.csv');
default_output = fullfile(project_root, 'outputs', 'sensitivity_training', ...
    'plot_only');
p = inputParser;
addParameter(p, 'SummaryFile', default_summary);
addParameter(p, 'OutputDir', default_output);
addParameter(p, 'StemName', 'training_mainDGP_aligned_deltaR');
parse(p, varargin{:});
summary_file = char(p.Results.SummaryFile);
output_dir = char(p.Results.OutputDir);
stem_name = char(p.Results.StemName);
assert(isfile(summary_file), 'TrainingSensitivity:MissingPlotSummary', ...
    'Training summary is missing: %s', summary_file);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

summary = readtable(summary_file, 'TextType', 'string');
validate_summary_for_plot(summary);
labels = ["Regular stationary Gaussian", ...
    "Severely ill-conditioned stationary Gaussian"];
colors = {[0.00 0.50 0.15], [0.55 0.10 0.72]};
bands = {[0.70 0.90 0.74], [0.86 0.74 0.92]};

regular_stem = fullfile(output_dir, [stem_name, '_regular']);
make_single_figure(summary(summary.environment == labels(1), :), ...
    labels(1), colors{1}, bands{1}, regular_stem);
ill_stem = fullfile(output_dir, [stem_name, '_ill_conditioned']);
make_single_figure(summary(summary.environment == labels(2), :), ...
    labels(2), colors{2}, bands{2}, ill_stem);
outputs = {[regular_stem '.png'], [regular_stem '.pdf'], ...
    [ill_stem '.png'], [ill_stem '.pdf']};
end

function validate_summary_for_plot(summary)
required = {'Ntr','environment','R0_ETO_pop_mean','R0_IEO_pop_mean', ...
    'Delta_R','CI_low','CI_high'};
assert(all(ismember(required, summary.Properties.VariableNames)), ...
    'TrainingSensitivity:PlotSchema', 'Training plot input schema is incomplete.');
labels = ["Regular stationary Gaussian", ...
    "Severely ill-conditioned stationary Gaussian"];
grid = [40,50,60,80,100,120,160,200,240,320,480];
assert(height(summary) == 22, 'TrainingSensitivity:PlotRowCount', ...
    'Training plot input must have exactly 22 rows.');
assert(numel(unique(summary.environment)) == 2, ...
    'TrainingSensitivity:PlotEnvironmentCount', 'Training plot input must have two environments.');
for k = 1:numel(labels)
    panel = sortrows(summary(summary.environment == labels(k), :), 'Ntr');
    assert(height(panel) == 11 && isequal(panel.Ntr', grid), ...
        'TrainingSensitivity:PlotPanelGrid', ...
        'Each training plot panel must contain the locked 11-point grid.');
end
assert(all(summary.CI_low <= summary.Delta_R & summary.Delta_R <= summary.CI_high), ...
    'TrainingSensitivity:PlotCI', 'Stored point estimates must lie inside ordered CIs.');
residual = max(abs(summary.Delta_R - ...
    (summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean)));
assert(residual <= 5e-8, 'TrainingSensitivity:PlotDeltaDirection', ...
    'Delta_R is inconsistent with ETO minus IEO beyond storage tolerance.');
end

function make_single_figure(panel, label, color, band, stem)
fig = figure('Color', 'white', 'Units', 'pixels', 'Position', [100 100 670 520], ...
    'Visible', 'off', 'Renderer', 'painters');
ax = axes(fig);
draw_panel(ax, sortrows(panel, 'Ntr'), label, color, band, true);
export_figure_pair(fig, stem);
close(fig);
end

function draw_panel(ax, panel, label, color, band, show_legend)
hold(ax, 'on');
x = panel.Ntr(:);
y = panel.Delta_R(:);
lo = panel.CI_low(:);
hi = panel.CI_high(:);
fill(ax, [x; flipud(x)], [lo; flipud(hi)], band, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', '95% CI');
plot(ax, x, y, '-o', 'Color', color, 'LineWidth', 2.2, ...
    'MarkerSize', 4.5, 'MarkerFaceColor', 'white', ...
    'DisplayName', '\Delta_R = ETO - IEO');
yline(ax, 0, '-', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
set(ax, 'XScale', 'log');
xticks(ax, [40 60 100 160 240 480]);
xticklabels(ax, {'40','60','100','160','240','480'});
title(ax, label, 'FontName', 'Arial', 'FontSize', 15);
xlabel(ax, 'training sample size N_{tr}', 'FontName', 'Arial', 'FontSize', 14);
ylabel(ax, '\Delta_R (ETO - IEO)', 'FontName', 'Arial', 'FontSize', 14);
set(ax, 'FontName', 'Arial', 'FontSize', 12, 'Color', 'white', ...
    'Box', 'on', 'LineWidth', 0.75, 'TickDir', 'in', ...
    'XGrid', 'on', 'YGrid', 'on', 'GridColor', [0.70 0.70 0.70], ...
    'GridAlpha', 0.35, 'Layer', 'top');
if show_legend
    legend(ax, 'Location', 'best', 'Box', 'on', 'Color', 'white', ...
        'EdgeColor', [0.25 0.25 0.25], 'FontSize', 11);
end
span = max([0; hi]) - min([0; lo]);
if span == 0
    span = 1;
end
ylim(ax, [min([0; lo]) - 0.10 * span, max([0; hi]) + 0.10 * span]);
end

function export_figure_pair(fig, stem)
exportgraphics(fig, [stem '.png'], 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
end
