function outputs = plot_horizon_regret_full(varargin)
%PLOT_HORIZON_REGRET_FULL Plot the final Main-DGP-aligned horizon sensitivity.
% The shaded confidence band is the paired percentile-bootstrap interval
% across Monte Carlo replications for Delta_R = R0_ETO_pop - R0_IEO_pop.

script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(script_dir));
p = inputParser;
addParameter(p, 'SummaryFile', fullfile(root_dir, 'tables', 'results', ...
    'horizon_mainDGP_aligned_full.csv'));
addParameter(p, 'OutputDir', fullfile(root_dir, 'figures'));
addParameter(p, 'StemName', 'horizon_mainDGP_aligned_deltaR');
parse(p, varargin{:});
summary_file = char(p.Results.SummaryFile);
fig_dir = char(p.Results.OutputDir);
if ~isfolder(fig_dir)
    mkdir(fig_dir);
end
assert(isfile(summary_file), 'HorizonSensitivity:MissingPlotSummary', ...
    'Horizon summary CSV is missing: %s', summary_file);
summary = readtable(summary_file, 'TextType', 'string');
validate_summary(summary);
stem = fullfile(fig_dir, char(p.Results.StemName));
make_delta_figure(summary, 'T', 'horizon T', stem, false);
outputs = {stem + ".png", stem + ".pdf"};
end

function validate_summary(summary)
required = {'T','environment','Delta_R','CI_low','CI_high'};
assert(all(ismember(required, summary.Properties.VariableNames)), ...
    'HorizonSensitivity:InvalidPlotSummary', 'Horizon summary schema is incomplete.');
labels = ["Regular stationary Gaussian", ...
    "Severely ill-conditioned stationary Gaussian"];
assert(height(summary) == 78 && numel(unique(summary.T)) == 39, ...
    'HorizonSensitivity:InvalidPlotSummary', 'Expected 78 rows and 39 horizon values.');
for k = 1:numel(labels)
    assert(sum(summary.environment == labels(k)) == 39, ...
        'HorizonSensitivity:InvalidPlotSummary', 'Missing or incomplete panel: %s', labels(k));
end
if all(ismember({'R0_ETO_pop_mean','R0_IEO_pop_mean'}, ...
        summary.Properties.VariableNames))
    delta_from_components = summary.R0_ETO_pop_mean - summary.R0_IEO_pop_mean;
    assert(max(abs(summary.Delta_R - delta_from_components)) <= 5e-8, ...
        'HorizonSensitivity:InvalidPlotSummary', ...
        'Delta_R is inconsistent with ETO - IEO beyond CSV rounding tolerance.');
end
assert(all(summary.CI_low <= summary.Delta_R & summary.Delta_R <= summary.CI_high), ...
    'HorizonSensitivity:InvalidPlotSummary', 'Point estimates must lie inside CI bands.');
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
    end
    xticks(ax, [10 50 100 150 200]);
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
