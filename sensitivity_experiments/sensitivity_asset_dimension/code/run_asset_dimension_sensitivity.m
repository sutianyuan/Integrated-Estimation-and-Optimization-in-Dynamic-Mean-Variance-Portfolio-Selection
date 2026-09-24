function outputs = run_asset_dimension_sensitivity(mode)
%RUN_ASSET_DIMENSION_SENSITIVITY Entry point for asset-dimension sensitivity.
% DRY runs two paired replications; FULL runs the paper's S=2000 design.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
mode = validatestring(lower(char(mode)), {'dry','full'});
code_dir = fileparts(mfilename('fullpath'));
runner_file = fullfile(code_dir, ...
    'run_block_gaussian_j300_variable_k_mc2000_wealth.m');
assert(isfile(runner_file), 'AssetDimension:MissingRunner', ...
    'Asset-dimension runner is missing: %s', runner_file);

original_path = path;
path_cleanup = onCleanup(@() path(original_path));
addpath(code_dir, '-begin');
if strcmp(mode, 'full')
    internal_mode = 'mc2000';
else
    internal_mode = 'smoke';
end
outputs = run_block_gaussian_j300_variable_k_mc2000_wealth(internal_mode);
clear path_cleanup
end
