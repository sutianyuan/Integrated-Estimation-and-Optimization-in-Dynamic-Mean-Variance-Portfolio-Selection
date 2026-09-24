function outputs = run_horizon_environment_sensitivity(mode, varargin)
%RUN_HORIZON_ENVIRONMENT_SENSITIVITY Entry point for horizon sensitivity.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
code_dir = fileparts(mfilename('fullpath'));
runner_dir = fullfile(code_dir, 'horizon_environment_sensitivity');
runner_file = fullfile(runner_dir, ...
    'run_horizon_environment_sensitivity_core.m');
assert(isfile(runner_file), 'HorizonSensitivity:MissingRunner', ...
    'Horizon runner is missing: %s', runner_file);

original_path = path;
path_cleanup = onCleanup(@() path(original_path));
addpath(runner_dir, '-begin');
outputs = run_horizon_environment_sensitivity_core(mode, varargin{:});
clear path_cleanup
end
