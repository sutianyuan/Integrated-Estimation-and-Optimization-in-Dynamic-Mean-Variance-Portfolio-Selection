function outputs = run_training_sample_sensitivity(mode, varargin)
%RUN_TRAINING_SAMPLE_SENSITIVITY Entry point for training-size sensitivity.

if nargin < 1 || isempty(mode)
    mode = 'dry';
end
code_dir = fileparts(mfilename('fullpath'));
runner_dir = fullfile(code_dir, 'training_sample_sensitivity');
runner_file = fullfile(runner_dir, ...
    'run_training_sample_sensitivity_core.m');
assert(isfile(runner_file), 'TrainingSensitivity:MissingRunner', ...
    'Training-sample sensitivity runner is missing: %s', runner_file);

original_path = path;
path_cleanup = onCleanup(@() path(original_path));
addpath(runner_dir, '-begin');
outputs = run_training_sample_sensitivity_core(mode, varargin{:});
clear path_cleanup
end
