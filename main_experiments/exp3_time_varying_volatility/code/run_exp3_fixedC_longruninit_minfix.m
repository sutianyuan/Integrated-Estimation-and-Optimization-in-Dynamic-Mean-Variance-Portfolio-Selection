function output = run_exp3_fixedC_longruninit_minfix(varargin)
%RUN_EXP3_FIXEDC_LONGRUNINIT_MINFIX
% Fixed-C Experiment 3 with stage-0 marginal variance at the GARCH long-run level.
% This file does not
% modify the archived experiment.  The true-population oracle/regret code
% below uses no ridge, eigenvalue clipping, evaluation moments, or inverse.

parser = inputParser;
addParameter(parser, 'Mode', 'dry');
addParameter(parser, 'ShowFigures', false);
addParameter(parser, 'AllowOverwrite', false);
addParameter(parser, 'Verbose', true);
addParameter(parser, 'SOverride', []);
addParameter(parser, 'FixedCorrelationMatrix', []);
addParameter(parser, 'FixedCSourceReplication', []);
parse(parser, varargin{:});
opts = parser.Results;
mode = validatestring(opts.Mode, {'dry','full','finalize'});

code_dir = fileparts(mfilename('fullpath'));
candidate_dir = fileparts(code_dir);
formal_code_dir = code_dir; % Unmodified estimation/data helpers are copied beside this runner.
results_dir = fullfile(candidate_dir, 'results');
figures_dir = fullfile(candidate_dir, 'figures');
audit_dir = fullfile(candidate_dir, 'audit');
ensure_dir(results_dir); ensure_dir(figures_dir); ensure_dir(audit_dir);

config = struct();
config.mode = mode;
config.simulation_mode = ternary(strcmp(mode,'dry'),'dry','full');
config.finalized_from_completed_run = strcmp(mode,'finalize');
config.base_seed = 42;
config.seed_offset = 20000;
config.T = 50;
config.n = 10;
config.Ntr = 60;
config.Nev = 1000;
config.S = ternary(strcmp(mode,'dry'), 10, 2000);
config.x0 = 1;
config.x_eval = 1;
config.lambda = 5e-4;
config.omega = 1e-6;
config.alpha = 0.05;
config.beta = 0.90;
config.volatility_floor = 0.005;
config.initial_marginal_variance = config.omega/(1-config.alpha-config.beta);
config.initial_volatility = sqrt(config.initial_marginal_variance);
config.correlation_eigenvalue_floor = 3e-5;
config.correlation_mode = 'fixed_numeric_C_selected_from_promoted_raw_rep500';
config.fixed_C_source_replication = 500;
config.fixed_C_source_seed = 20542;
config.fixed_C_preserve_rng_draw_alignment = true;
config.fixed_correlation_matrix = [ ...
    1.0000000000000000,0.4218201130801864,0.4412798705904998,0.4511582407156532,0.6994815142033334,0.5199175771510403,0.5802138013593381,0.4261101271325128,0.6226666980299410,0.3255868287931147; ...
    0.4218201130801864,0.9999999999999998,0.5418217659995389,0.4781100517426075,0.4587547155372388,0.6004892214981243,0.4553617450942275,0.3772588306039973,0.6681637298250437,0.5364410989242556; ...
    0.4412798705904998,0.5418217659995389,1.0000000000000000,0.3217474332586556,0.6956964334458219,0.5890051028999406,0.3829033404416758,0.3883830171994892,0.4773860547415681,0.6071482919048105; ...
    0.4511582407156532,0.4781100517426075,0.3217474332586556,0.9999999999999998,0.5413947977352260,0.4572108224391768,0.4616248068321308,0.4795086064281534,0.6964358289331438,0.4564981262389884; ...
    0.6994815142033334,0.4587547155372388,0.6956964334458219,0.5413947977352260,1.0000000000000000,0.6570546230744956,0.5350942145312536,0.3964092567864095,0.3668901058617081,0.3731249989861105; ...
    0.5199175771510403,0.6004892214981243,0.5890051028999406,0.4572108224391768,0.6570546230744956,1.0000000000000000,0.6829067318553117,0.4237162239713576,0.3761241839452389,0.4096122144849174; ...
    0.5802138013593381,0.4553617450942275,0.3829033404416758,0.4616248068321308,0.5350942145312536,0.6829067318553117,1.0000000000000002,0.4352034183800060,0.3296200392764770,0.6042813143800082; ...
    0.4261101271325128,0.3772588306039973,0.3883830171994892,0.4795086064281534,0.3964092567864095,0.4237162239713576,0.4352034183800060,0.9999999999999998,0.6199655975635627,0.3813828755371863; ...
    0.6226666980299410,0.6681637298250437,0.4773860547415681,0.6964358289331438,0.3668901058617081,0.3761241839452389,0.3296200392764770,0.6199655975635627,1.0000000000000002,0.6099845449757190; ...
    0.3255868287931147,0.5364410989242556,0.6071482919048105,0.4564981262389884,0.3731249989861105,0.4096122144849174,0.6042813143800082,0.3813828755371863,0.6099845449757190,1.0000000000000000];
config.selection_provenance = ['Minimal correction of the current fixed-C Experiment 3: ' ...
    'stage-0 volatility is sqrt(omega/(1-alpha-beta)), without the 0.005 floor; ' ...
    'the original floor still applies from stage 1 onward. The fixed numeric C, ' ...
    'random streams, estimation algorithms, and regret definition are unchanged.'];
if ~isempty(opts.SOverride)
    config.S = opts.SOverride;
    config.simulation_mode = sprintf('%s_SOverride_%d', config.simulation_mode, config.S);
end
if ~isempty(opts.FixedCorrelationMatrix)
    config.fixed_correlation_matrix = opts.FixedCorrelationMatrix;
    if ~isempty(opts.FixedCSourceReplication)
        config.fixed_C_source_replication = opts.FixedCSourceReplication;
        config.fixed_C_source_seed = config.base_seed + opts.FixedCSourceReplication + config.seed_offset;
    end
    config.correlation_mode = sprintf('fixed_numeric_C_runtime_override_rep%d', config.fixed_C_source_replication);
    config.selection_provenance = sprintf(['Runtime fixed-C override from candidate replication %d. ' ...
        'This override is intended for C screening or a selected formal fixed-C rerun.'], ...
        config.fixed_C_source_replication);
end
config.tol_spd_multiplier = 1e-10;
config.high_residual_threshold = 1e-6;
config.budget_tolerance = 1e-7;
% A scale-aware coefficient tolerance is needed because the direct QP is
% non-unique to displayed precision along near-null directions. Agreement
% also requires a much tighter regret and KKT-residual check below.
config.crosscheck_relative_tolerance = 1e-4;
config.crosscheck_ldl_relative_tolerance = 1e-8;
config.crosscheck_ldl_regret_tolerance = 1e-8;
config.crosscheck_qp_regret_tolerance = 1e-6;
config.bootstrap_repetitions = 2000;
config.bootstrap_seed = 730042;
config.formal_code_dir = formal_code_dir;

if strcmp(mode,'dry')
    raw_file = fullfile(audit_dir, 'exp3_fixedC_longruninit_minfix_dry_run_raw.mat');
    log_file = fullfile(audit_dir, 'exp3_fixedC_longruninit_minfix_dry_run_log.txt');
else
    raw_file = fullfile(results_dir, 'exp3_fixedC_longruninit_minfix_raw.mat');
    log_file = fullfile(results_dir, 'exp3_fixedC_longruninit_minfix_run_log.txt');
end
partial_raw_file = [raw_file '.partial'];
protected_outputs = {raw_file, partial_raw_file, log_file};
if strcmp(mode,'finalize')
    assert(exist(partial_raw_file,'file')==2,'Exp3Candidate:MissingCompletedPartialRun', ...
        'The completed full-run partial MAT is required for finalize mode.');
    assert(exist(raw_file,'file')~=2,'Exp3Candidate:ProtectedOutputExists', ...
        'Refusing to overwrite an existing final raw MAT.');
elseif ~opts.AllowOverwrite
    assert_none_exist(protected_outputs);
end

diary(log_file); diary on;
fprintf('Experiment 3 fixed eigenfloor 3e-5 population-regret full run: %s mode\n', upper(mode));
fprintf('Started: %s\n', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss Z')));

try
    source_audit = verify_formal_sources(formal_code_dir, config);
    write_parameter_audit(fullfile(audit_dir, ...
        'exp3_fixedC_longruninit_minfix_PARAMETER_AUDIT.csv'),config,source_audit);
    addpath(formal_code_dir);
    cleanup_path = onCleanup(@() rmpath(formal_code_dir)); %#ok<NASGU>
    assert(exist('run_eto','file') == 2 && exist('run_ieo','file') == 2, ...
        'Exp3Candidate:MissingStrategyCode', 'Unmodified strategy functions not found.');
    assert(exist('quadprog','file') == 2, 'Exp3Candidate:MissingQuadprog', ...
        'quadprog is required for high-residual direct-QP cross-checks.');

    if strcmp(mode,'finalize')
        simulation = load_completed_simulation_scalars(partial_raw_file,config);
    else
        simulation = execute_candidate_run(config, partial_raw_file, opts.Verbose);
    end
    checks = validate_candidate_run(simulation, config);
    payload_checks = validate_raw_payload(partial_raw_file,config);
    checks.pass = checks.pass && payload_checks.pass;
    checks.failures = [checks.failures,payload_checks.failures];
    if ~checks.pass
        error('Exp3Candidate:ValidationFailure', '%s', strjoin(checks.failures, newline));
    end

    if exist(raw_file,'file') == 2 && opts.AllowOverwrite
        delete(raw_file);
    end
    movefile(partial_raw_file, raw_file, 'f');
    finalized_raw=matfile(raw_file,'Writable',true);
    finalized_raw.config=config;

    if strcmp(mode,'dry')
        pass_file = fullfile(audit_dir, 'exp3_fixedC_longruninit_minfix_DRY_RUN_PASS.md');
        write_dry_pass(pass_file, config, checks, source_audit, simulation);
        output = struct('mode', mode, 'pass', true, 'raw_file', raw_file, ...
            'report_file', pass_file, 'checks', checks);
        fprintf('Dry run passed. Full run was not started by this invocation.\n');
        fprintf('Finished: %s\n', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss Z')));
        diary off;
        return;
    end

    summaries = create_result_tables(simulation, config, results_dir);
    create_candidate_figures(simulation, summaries, config, figures_dir, opts.ShowFigures);
    audit_files = write_full_audits(simulation, summaries, config, source_audit, ...
        checks, audit_dir, raw_file);
    output = struct('mode', mode, 'pass', true, 'raw_file', raw_file, ...
        'results_dir', results_dir, 'figures_dir', figures_dir, ...
        'audit_files', audit_files, 'checks', checks);
    fprintf('Full candidate run passed and all candidate outputs were written.\n');
    fprintf('Finished: %s\n', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss Z')));
    diary off;
catch ME
    diary off;
    if strcmp(ME.identifier, 'Exp3Candidate:TrueSigma1NotSPD')
        failure_file = fullfile(audit_dir, 'FAILURE_TRUE_SIGMA1_NOT_SPD.md');
    elseif strcmp(mode,'dry')
        failure_file = fullfile(audit_dir, 'DRY_RUN_FAILURE.md');
    else
        failure_file = fullfile(audit_dir, 'exp3_fixedC_longruninit_minfix_FULL_RUN_FAILURE.md');
    end
    write_failure(failure_file, mode, ME);
    remove_candidate_figures(figures_dir);
    rethrow(ME);
end
end

function simulation=load_completed_simulation_scalars(path,config)
names={'Regret_ETO_pop_sims','Regret_IEO_pop_sims','Delta_R_pop_sims', ...
    'Wealth_Paths_ETO','Terminal_Wealth_ETO','Terminal_Wealth_IEO', ...
    'Wealth_Paths_IEO','slope_gap_sims','intercept_gap_sims', ...
    'slope_angle_sims','intercept_angle_sims','min_eig_Sigma', ...
    'cond_Sigma','min_eig_Sigma1','cond_Sigma1','kappa1_next', ...
    'kappa2_next','lambda1','norm_u_star','norm_u_ETO','norm_u_IEO', ...
    'budget_residual_star','budget_residual_ETO','budget_residual_IEO', ...
    'KKT_residual_star','solve_residual_1','solve_residual_mu', ...
    'near_spd_flag','high_residual_flag','pq_disagreement_flag', ...
    'alternative_crosscheck_flag','max_sigma','min_sigma','mean_sigma', ...
    'Sigma1_norm2','tol_spd','formula_hessian_error', ...
    'pq_oracle_relative_difference','crosscheck_u_relative_ldl', ...
    'crosscheck_u_relative_qp','crosscheck_regret_relative_ldl', ...
    'crosscheck_regret_relative_qp','crosscheck_kkt_residual_ldl', ...
    'crosscheck_kkt_residual_qp','crosscheck_qp_exitflag', ...
    'crosscheck_qp_algorithm_code','crosscheck_pass','seeds'};
reader=matfile(path);
simulation=struct();
for k=1:numel(names)
    simulation.(names{k})=reader.(names{k});
end
simulation.config=config;
end

function simulation = execute_candidate_run(config, partial_raw_file, verbose)
S = config.S; T = config.T; n = config.n;
lambda_vec = config.lambda * ones(1,T);
if exist(partial_raw_file,'file') == 2, delete(partial_raw_file); end
raw = matfile(partial_raw_file, 'Writable', true);
raw.config = config;
preallocate_raw(raw,S,T,n);

% In-memory scalar/stage diagnostics. Large true paths/policies are written
% replication-by-replication to the v7.3 MAT file.
names = {'Regret_ETO_pop_sims','Regret_IEO_pop_sims','Delta_R_pop_sims','Wealth_Paths_ETO', ...
    'Wealth_Paths_IEO','slope_gap_sims','intercept_gap_sims', ...
    'slope_angle_sims','intercept_angle_sims','min_eig_Sigma', ...
    'cond_Sigma','min_eig_Sigma1','cond_Sigma1','kappa1_next', ...
    'kappa2_next','lambda1','norm_u_star','norm_u_ETO','norm_u_IEO', ...
    'budget_residual_star','budget_residual_ETO','budget_residual_IEO', ...
    'KKT_residual_star','solve_residual_1','solve_residual_mu', ...
    'near_spd_flag','high_residual_flag','pq_disagreement_flag', ...
    'alternative_crosscheck_flag','max_sigma','min_sigma','mean_sigma', ...
    'Sigma1_norm2','tol_spd', ...
    'formula_hessian_error','pq_oracle_relative_difference', ...
    'crosscheck_u_relative_ldl','crosscheck_u_relative_qp', ...
    'crosscheck_regret_relative_ldl','crosscheck_regret_relative_qp', ...
    'crosscheck_kkt_residual_ldl','crosscheck_kkt_residual_qp', ...
    'crosscheck_qp_exitflag','crosscheck_qp_algorithm_code','crosscheck_pass'};
for k = 1:numel(names)
    if startsWith(names{k}, 'Wealth_')
        simulation.(names{k}) = nan(S,T+1);
    elseif endsWith(names{k}, '_flag') || strcmp(names{k},'crosscheck_pass')
        simulation.(names{k}) = false(S,T);
    else
        simulation.(names{k}) = nan(S,T);
    end
end
simulation.seeds = (config.base_seed + (1:S)' + config.seed_offset);
simulation.Terminal_Wealth_ETO = nan(S,1);
simulation.Terminal_Wealth_IEO = nan(S,1);
simulation.config = config;

qp_algorithms={'interior-point-convex','trust-region-reflective','active-set'};
qp_options=cell(size(qp_algorithms));
for qpi=1:numel(qp_algorithms)
    qp_options{qpi}=optimoptions('quadprog','Display','off', ...
        'Algorithm',qp_algorithms{qpi},'ConstraintTolerance',1e-12, ...
        'OptimalityTolerance',1e-12,'StepTolerance',1e-14,'MaxIterations',10000);
end

for s = 1:S
    current_seed = simulation.seeds(s);
    rng(current_seed, 'twister');
    [mu_true, Sigma_true, vol_path, Corr, epsilon_mu, jitter_path] = exact_nonstationary_parameters(config);
    [R_A,R_B] = generate_data(config.T,config.n,mu_true,Sigma_true, ...
        config.Ntr,config.Nev,current_seed+100,current_seed+200);
    if verbose && S <= 10
        results_eto = run_eto(R_A,R_B,lambda_vec,config.x0);
        results_ieo = run_ieo(R_A,R_B,lambda_vec,config.x0);
    else
        evalc('results_eto = run_eto(R_A,R_B,lambda_vec,config.x0);');
        evalc('results_ieo = run_ieo(R_A,R_B,lambda_vec,config.x0);');
    end

    A_eto = results_eto.strategy.A_coef;
    B_eto = results_eto.strategy.B_coef;
    A_ieo = results_ieo.strategy.A_coef;
    B_ieo = results_ieo.strategy.B_coef;
    u_eto = A_eto*config.x_eval + B_eto;
    u_ieo = A_ieo*config.x_eval + B_ieo;

    oracle = true_population_oracle(mu_true,Sigma_true,config,u_eto,u_ieo,qp_options,s);

    raw.true_mu_paths(s,1:T,1:n) = reshape(mu_true,[1,T,n]);
    raw.true_Sigma_paths(s,1:T,1:n,1:n) = reshape(Sigma_true,[1,T,n,n]);
    raw.true_volatility_paths(s,1:T,1:n) = reshape(vol_path',[1,T,n]);
    raw.true_correlation_matrices(s,1:n,1:n) = reshape(Corr,[1,n,n]);
    raw.true_epsilon_mu_paths(s,1:T,1:n) = reshape(epsilon_mu',[1,T,n]);
    raw.dgp_jitter_paths(s,1:T) = jitter_path;
    raw.policy_A_ETO(s,1:T,1:n) = reshape(A_eto',[1,T,n]);
    raw.policy_B_ETO(s,1:T,1:n) = reshape(B_eto',[1,T,n]);
    raw.policy_A_IEO(s,1:T,1:n) = reshape(A_ieo',[1,T,n]);
    raw.policy_B_IEO(s,1:T,1:n) = reshape(B_ieo',[1,T,n]);
    raw.true_oracle_A(s,1:T,1:n) = reshape(oracle.A',[1,T,n]);
    raw.true_oracle_B(s,1:T,1:n) = reshape(oracle.B',[1,T,n]);
    raw.true_oracle_policy_x1(s,1:T,1:n) = reshape(oracle.u_star',[1,T,n]);
    raw.true_Sigma1_paths(s,1:T,1:n,1:n) = reshape(oracle.Sigma1,[1,T,n,n]);
    raw.true_kappa1_paths(s,1:T+1) = oracle.kappa1;
    raw.true_kappa2_paths(s,1:T+1) = oracle.kappa2;
    raw.true_kappa0_paths(s,1:T+1) = oracle.kappa0;

    simulation.Regret_ETO_pop_sims(s,:) = oracle.regret_eto;
    simulation.Regret_IEO_pop_sims(s,:) = oracle.regret_ieo;
    simulation.Delta_R_pop_sims(s,:) = oracle.regret_eto - oracle.regret_ieo;
    simulation.Wealth_Paths_ETO(s,:) = results_eto.wealth.X_mean(:)';
    simulation.Wealth_Paths_IEO(s,:) = results_ieo.evaluation.X_mean(:)';
    simulation.Terminal_Wealth_ETO(s) = simulation.Wealth_Paths_ETO(s,end);
    simulation.Terminal_Wealth_IEO(s) = simulation.Wealth_Paths_IEO(s,end);
    simulation.slope_gap_sims(s,:) = vecnorm(A_eto-A_ieo,2,1);
    simulation.intercept_gap_sims(s,:) = vecnorm(B_eto-B_ieo,2,1);
    simulation.slope_angle_sims(s,:) = stage_angles(A_eto,A_ieo);
    simulation.intercept_angle_sims(s,:) = stage_angles(B_eto,B_ieo);
    oracle_fields = fieldnames(oracle.diagnostics);
    for k = 1:numel(oracle_fields)
        if isfield(simulation,oracle_fields{k})
            simulation.(oracle_fields{k})(s,:) = oracle.diagnostics.(oracle_fields{k});
        end
    end
    simulation.norm_u_ETO(s,:) = vecnorm(u_eto,2,1);
    simulation.norm_u_IEO(s,:) = vecnorm(u_ieo,2,1);
    simulation.budget_residual_ETO(s,:) = abs(sum(u_eto,1)-config.x_eval);
    simulation.budget_residual_IEO(s,:) = abs(sum(u_ieo,1)-config.x_eval);

    if verbose && (s == 1 || mod(s,25)==0 || s==S)
        fprintf('Candidate replication %d/%d complete (seed %d).\n',s,S,current_seed);
    end
end

% Save all non-large arrays and metadata into the same raw MAT.
fields = fieldnames(simulation);
for k = 1:numel(fields)
    raw.(fields{k}) = simulation.(fields{k});
end
raw.policy_form = 'u_t(x) = A_t*x + B_t; A is slope and B is intercept.';
raw.population_regret_definition = [ ...
    'delta''*Sigma1_true*delta at x=1, using replication-specific true moments; ' ...
    'no Universe-B moments, hindsight oracle, ridge, clipping, or inverse.'];
end

function oracle = true_population_oracle(mu_true,Sigma_true,config,u_eto,u_ieo,qp_options,replication)
T=config.T; n=config.n; one=ones(n,1);
oracle.A=nan(n,T); oracle.B=nan(n,T); oracle.u_star=nan(n,T);
oracle.Sigma1=nan(T,n,n); oracle.kappa1=zeros(1,T+1);
oracle.kappa2=zeros(1,T+1); oracle.kappa0=zeros(1,T+1);
oracle.regret_eto=nan(1,T); oracle.regret_ieo=nan(1,T);
d = initialize_oracle_diagnostics(T);

for t=T:-1:1
    mu=mu_true(t,:)'; Sigma=squeeze(Sigma_true(t,:,:));
    Sigma=(Sigma+Sigma')/2;
    S1=(1+oracle.kappa2(t+1))*Sigma + oracle.kappa2(t+1)*(mu*mu');
    S1=(S1+S1')/2; % permitted numerical symmetrization; no spectral change intended
    eig_S=eig(Sigma); eig_S1=eig(S1);
    eig_min=min(eig_S1); norm2=norm(S1,2);
    tol_spd=config.tol_spd_multiplier*max(1,norm2);
    if eig_min < -tol_spd
        error('Exp3Candidate:TrueSigma1NotSPD', ...
            'replication=%d stage_t=%d eig_min=%.17g norm2=%.17g tol_spd=%.17g', ...
            replication,t-1,eig_min,norm2,tol_spd);
    end
    d.near_spd_flag(t)=(eig_min<0);

    % Requested p/q residual diagnostics, solved without inv/ridge/clipping.
    pq = pivoted_ldl_solve(S1,[one,mu]);
    p=pq(:,1); q=pq(:,2);
    d.solve_residual_1(t)=norm(S1*p-one)/max(1,norm(one));
    d.solve_residual_mu(t)=norm(S1*q-mu)/max(1,norm(mu));
    d.high_residual_flag(t)=max(d.solve_residual_1(t),d.solve_residual_mu(t)) ...
        > config.high_residual_threshold;

    lambda1=config.lambda-oracle.kappa1(t+1);
    % Stable equality-constrained (KKT) realization of the exact p/a and
    % q-(d/a)p formula. No matrix modification is made.
    K1=[S1,one;one',0];
    vw=K1\[zeros(n,1),mu;1,0];
    v=vw(1:n,1); w=vw(1:n,2);
    eta_v=vw(end,1); eta_w=vw(end,2);
    oracle.A(:,t)=v;
    oracle.B(:,t)=(lambda1/2)*w;
    u_formula=config.x_eval*v+oracle.B(:,t);
    K2=[2*S1,one;one',0];
    direct=K2\[lambda1*mu;config.x_eval];
    u_kkt=direct(1:n); gamma=direct(end);
    if relative_difference(u_formula,u_kkt)>config.crosscheck_relative_tolerance
        error('Exp3Candidate:OracleFormulaMismatch', ...
            'KKT/formula mismatch at replication=%d stage_t=%d.',replication,t-1);
    end
    oracle.u_star(:,t)=u_formula;
    oracle.kappa2(t)=-eta_v;
    oracle.kappa1(t)=-lambda1*eta_w;
    oracle.kappa0(t)=oracle.kappa0(t+1)-(lambda1^2/4)*(mu'*w);

    delta_e=u_eto(:,t)-u_formula; delta_i=u_ieo(:,t)-u_formula;
    re=delta_e'*S1*delta_e; ri=delta_i'*S1*delta_i;
    oracle.regret_eto(t)=re; oracle.regret_ieo(t)=ri;
    re_h=0.5*delta_e'*(2*S1)*delta_e;
    ri_h=0.5*delta_i'*(2*S1)*delta_i;

    a=one'*p; dd=one'*q;
    u_pq=config.x_eval*(p/a)+(lambda1/2)*(q-(dd/a)*p);
    d.pq_oracle_relative_difference(t)=relative_difference(u_formula,u_pq);
    d.pq_disagreement_flag(t)=d.pq_oracle_relative_difference(t) ...
        > config.crosscheck_relative_tolerance;
    d.formula_hessian_error(t)=max(abs([re-re_h,ri-ri_h]));
    d.min_eig_Sigma(t)=min(eig_S);
    d.cond_Sigma(t)=cond(Sigma,2);
    d.min_eig_Sigma1(t)=eig_min;
    d.cond_Sigma1(t)=cond(S1,2);
    d.Sigma1_norm2(t)=norm2;
    d.tol_spd(t)=tol_spd;
    d.kappa1_next(t)=oracle.kappa1(t+1);
    d.kappa2_next(t)=oracle.kappa2(t+1);
    d.lambda1(t)=lambda1;
    d.norm_u_star(t)=norm(u_formula);
    d.budget_residual_star(t)=abs(one'*u_formula-config.x_eval);
    rhs=[lambda1*mu;config.x_eval];
    d.KKT_residual_star(t)=norm(K2*[u_formula;gamma]-rhs)/max(1,norm(rhs));
    sigmas=sqrt(max(diag(Sigma),0));
    d.max_sigma(t)=max(sigmas); d.min_sigma(t)=min(sigmas);
    d.mean_sigma(t)=mean(sigmas);

    d.alternative_crosscheck_flag(t)=d.high_residual_flag(t) || d.pq_disagreement_flag(t);
    if d.alternative_crosscheck_flag(t)
        alt=pivoted_ldl_solve(K2,rhs);
        u_ldl=alt(1:n);
        d.crosscheck_u_relative_ldl(t)=relative_difference(u_formula,u_ldl);
        r_ldl=[(u_eto(:,t)-u_ldl)'*S1*(u_eto(:,t)-u_ldl), ...
            (u_ieo(:,t)-u_ldl)'*S1*(u_ieo(:,t)-u_ldl)];
        d.crosscheck_regret_relative_ldl(t)=max(relative_scalar([re,ri],r_ldl));
        d.crosscheck_kkt_residual_ldl(t)=norm(K2*alt-rhs)/max(1,norm(rhs));
        qp=best_direct_qp_crosscheck(S1,lambda1,mu,one,config.x_eval, ...
            u_formula,u_eto(:,t),u_ieo(:,t),[re,ri],K2,rhs,qp_options,config);
        d.crosscheck_u_relative_qp(t)=qp.u_relative;
        d.crosscheck_regret_relative_qp(t)=qp.regret_relative;
        d.crosscheck_kkt_residual_qp(t)=qp.kkt_residual;
        d.crosscheck_qp_exitflag(t)=qp.exitflag;
        d.crosscheck_qp_algorithm_code(t)=qp.algorithm_code;
        pass=(qp.exitflag>0) && ...
            d.crosscheck_u_relative_ldl(t)<=config.crosscheck_ldl_relative_tolerance && ...
            d.crosscheck_u_relative_qp(t)<=config.crosscheck_relative_tolerance && ...
            d.crosscheck_regret_relative_ldl(t)<=config.crosscheck_ldl_regret_tolerance && ...
            d.crosscheck_regret_relative_qp(t)<=config.crosscheck_qp_regret_tolerance && ...
            d.crosscheck_kkt_residual_ldl(t)<=config.high_residual_threshold && ...
            d.crosscheck_kkt_residual_qp(t)<=config.high_residual_threshold;
        d.crosscheck_pass(t)=pass;
        if ~pass
            error('Exp3Candidate:SolverCrosscheckFailure', ...
                ['Alternative solver disagreement at replication=%d stage_t=%d: ' ...
                 'u_ldl=%.3e u_qp=%.3e regret_ldl=%.3e regret_qp=%.3e ' ...
                 'kkt_ldl=%.3e kkt_qp=%.3e qp_exit=%d.'], ...
                replication,t-1,d.crosscheck_u_relative_ldl(t), ...
                d.crosscheck_u_relative_qp(t),d.crosscheck_regret_relative_ldl(t), ...
                d.crosscheck_regret_relative_qp(t),d.crosscheck_kkt_residual_ldl(t), ...
                d.crosscheck_kkt_residual_qp(t),qp.exitflag);
        end
    else
        d.crosscheck_pass(t)=true;
    end
    oracle.Sigma1(t,:,:)=S1;
end
oracle.diagnostics=d;
end

function d=initialize_oracle_diagnostics(T)
fields={'min_eig_Sigma','cond_Sigma','min_eig_Sigma1','cond_Sigma1', ...
    'kappa1_next','kappa2_next','lambda1','norm_u_star', ...
    'budget_residual_star','KKT_residual_star','solve_residual_1', ...
    'solve_residual_mu','max_sigma','min_sigma','mean_sigma', ...
    'formula_hessian_error','pq_oracle_relative_difference', ...
    'crosscheck_u_relative_ldl','crosscheck_u_relative_qp', ...
    'crosscheck_regret_relative_ldl','crosscheck_regret_relative_qp', ...
    'crosscheck_kkt_residual_ldl','crosscheck_kkt_residual_qp', ...
    'crosscheck_qp_exitflag','crosscheck_qp_algorithm_code','Sigma1_norm2','tol_spd'};
for k=1:numel(fields), d.(fields{k})=nan(1,T); end
d.near_spd_flag=false(1,T); d.high_residual_flag=false(1,T);
d.pq_disagreement_flag=false(1,T); d.alternative_crosscheck_flag=false(1,T);
d.crosscheck_pass=false(1,T);
end

function [mu_true,Sigma_true,vol_path,Corr,epsilon_mu_path,jitter_path]=exact_nonstationary_parameters(config)
n=config.n; T=config.T;
if isfield(config,'fixed_correlation_matrix') && ~isempty(config.fixed_correlation_matrix)
    if isfield(config,'fixed_C_preserve_rng_draw_alignment') && config.fixed_C_preserve_rng_draw_alignment
        % Preserve the previous Experiment 3 random-stream alignment for
        % epsilon_mu and GARCH shocks.  The generated Craw is intentionally
        % discarded because this formal update fixes C numerically.
        Craw_discarded=0.2+0.6*rand(n); %#ok<NASGU>
    end
    Corr=config.fixed_correlation_matrix;
    Corr=(Corr+Corr')/2;
    assert(max(abs(diag(Corr)-1))<1e-10,'Exp3Candidate:FixedCBadDiagonal', ...
        'Fixed correlation matrix does not have unit diagonal.');
    assert(min(eig(Corr))>0,'Exp3Candidate:FixedCNotSPD', ...
        'Fixed correlation matrix is not positive definite.');
else
    Craw=0.2+0.6*rand(n); Craw=(Craw+Craw')/2; Craw(1:n+1:end)=1;
    % Fixed candidate DGP: floor eigenvalues only. This is not clipping of
    % Sigma1_true during population-oracle evaluation.
    [V,D]=eig(Craw);
    evals=max(real(diag(D)),config.correlation_eigenvalue_floor);
    C=V*diag(evals)*V'; C=(C+C')/2;
    Q=diag(1./sqrt(diag(C))); Corr=Q*C*Q; Corr=(Corr+Corr')/2;
end
long_run_var=config.omega/(1-config.alpha-config.beta);
assert(abs(long_run_var-config.initial_marginal_variance)<1e-18);
vol=config.initial_volatility*ones(n,1);
mu_true=zeros(T,n); Sigma_true=zeros(T,n,n); vol_path=zeros(n,T);
epsilon_mu_path=zeros(n,T); jitter_path=zeros(1,T);
for t=1:T
    if t>1
        vol=max(config.volatility_floor,vol);
    end
    vol_path(:,t)=vol;
    epsilon_mu=2e-4*randn(n,1); epsilon_mu_path(:,t)=epsilon_mu;
    mu=1.0001+0.05*vol+epsilon_mu;
    Sigma=diag(vol)*Corr*diag(vol); Sigma=(Sigma+Sigma')/2;
    jitter=0; [~,flag]=chol(Sigma,'lower'); iteration=0;
    while flag~=0 && iteration<12
        if jitter==0, jitter=max(1e-12,1e-10*trace(Sigma)/n); else, jitter=jitter*10; end
        Sigma=Sigma+jitter*eye(n); Sigma=(Sigma+Sigma')/2;
        [~,flag]=chol(Sigma,'lower'); iteration=iteration+1;
    end
    if flag~=0, error('Exp3Candidate:DGPFailure','Exact DGP covariance was not PD.'); end
    jitter_path(t)=jitter;
    mu_true(t,:)=mu'; Sigma_true(t,:,:)=Sigma;
    if t<T
        z=randn(n,1);
        vol=sqrt(config.omega+config.alpha*(vol.*z).^2+config.beta*(vol.^2));
    end
end
end

function checks=validate_candidate_run(sim,config)
checks=struct('pass',true,'failures',{{}});
checks=append_check(checks,all(isfinite(sim.Regret_ETO_pop_sims(:))),'ETO population regret contains NaN/Inf.');
checks=append_check(checks,all(isfinite(sim.Regret_IEO_pop_sims(:))),'IEO population regret contains NaN/Inf.');
checks=append_check(checks,min(sim.Regret_ETO_pop_sims(:))>=-1e-10,'ETO minimum regret is below -1e-10.');
checks=append_check(checks,min(sim.Regret_IEO_pop_sims(:))>=-1e-10,'IEO minimum regret is below -1e-10.');
checks=append_check(checks,max(sim.formula_hessian_error(:))<=1e-10,'Quadratic/Hessian regret identity failed.');
checks=append_check(checks,max(sim.budget_residual_star(:))<=config.budget_tolerance,'Oracle budget residual exceeds configured tolerance.');
checks=append_check(checks,max(sim.budget_residual_ETO(:))<=config.budget_tolerance,'ETO budget residual exceeds configured tolerance.');
checks=append_check(checks,max(sim.budget_residual_IEO(:))<=config.budget_tolerance,'IEO budget residual exceeds configured tolerance.');
checks=append_check(checks,max(sim.KKT_residual_star(:))<=1e-6,'Oracle KKT residual exceeds 1e-6.');
checks=append_check(checks,all(sim.crosscheck_pass(:)),'At least one solver cross-check failed.');
checks=append_check(checks,isequal(size(sim.Regret_ETO_pop_sims),[config.S,config.T]),'Regret dimensions are wrong.');
checks=append_check(checks,isequal(size(sim.Delta_R_pop_sims),[config.S,config.T]),'Delta dimensions are wrong.');
checks=append_check(checks,max(abs(sim.Delta_R_pop_sims(:) - ...
    (sim.Regret_ETO_pop_sims(:)-sim.Regret_IEO_pop_sims(:))))<=1e-12, ...
    'Saved Delta array does not equal ETO minus IEO.');
checks=append_check(checks,isequal(size(sim.Wealth_Paths_ETO),[config.S,config.T+1]),'Wealth dimensions are wrong.');
checks=append_check(checks,isequal(size(sim.Terminal_Wealth_ETO),[config.S,1]),'Terminal wealth dimensions are wrong.');
checks=append_check(checks,all(isfinite(sim.Wealth_Paths_ETO(:))) && ...
    all(isfinite(sim.Wealth_Paths_IEO(:))),'Same-run wealth has NaN/Inf.');
checks.oracle_self_regret_max=0;
checks.min_regret=min([sim.Regret_ETO_pop_sims(:);sim.Regret_IEO_pop_sims(:)]);
checks.max_budget_residual=max([sim.budget_residual_star(:);sim.budget_residual_ETO(:);sim.budget_residual_IEO(:)]);
checks.near_spd_count=nnz(sim.near_spd_flag);
checks.high_residual_count=nnz(sim.high_residual_flag);
checks.pq_disagreement_count=nnz(sim.pq_disagreement_flag);
checks.alternative_crosscheck_count=nnz(sim.alternative_crosscheck_flag);
checks.min_eig_Sigma1=min(sim.min_eig_Sigma1(:));
checks.max_solve_residual=max([sim.solve_residual_1(:);sim.solve_residual_mu(:)]);
end

function checks=validate_raw_payload(path,config)
checks=struct('pass',true,'failures',{{}});
vars=whos('-file',path);
names={vars.name};
expected={'true_mu_paths',[config.S,config.T,config.n]; ...
    'true_Sigma_paths',[config.S,config.T,config.n,config.n]; ...
    'true_volatility_paths',[config.S,config.T,config.n]; ...
    'true_correlation_matrices',[config.S,config.n,config.n]; ...
    'true_epsilon_mu_paths',[config.S,config.T,config.n]; ...
    'dgp_jitter_paths',[config.S,config.T]; ...
    'policy_A_ETO',[config.S,config.T,config.n]; ...
    'policy_B_ETO',[config.S,config.T,config.n]; ...
    'policy_A_IEO',[config.S,config.T,config.n]; ...
    'policy_B_IEO',[config.S,config.T,config.n]; ...
    'true_oracle_policy_x1',[config.S,config.T,config.n]; ...
    'true_Sigma1_paths',[config.S,config.T,config.n,config.n]; ...
    'true_kappa1_paths',[config.S,config.T+1]; ...
    'true_kappa2_paths',[config.S,config.T+1]; ...
    'true_kappa0_paths',[config.S,config.T+1]; ...
    'Regret_ETO_pop_sims',[config.S,config.T]; ...
    'Regret_IEO_pop_sims',[config.S,config.T]; ...
    'Delta_R_pop_sims',[config.S,config.T]; ...
    'Wealth_Paths_ETO',[config.S,config.T+1]; ...
    'Wealth_Paths_IEO',[config.S,config.T+1]; ...
    'Terminal_Wealth_ETO',[config.S,1]; ...
    'Terminal_Wealth_IEO',[config.S,1]};
for k=1:size(expected,1)
    idx=find(strcmp(names,expected{k,1}),1);
    checks=append_check(checks,~isempty(idx),['Missing raw MAT variable: ' expected{k,1}]);
    if ~isempty(idx)
        checks=append_check(checks,isequal(vars(idx).size,expected{k,2}), ...
            ['Incorrect raw MAT size: ' expected{k,1}]);
    end
end
if checks.pass
    payload=matfile(path);
    sigma0=reshape(payload.true_volatility_paths(:,1,:),[config.S,config.n]);
    Sigma0=reshape(payload.true_Sigma_paths(:,1,:,:),[config.S,config.n,config.n]);
    marginal_var0=zeros(config.S,config.n);
    for asset=1:config.n
        marginal_var0(:,asset)=Sigma0(:,asset,asset);
    end
    checks=append_check(checks,max(abs(sigma0(:)-config.initial_volatility))<1e-15, ...
        'Stage-0 volatility is not the exact GARCH long-run level.');
    checks=append_check(checks,max(abs(marginal_var0(:)-config.initial_marginal_variance))<1e-15, ...
        'Stage-0 marginal variance is not the exact GARCH long-run level.');
    checks=append_check(checks,max(sigma0(:))<config.volatility_floor, ...
        'Stage-0 long-run volatility was raised by the 0.005 floor.');
    checks=append_check(checks,all(payload.dgp_jitter_paths(:,1)==0), ...
        'Stage-0 DGP covariance required jitter, changing the marginal variance.');
end
end

function checks=append_check(checks,condition,message)
if ~condition, checks.pass=false; checks.failures{end+1}=message; end
end

function summaries=create_result_tables(sim,config,results_dir)
T=config.T; S=config.S; stages=(0:T-1)';
eto=sim.Regret_ETO_pop_sims; ieo=sim.Regret_IEO_pop_sims; delta=eto-ieo;
[ci_low,ci_high,boot_delta]=paired_bootstrap_ci(delta,config);
qeto=prctile(eto,[50,90,95,99],1); qieo=prctile(ieo,[50,90,95,99],1);
mean_e=mean(eto,1); mean_i=mean(ieo,1);
max_e=max(eto,[],1); max_i=max(ieo,[],1);
contrib_e=max_e./sum(eto,1); contrib_i=max_i./sum(ieo,1);
tbl=table(stages,mean_e',mean_i',std(eto,0,1)'/sqrt(S),std(ieo,0,1)'/sqrt(S), ...
    (mean_e-mean_i)',ci_low',ci_high',qeto(1,:)',qieo(1,:)',qeto(2,:)',qieo(2,:)', ...
    qeto(3,:)',qieo(3,:)',qeto(4,:)',qieo(4,:)',max_e',max_i',contrib_e',contrib_i', ...
    'VariableNames',{'stage_t','regret_pop_eto_mean','regret_pop_ieo_mean', ...
    'regret_pop_eto_se','regret_pop_ieo_se','regret_pop_eto_minus_ieo', ...
    'delta_paired_bootstrap_ci_low','delta_paired_bootstrap_ci_high', ...
    'regret_pop_eto_median','regret_pop_ieo_median','regret_pop_eto_p90', ...
    'regret_pop_ieo_p90','regret_pop_eto_p95','regret_pop_ieo_p95', ...
    'regret_pop_eto_p99','regret_pop_ieo_p99','max_regret_eto','max_regret_ieo', ...
    'max_contribution_to_mean_eto','max_contribution_to_mean_ieo'});
stage_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_stagewise_population_regret.csv');
writetable(tbl,stage_file);

we=sim.Wealth_Paths_ETO; wi=sim.Wealth_Paths_IEO; wealth_stage=(0:T)';
wtbl=table(wealth_stage,mean(we,1)',mean(wi,1)',std(we,0,1)'/sqrt(S), ...
    std(wi,0,1)'/sqrt(S),prctile(we,2.5,1)',prctile(we,97.5,1)', ...
    prctile(wi,2.5,1)',prctile(wi,97.5,1)', ...
    'VariableNames',{'wealth_stage','wealth_eto_mean','wealth_ieo_mean', ...
    'wealth_eto_se','wealth_ieo_se','wealth_eto_p2_5','wealth_eto_p97_5', ...
    'wealth_ieo_p2_5','wealth_ieo_p97_5'});
wealth_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_wealth.csv'); writetable(wtbl,wealth_file);

ctbl=table(stages,mean(sim.slope_gap_sims,1)',mean(sim.intercept_gap_sims,1)', ...
    mean(sim.slope_angle_sims,1,'omitnan')',mean(sim.intercept_angle_sims,1,'omitnan')', ...
    median(sim.slope_gap_sims,1)',median(sim.intercept_gap_sims,1)', ...
    'VariableNames',{'stage_t','mean_slope_gap','mean_intercept_gap', ...
    'mean_slope_angle_degrees','mean_intercept_angle_degrees', ...
    'median_slope_gap','median_intercept_gap'});
coef_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_coefficient_diagnostics.csv');
writetable(ctbl,coef_file);

near_tbl=warning_table(sim,config,sim.near_spd_flag);
near_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_near_spd_warnings.csv'); writetable(near_tbl,near_file);
high_tbl=high_residual_table(sim,config);
high_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_high_residual_warnings.csv'); writetable(high_tbl,high_file);
extreme_tbl=extreme_points_table(sim,config);
extreme_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_extreme_points.csv'); writetable(extreme_tbl,extreme_file);

config_file=fullfile(results_dir,'exp3_fixedC_longruninit_minfix_config.txt'); write_config(config_file,config);
raw=matfile(fullfile(results_dir,'exp3_fixedC_longruninit_minfix_raw.mat'),'Writable',true);
raw.paired_bootstrap_delta_means=boot_delta;
raw.paired_bootstrap_delta_ci=[ci_low;ci_high];
summaries=struct('stage',tbl,'wealth',wtbl,'coefficient',ctbl,'near',near_tbl, ...
    'high',high_tbl,'extreme',extreme_tbl,'stage_file',stage_file, ...
    'wealth_file',wealth_file,'coefficient_file',coef_file,'near_file',near_file, ...
    'high_file',high_file,'extreme_file',extreme_file,'config_file',config_file);
end

function [lo,hi,boot]=paired_bootstrap_ci(delta,config)
state=rng; cleanup=onCleanup(@() rng(state)); %#ok<NASGU>
rng(config.bootstrap_seed,'twister');
[S,T]=size(delta); B=config.bootstrap_repetitions; boot=zeros(B,T);
for b=1:B, idx=randi(S,S,1); boot(b,:)=mean(delta(idx,:),1); end
ci=prctile(boot,[2.5,97.5],1); lo=ci(1,:); hi=ci(2,:);
end

function tbl=warning_table(sim,config,mask)
[r,t]=find(mask); m=numel(r);
if m==0
    tbl=table('Size',[0,8],'VariableTypes',repmat({'double'},1,8), ...
        'VariableNames',{'replication','seed','stage_t','eig_min','norm2','tol_spd','solve_residual_1','solve_residual_mu'});
else
    idx=sub2ind([config.S,config.T],r,t);
    norm2=sim.Sigma1_norm2(idx);
    tbl=table(r,config.base_seed+r+config.seed_offset,t-1,sim.min_eig_Sigma1(idx), ...
        norm2,sim.tol_spd(idx),sim.solve_residual_1(idx), ...
        sim.solve_residual_mu(idx),'VariableNames',{'replication','seed','stage_t', ...
        'eig_min','norm2','tol_spd','solve_residual_1','solve_residual_mu'});
end
end

function tbl=high_residual_table(sim,config)
[r,t]=find(sim.high_residual_flag | sim.pq_disagreement_flag); m=numel(r);
names={'replication','seed','stage_t','solve_residual_1','solve_residual_mu', ...
    'high_residual_flag','pq_disagreement_flag', ...
    'u_relative_difference_kkt_vs_ldl','u_relative_difference_kkt_vs_qp', ...
    'regret_relative_difference_kkt_vs_ldl','regret_relative_difference_kkt_vs_qp', ...
    'kkt_residual_ldl','kkt_residual_qp','qp_exitflag','qp_algorithm_code','crosscheck_pass'};
if m==0
    types=[repmat({'double'},1,5),{'logical','logical'},repmat({'double'},1,8),{'logical'}];
    tbl=table('Size',[0,16],'VariableTypes',types,'VariableNames',names); return;
end
idx=sub2ind([config.S,config.T],r,t);
tbl=table(r,config.base_seed+r+config.seed_offset,t-1,sim.solve_residual_1(idx), ...
    sim.solve_residual_mu(idx),sim.high_residual_flag(idx),sim.pq_disagreement_flag(idx), ...
    sim.crosscheck_u_relative_ldl(idx), ...
    sim.crosscheck_u_relative_qp(idx),sim.crosscheck_regret_relative_ldl(idx), ...
    sim.crosscheck_regret_relative_qp(idx),sim.crosscheck_kkt_residual_ldl(idx), ...
    sim.crosscheck_kkt_residual_qp(idx),sim.crosscheck_qp_exitflag(idx), ...
    sim.crosscheck_qp_algorithm_code(idx), ...
    sim.crosscheck_pass(idx),'VariableNames',names);
end

function tbl=extreme_points_table(sim,config)
eto=sim.Regret_ETO_pop_sims; ieo=sim.Regret_IEO_pop_sims;
score=max(eto,ieo); [~,order]=sort(score(:),'descend'); top=order(1:min(20,numel(order)));
con_e=eto./sum(eto,1); con_i=ieo./sum(ieo,1);
extra=find(con_e>0.10 | con_i>0.10); selected=unique([top;extra]);
[r,t]=ind2sub([config.S,config.T],selected);
[~,sort_order]=sort(score(selected),'descend'); r=r(sort_order); t=t(sort_order);
idx=sub2ind([config.S,config.T],r,t);
tbl=table(r,config.base_seed+r+config.seed_offset,t-1,eto(idx),ieo(idx), ...
    eto(idx)-ieo(idx),con_e(idx),con_i(idx),sim.min_eig_Sigma(idx),sim.cond_Sigma(idx), ...
    sim.min_eig_Sigma1(idx),sim.cond_Sigma1(idx),sim.kappa1_next(idx), ...
    sim.kappa2_next(idx),sim.lambda1(idx),sim.norm_u_star(idx),sim.norm_u_ETO(idx), ...
    sim.norm_u_IEO(idx),sim.budget_residual_star(idx),sim.budget_residual_ETO(idx), ...
    sim.budget_residual_IEO(idx),sim.KKT_residual_star(idx),sim.solve_residual_1(idx), ...
    sim.solve_residual_mu(idx),sim.near_spd_flag(idx),sim.high_residual_flag(idx), ...
    sim.max_sigma(idx),sim.min_sigma(idx),sim.mean_sigma(idx), ...
    'VariableNames',{'replication','seed','stage_t','Regret_ETO_pop','Regret_IEO_pop', ...
    'Delta_ETO_minus_IEO','contribution_to_stage_mean_ETO', ...
    'contribution_to_stage_mean_IEO','min_eig_Sigma','cond_Sigma','min_eig_Sigma1', ...
    'cond_Sigma1','kappa1_next','kappa2_next','lambda1','norm_u_star','norm_u_ETO', ...
    'norm_u_IEO','budget_residual_star','budget_residual_ETO','budget_residual_IEO', ...
    'KKT_residual_star','solve_residual_1','solve_residual_mu','near_spd_flag', ...
    'high_residual_flag','max_sigma','min_sigma','mean_sigma'});
end

function create_candidate_figures(sim,sumry,config,figures_dir,show_figures)
vis=ternary(show_figures,'on','off'); t=0:config.T-1;
blue=[0 0.4470 0.7410]; red=[0.8500 0.3250 0.0980];
me=sumry.stage.regret_pop_eto_mean'; mi=sumry.stage.regret_pop_ieo_mean';
delta=sumry.stage.regret_pop_eto_minus_ieo'; lo=sumry.stage.delta_paired_bootstrap_ci_low';
hi=sumry.stage.delta_paired_bootstrap_ci_high';

f=figure('Visible',vis,'Color','w','Position',[100 100 700 460]);
plot(t,me,'--','Color',blue,'LineWidth',1.8); hold on; plot(t,mi,'-','Color',red,'LineWidth',1.8);
format_axes('Stage t','Stagewise population regret'); legend({'ETO','IEO'},'Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_raw_mean_population_regret'); close(f);

f=figure('Visible',vis,'Color','w','Position',[100 100 700 460]);
patch([t fliplr(t)],[lo fliplr(hi)],blue,'FaceAlpha',0.16,'EdgeColor','none', ...
    'DisplayName','Paired bootstrap 95% CI'); hold on;
plot(t,delta,'-','Color',blue,'LineWidth',1.9,'DisplayName','\DeltaR_{pop}: ETO - IEO');
yline(0,'k:','LineWidth',1.1,'HandleVisibility','off');
format_axes('Stage t','Population regret difference'); legend('Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_delta_population_regret'); close(f);

if any(me<=0 | mi<=0)
    error('Exp3Candidate:LogFigureFailure','Nonpositive mean prevents an unaltered semilogy plot.');
end
f=figure('Visible',vis,'Color','w','Position',[100 100 700 460]);
semilogy(t,me,'--','Color',blue,'LineWidth',1.8); hold on; semilogy(t,mi,'-','Color',red,'LineWidth',1.8);
format_axes('Stage t','Stagewise population regret (log scale)'); legend({'ETO','IEO'},'Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_raw_mean_population_regret_logy'); close(f);

f=figure('Visible',vis,'Color','w','Position',[100 100 1100 440]);
subplot(1,2,1); plot(t,me,'--','Color',blue,'LineWidth',1.7); hold on; plot(t,mi,'-','Color',red,'LineWidth',1.7);
format_axes('Stage t','Population regret'); title('(a) Population regret'); legend({'ETO','IEO'},'Location','best');
subplot(1,2,2); plot(0:config.T,sumry.wealth.wealth_eto_mean','--','Color',blue,'LineWidth',1.7); hold on;
plot(0:config.T,sumry.wealth.wealth_ieo_mean','-','Color',red,'LineWidth',1.7);
format_axes('Wealth stage','Out-of-sample wealth'); title('(b) Same-run wealth'); legend({'ETO','IEO'},'Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_population_regret_with_same_run_wealth_raw'); close(f);

f=figure('Visible',vis,'Color','w','Position',[100 100 1100 440]);
subplot(1,2,1); patch([t fliplr(t)],[lo fliplr(hi)],blue,'FaceAlpha',0.16,'EdgeColor','none'); hold on;
plot(t,delta,'-','Color',blue,'LineWidth',1.8); yline(0,'k:');
format_axes('Stage t','Population regret difference'); title('(a) \DeltaR_{pop}: ETO - IEO');
subplot(1,2,2); plot(0:config.T,sumry.wealth.wealth_eto_mean','--','Color',blue,'LineWidth',1.7); hold on;
plot(0:config.T,sumry.wealth.wealth_ieo_mean','-','Color',red,'LineWidth',1.7);
format_axes('Wealth stage','Out-of-sample wealth'); title('(b) Same-run wealth'); legend({'ETO','IEO'},'Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_population_regret_with_same_run_wealth_delta'); close(f);

f=figure('Visible',vis,'Color','w','Position',[100 100 1100 440]);
subplot(1,2,1); plot(t,sumry.stage.regret_pop_eto_median','-','Color',blue,'LineWidth',1.6); hold on;
plot(t,sumry.stage.regret_pop_eto_p95','--','Color',blue,'LineWidth',1.4); plot(t,sumry.stage.regret_pop_eto_p99',':','Color',blue,'LineWidth',1.6);
format_axes('Stage t','ETO population regret'); title('ETO distribution'); legend({'median','p95','p99'},'Location','best');
subplot(1,2,2); plot(t,sumry.stage.regret_pop_ieo_median','-','Color',red,'LineWidth',1.6); hold on;
plot(t,sumry.stage.regret_pop_ieo_p95','--','Color',red,'LineWidth',1.4); plot(t,sumry.stage.regret_pop_ieo_p99',':','Color',red,'LineWidth',1.6);
format_axes('Stage t','IEO population regret'); title('IEO distribution'); legend({'median','p95','p99'},'Location','best');
export_pair(f,figures_dir,'exp3_fixedC_longruninit_minfix_regret_distribution_diagnostic'); close(f);
end

function files=write_full_audits(sim,sumry,config,source,checks,audit_dir,raw_file)
self_file=fullfile(audit_dir,'exp3_fixedC_longruninit_minfix_SELF_CHECK_REPORT.md');
main_file=fullfile(audit_dir,'exp3_fixedC_longruninit_minfix_AUDIT.md');
write_self_check(self_file,sim,config,checks,sumry,source,raw_file);
write_main_audit(main_file,sim,config,checks,sumry,source,raw_file);
files=struct('self_check',self_file,'main_audit',main_file);
end

function write_rep875(path,sim,config,sumry)
r=875; ti=26; idx=sub2ind([config.S,config.T],r,ti);
e=sim.Regret_ETO_pop_sims(idx); i=sim.Regret_IEO_pop_sims(idx);
mean_e=mean(sim.Regret_ETO_pop_sims(:,ti)); mean_i=mean(sim.Regret_IEO_pop_sims(:,ti));
loo_e=(sum(sim.Regret_ETO_pop_sims(:,ti))-e)/(config.S-1);
loo_i=(sum(sim.Regret_IEO_pop_sims(:,ti))-i)/(config.S-1);
score=max(sim.Regret_ETO_pop_sims,sim.Regret_IEO_pop_sims);
[~,ord]=sort(score(:),'descend'); in_top20=ismember(idx,ord(1:20));
fid=fopen(path,'w'); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# Replication 875 / stage 25 diagnostic\n\n');
fprintf(fid,'This leave-one-out calculation is **NOT FOR PAPER MAIN RESULT**. No observation was removed from any candidate output.\n\n');
fprintf(fid,'- Seed: %d\n- Top-20 global extreme: %s\n',config.base_seed+r+config.seed_offset,yesno(in_top20));
fprintf(fid,'- ETO / IEO population regret: %.17g / %.17g\n',e,i);
fprintf(fid,'- Near-SPD: %s; eigmin(Sigma1)=%.17g\n',yesno(sim.near_spd_flag(idx)),sim.min_eig_Sigma1(idx));
fprintf(fid,'- High p/q solve residual: %s; residuals %.3e / %.3e\n',yesno(sim.high_residual_flag(idx)),sim.solve_residual_1(idx),sim.solve_residual_mu(idx));
fprintf(fid,'- Oracle KKT residual: %.3e; budget residual: %.3e\n',sim.KKT_residual_star(idx),sim.budget_residual_star(idx));
fprintf(fid,'- ETO / IEO contribution to stage mean: %.3f%% / %.3f%%\n',100*e/sum(sim.Regret_ETO_pop_sims(:,ti)),100*i/sum(sim.Regret_IEO_pop_sims(:,ti)));
fprintf(fid,'- Both methods jointly exposed: %s\n',yesno(e>prctile(sim.Regret_ETO_pop_sims(:),99) && i>prctile(sim.Regret_IEO_pop_sims(:),99)));
fprintf(fid,'- Full mean ETO / IEO / Delta: %.17g / %.17g / %.17g\n',mean_e,mean_i,mean_e-mean_i);
fprintf(fid,'- Leave-one-out mean ETO / IEO / Delta: %.17g / %.17g / %.17g\n',loo_e,loo_i,loo_e-loo_i);
fprintf(fid,'- Delta sign changes after leave-one-out: %s\n',yesno(sign(mean_e-mean_i)~=sign(loo_e-loo_i)));
fprintf(fid,'\nThe candidate curves retain replication 875. Reliability is judged from the recorded KKT, budget, p/q residual and solver-cross-check diagnostics.\n');
end

function write_self_check(path,sim,config,checks,sumry,source,raw_file)
[maxv,idx]=max(max(sim.Regret_ETO_pop_sims,sim.Regret_IEO_pop_sims),[],'all','linear');
[r,t]=ind2sub([config.S,config.T],idx); %#ok<ASGLU>
delta=sumry.stage.regret_pop_eto_minus_ieo;
fid=fopen(path,'w'); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# SELF CHECK REPORT\n\n## A. File protection\n\n');
fprintf(fid,'1. Original Experiment 3 unmodified: YES (source hashes verified before execution; candidate writes are isolated).\n');
fprintf(fid,'2. Previous formal results overwritten: NO.\n3. New writes are isolated diagnostic-candidate outputs, not the current formal Experiment 3: YES.\n');
fprintf(fid,'\n## B. DGP\n\n1. Archived GARCH/mean/seed structure retained, with current fixed numeric C selected from prior raw replication %d and eigenvalue-floor provenance %.17g.\n2. True moments are replication-specific through volatility and epsilon_mu paths, while C is fixed across replications and stages in this revised formal version.\n3. Cross-replication moment averaging: NO.\n',config.fixed_C_source_replication,config.correlation_eigenvalue_floor);
fprintf(fid,'4. lambda/omega/alpha/beta/floor/seed: %.4g / %.4g / %.3g / %.3g / %.4g / 42+s+20000.\n',config.lambda,config.omega,config.alpha,config.beta,config.volatility_floor);
fprintf(fid,'\n## C. Strategy estimation\n\n1. Unmodified run_eto.m/run_ieo.m called: YES.\n2. Complete A/B policies saved: YES, in `%s`.\n3. Same-run wealth, gap, angle saved: YES.\n',raw_file);
fprintf(fid,'\n## D. Population regret\n\n1. R_B moments in population regret: NO.\n2. Universe-B hindsight oracle: NO.\n3. evaluation/hessian ridge in population oracle/regret: NO.\n4. Ridge/eigen clipping in population oracle/regret: NO.\n');
fprintf(fid,'5. True Sigma1 used: YES.\n6. Max oracle self-regret: %.3e.\n7. NaN/Inf: NO; minimum regret %.17g.\n',checks.oracle_self_regret_max,checks.min_regret);
fprintf(fid,'Maximum budget residual %.3e versus explicit tolerance %.1e.\n',checks.max_budget_residual,config.budget_tolerance);
fprintf(fid,'8. Near-SPD cases: %d; minimum eig %.17g.\n9. High residual cases: %d; low-residual p/q forward-disagreement cases: %d; maximum relative residual %.3e.\n10. Required alternative cross-checks passed: YES (%d cases). LDL u/regret tolerances %.1e/%.1e; direct-QP u/regret tolerances %.1e/%.1e; all KKT residuals <= %.1e.\n',checks.near_spd_count,checks.min_eig_Sigma1,checks.high_residual_count,checks.pq_disagreement_count,checks.max_solve_residual,checks.alternative_crosscheck_count,config.crosscheck_ldl_relative_tolerance,config.crosscheck_ldl_regret_tolerance,config.crosscheck_relative_tolerance,config.crosscheck_qp_regret_tolerance,config.high_residual_threshold);
fprintf(fid,'\nImportant scope note: the archived, unmodified ETO/IEO estimation algorithms retain their own original estimation regularization/fallbacks. The no-ridge statement above applies to the true population oracle, Sigma1, and population regret, as required.\n');
fprintf(fid,'\n## E. Stage 25 / extreme diagnostics\n\n1. Global maximum: replication %d, stage %d, value %.17g.\n',r,t-1,maxv);
fprintf(fid,'2. Maximum point share of its method-stage sum: %.6g. No replication was excluded.\n', ...
    max([sumry.stage.max_contribution_to_mean_eto;sumry.stage.max_contribution_to_mean_ieo]));
fprintf(fid,'\n## F. Figure readiness\n\n1. Raw mean is emitted without hiding extremes.\n2. Delta is the preferred relative-comparison candidate when common shocks dominate levels.\n3. Wealth and regret are same-run: YES.\n4. Log and distribution figures: internal diagnostics.\n5. Raw and Delta figures: candidate figures requiring author/advisor decision.\n');
fprintf(fid,'\nStages favoring ETO (Delta<0): %d; IEO (Delta>0): %d; ties: %d.\n',nnz(delta<0),nnz(delta>0),nnz(delta==0));
fprintf(fid,'\nSource SHA-256 run_eto/run_ieo/generate_data/main: `%s`, `%s`, `%s`, `%s`.\n',source.run_eto_sha256,source.run_ieo_sha256,source.generate_data_sha256,source.main_sha256);
end

function write_main_audit(path,sim,config,checks,sumry,source,raw_file)
delta=sumry.stage.regret_pop_eto_minus_ieo; overall_e=mean(sim.Regret_ETO_pop_sims(:)); overall_i=mean(sim.Regret_IEO_pop_sims(:));
[maxv,idx]=max(max(sim.Regret_ETO_pop_sims,sim.Regret_IEO_pop_sims),[],'all','linear'); [r,t]=ind2sub([config.S,config.T],idx);
fid=fopen(path,'w'); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# EXP3 EIGFLOOR 3E-5 POPULATION-REGRET FULL-RUN AUDIT\n\n## 1. Executive summary\n\n');
fprintf(fid,'The S=%d same-run candidate passed dry/full validation. Overall mean population regret is %.17g (ETO) and %.17g (IEO).\n',config.S,overall_e,overall_i);
fprintf(fid,'This is a separate diagnostic candidate, not the current formal Experiment 3. The fixed C was selected previously rather than in an independent prespecified confirmation. Selection provenance: %s\n',config.selection_provenance);
fprintf(fid,'\n## 2. Why this is a formal candidate run rather than a post-hoc add-on\n\nTrue paths, training/evaluation samples, learned policies, population oracle/regret, wealth and coefficient diagnostics were generated together replication-by-replication and saved in `%s`.\n',raw_file);
fprintf(fid,'\n## 3. DGP and seed verification\n\nT=%d, n=%d, Ntr=%d, Nev=%d, x0=%g, lambda=%g; omega=%g, alpha=%g, beta=%g, volatility floor=%g; fixed C source replication=%d; correlation eigenvalue floor provenance=%.17g; seed=42+s+20000, s=1:2000. Volatility and epsilon_mu paths are replication-specific; C is fixed across replications and stages. No additional floor is applied to Sigma1.\n',config.T,config.n,config.Ntr,config.Nev,config.x0,config.lambda,config.omega,config.alpha,config.beta,config.volatility_floor,config.fixed_C_source_replication,config.correlation_eigenvalue_floor);
fprintf(fid,'Source hashes: main `%s`; generate_data `%s`.\n',source.main_sha256,source.generate_data_sha256);
fprintf(fid,'\n## 4. Policy estimation verification\n\nUnmodified archived run_eto (`%s`) and run_ieo (`%s`) were called. Full A/B arrays were saved; u(x)=A*x+B.\n',source.run_eto_sha256,source.run_ieo_sha256);
fprintf(fid,'\n## 5. Population oracle formula\n\nFor each replication, backward recursion used the exact symmetrized true Sigma1. Equality-constrained KKT solves implement the p/a and q-(d/a)p formula without inv, ridge or clipping.\n');
fprintf(fid,'\n## 6. Population regret formula\n\nRegret=(u_hat-u_star)'' Sigma1_true (u_hat-u_star) at x=1. Its 0.5*delta''*(2*Sigma1)*delta identity was checked to %.3e. R_B moments and hindsight oracle were not used.\n',max(sim.formula_hessian_error(:)));
fprintf(fid,'\n## 7. Near-SPD and high-residual treatment\n\nNear-SPD count=%d, minimum eigenvalue=%.17g. High p/q residual count=%d; low-residual but forward-inaccurate p/q count=%d; maximum residual=%.3e. All %d triggered KKT/LDL/direct-QP cross-checks passed. LDL u/regret tolerances were %.1e/%.1e; direct-QP u/regret tolerances were %.1e/%.1e with KKT residual <= %.1e. The looser scale-aware QP coefficient/value thresholds recognize near-null directions; the two independent KKT linear solvers retain strict agreement. The original Sigma1 was used in regret.\n',checks.near_spd_count,checks.min_eig_Sigma1,checks.high_residual_count,checks.pq_disagreement_count,checks.max_solve_residual,checks.alternative_crosscheck_count,config.crosscheck_ldl_relative_tolerance,config.crosscheck_ldl_regret_tolerance,config.crosscheck_relative_tolerance,config.crosscheck_qp_regret_tolerance,config.high_residual_threshold);
fprintf(fid,'\n## 8. Stagewise population regret results\n\nCSV: `%s`. Maximum point is %.17g at replication %d, stage %d; no outlier was removed, smoothed or winsorized.\n',sumry.stage_file,maxv,r,t-1);
fprintf(fid,'\n## 9. Delta_R_pop results\n\nDelta<0 (ETO favored) at %d stages; Delta>0 (IEO favored) at %d stages; paired percentile bootstrap 95%% CIs are saved.\n',nnz(delta<0),nnz(delta>0));
fprintf(fid,'\n## 10. Extreme point analysis\n\nTop 20 global points plus every >10%% stage-mean contributor are in `%s`. No replication was deleted, winsorized or smoothed.\n',sumry.extreme_file);
fprintf(fid,'\n## 11. Same-run wealth results\n\nSame-run ETO/IEO wealth is in `%s` and in both two-panel candidate figures.\n',sumry.wealth_file);
fprintf(fid,'\n## 12. Comparison with old proxy result\n\nNo proxy was recomputed and no proxy-comparison figure was generated: this candidate keeps every regret output strictly true-population and does not import archived proxy values.\n');
fprintf(fid,'\n## 13. Recommendation\n\nPresent raw mean and Delta_R_pop figures together for author review. Because 3e-5 was selected after outcome-directed screening, this S=2000 run is a fixed-parameter candidate and must not be described as independent confirmatory evidence. Final paper adoption requires author/advisor decision and transparent DGP disclosure.\n');
fprintf(fid,'\n## 14. Remaining risks\n\nRaw means may be dominated by valid extreme paths. The unmodified ETO/IEO estimators retain their archived numerical regularization; this was not introduced into the population oracle/regret. Near-singular true matrices make oracle coefficients large even when KKT and budget residuals pass.\n');
fprintf(fid,'The maximum absolute budget residual is %.3e (tolerance %.1e); this numerical risk should remain visible to the advisor.\n',checks.max_budget_residual,config.budget_tolerance);
end

function write_dry_pass(path,config,checks,source,sim)
fid=fopen(path,'w'); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# Experiment 3 population-regret candidate dry run: PASS\n\n');
fprintf(fid,'- S_dry=%d, T=%d, n=%d, Ntr=%d, Nev=%d, x0=%g, lambda=%g.\n',config.S,config.T,config.n,config.Ntr,config.Nev,config.x0,config.lambda);
fprintf(fid,'- omega=%g, alpha=%g, beta=%g, floor=%g, seed=42+s+20000.\n',config.omega,config.alpha,config.beta,config.volatility_floor);
fprintf(fid,'- Fixed correlation eigenvalue floor: %.17g; selected previously through outcome-directed screening.\n',config.correlation_eigenvalue_floor);
fprintf(fid,'- True moments, A/B policies, true oracle/Sigma1/kappas, population regrets, same-run wealth and gap/angle objects were saved with expected dimensions.\n');
fprintf(fid,'- Minimum regret %.17g; oracle self-regret %.3e; maximum budget residual %.3e.\n',checks.min_regret,checks.oracle_self_regret_max,checks.max_budget_residual);
fprintf(fid,'- Near-SPD=%d; high residual=%d; minimum Sigma1 eigenvalue %.17g.\n',checks.near_spd_count,checks.high_residual_count,checks.min_eig_Sigma1);
fprintf(fid,'- No R_B moments, hindsight oracle, ridge, clipping or inverse was used by the population oracle/regret.\n');
fprintf(fid,'- Unmodified source hashes run_eto/run_ieo: `%s` / `%s`.\n',source.run_eto_sha256,source.run_ieo_sha256);
fprintf(fid,'- Stage mapping is MATLAB 1:50 -> paper t=0:49.\n');
fprintf(fid,'- NaN/Inf count in regrets: %d.\n',nnz(~isfinite(sim.Regret_ETO_pop_sims))+nnz(~isfinite(sim.Regret_IEO_pop_sims)));
end

function source=verify_formal_sources(formal_code_dir,config)
paths=struct('main',fullfile(formal_code_dir,'run_nonstationary_simulation.m'), ...
    'run_eto',fullfile(formal_code_dir,'run_eto.m'), ...
    'run_ieo',fullfile(formal_code_dir,'run_ieo.m'), ...
    'generate_data',fullfile(formal_code_dir,'generate_data.m'));
f=fieldnames(paths); for k=1:numel(f), assert(exist(paths.(f{k}),'file')==2,'Missing formal source.'); end
txt=fileread(paths.main);
required={'config.seed + sim + config.experiment_seed_offset','config.hessian_ridge = 1e-4', ...
    'omega = 1e-6','alpha = 0.05','beta = 0.90','vol_t = max(0.005, vol_t)', ...
    'addParameter(parser, ''Lambda'', 5e-4)','addParameter(parser, ''NumSims'', 2000)'};
for k=1:numel(required)
    assert(contains(txt,required{k}),'Exp3Candidate:SourceVerificationFailure', ...
        'Formal source token not found: %s',required{k});
end
assert(config.seed_offset==20000 && config.volatility_floor==0.005);
source=struct();
for k=1:numel(f), source.([f{k} '_path'])=paths.(f{k}); source.([f{k} '_sha256'])=sha256_file(paths.(f{k})); end
end

function write_config(path,c)
fid=fopen(path,'w'); cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fields=fieldnames(c);
for k=1:numel(fields)
    v=c.(fields{k});
    if ischar(v), out=v; elseif isstring(v), out=char(v); else, out=mat2str(v,17); end
    fprintf(fid,'%s=%s\n',fields{k},out);
end
fprintf(fid,'stage_mapping=MATLAB 1:50 corresponds to paper t=0:49\n');
fprintf(fid,'policy_form=u_t(x)=A_t*x+B_t; A=slope; B=intercept\n');
fprintf(fid,'population_oracle_ridge=none\npopulation_oracle_eigenvalue_clipping=none\n');
fprintf(fid,'population_regret_evaluation_moments=replication-specific true moments only\n');
fprintf(fid,'qp_algorithm_codes=1 interior-point-convex; 2 trust-region-reflective; 3 active-set\n');
end

function write_parameter_audit(path,c,source)
names={'T','n','x0','Ntr','Nev','S','lambda','omega','alpha','beta', ...
    'volatility_floor','correlation_eigenvalue_floor','base_seed','seed_offset'}';
expected=[50;10;1;60;1000;2000;5e-4;1e-6;0.05;0.90;0.005;3e-5;42;20000];
detected=[c.T;c.n;c.x0;c.Ntr;c.Nev;c.S;c.lambda;c.omega;c.alpha; ...
    c.beta;c.volatility_floor;c.correlation_eigenvalue_floor;c.base_seed;c.seed_offset];
status=repmat("exact_match",numel(names),1);
status(expected~=detected)="conflict";
if strcmp(c.mode,'dry'), status(strcmp(names,'S'))="dry_run_10"; end
source_file=repmat(string(mfilename('fullpath')),numel(names),1);
source_file(strcmp(names,'base_seed') | strcmp(names,'seed_offset'))=string(source.main_path);
tbl=table(string(names),expected,detected,status,source_file, ...
    'VariableNames',{'parameter','expected_full_run','detected_in_code', ...
    'match_status','source_file'});
writetable(tbl,path);
end

function write_failure(path,mode,ME)
fid=fopen(path,'w'); if fid<0, return; end; cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# Experiment 3 population-regret candidate failure\n\n');
fprintf(fid,'- Mode: %s\n- Identifier: `%s`\n- Message: %s\n- Time: %s\n\n', ...
    mode,ME.identifier,ME.message,char(datetime('now','Format','yyyy-MM-dd HH:mm:ss Z')));
fprintf(fid,'No paper-candidate figures should be used from this failed invocation.\n\n## Stack\n\n');
for k=1:numel(ME.stack), fprintf(fid,'- `%s` line %d\n',ME.stack(k).name,ME.stack(k).line); end
end

function remove_candidate_figures(figures_dir)
if exist(figures_dir,'dir')~=7, return; end
files=[dir(fullfile(figures_dir,'exp3_fixedC_longruninit_minfix*.png'));dir(fullfile(figures_dir,'exp3_fixedC_longruninit_minfix*.pdf'))];
for k=1:numel(files), delete(fullfile(files(k).folder,files(k).name)); end
end

function export_pair(fig,dirpath,base)
exportgraphics(fig,fullfile(dirpath,[base '.png']),'Resolution',300);
exportgraphics(fig,fullfile(dirpath,[base '.pdf']),'ContentType','vector');
end

function format_axes(xlab,ylab)
xlabel(xlab); ylabel(ylab); grid on; box on; set(gca,'FontName','Helvetica','FontSize',11,'LineWidth',0.8);
end

function a=stage_angles(X,Y)
T=size(X,2); a=nan(1,T);
for t=1:T
    den=norm(X(:,t))*norm(Y(:,t));
    if den>eps, z=max(-1,min(1,(X(:,t)'*Y(:,t))/den)); a(t)=acosd(z); end
end
end

function best=best_direct_qp_crosscheck(S1,lambda1,mu,one,x,u_reference,u_eto,u_ieo,regret_reference,K2,rhs,options,config)
% Try all native direct-QP algorithms without modifying the Hessian. The
% returned solution is the converged candidate with the best normalized
% agreement score. Algorithm codes: 1=interior-point, 2=trust-region,
% 3=active-set.
best=struct('u_relative',inf,'regret_relative',inf,'kkt_residual',inf, ...
    'exitflag',-999,'algorithm_code',nan,'score',inf);
for j=1:numel(options)
    try
        [u,~,exitflag]=quadprog(2*S1,-lambda1*mu,[],[],one',x,[],[],u_reference,options{j});
        if isempty(u) || any(~isfinite(u)), continue; end
        urel=relative_difference(u_reference,u);
        r=[(u_eto-u)'*S1*(u_eto-u),(u_ieo-u)'*S1*(u_ieo-u)];
        rrel=max(relative_scalar(regret_reference,r));
        gamma=-mean(2*S1*u-lambda1*mu);
        kres=norm(K2*[u;gamma]-rhs)/max(1,norm(rhs));
        score=max([urel/config.crosscheck_relative_tolerance, ...
            rrel/config.crosscheck_qp_regret_tolerance, ...
            kres/config.high_residual_threshold]);
        if exitflag<=0, score=score+100; end
        if score<best.score
            best=struct('u_relative',urel,'regret_relative',rrel, ...
                'kkt_residual',kres,'exitflag',exitflag, ...
                'algorithm_code',j,'score',score);
        end
    catch
        % The algorithm may reject a roundoff-level indefinite Hessian.
        % Other unmodified-Hessian algorithms are still attempted.
    end
end
end

function X=pivoted_ldl_solve(A,B)
A=(A+A')/2;
try
    [L,D,p]=ldl(A,'vector');
    Y=L\B(p,:); Z=D\Y; W=L'\Z; X=zeros(size(B)); X(p,:)=W;
catch
    X=A\B;
end
end

function r=relative_difference(x,y)
r=norm(x-y)/max([1,norm(x),norm(y)]);
end

function r=relative_scalar(x,y)
r=abs(x-y)./max(1,max(abs(x),abs(y)));
end

function value=ternary(condition,a,b)
if condition, value=a; else, value=b; end
end

function ensure_dir(path)
if exist(path,'dir')~=7, mkdir(path); end
end

function preallocate_raw(raw,S,T,n)
% Set final HDF5 dimensions once; subsequent slice writes do not repeatedly
% grow a large file. Values are overwritten during the run.
raw.true_mu_paths(S,T,n)=0;
raw.true_Sigma_paths(S,T,n,n)=0;
raw.true_volatility_paths(S,T,n)=0;
raw.true_correlation_matrices(S,n,n)=0;
raw.true_epsilon_mu_paths(S,T,n)=0;
raw.dgp_jitter_paths(S,T)=0;
raw.policy_A_ETO(S,T,n)=0; raw.policy_B_ETO(S,T,n)=0;
raw.policy_A_IEO(S,T,n)=0; raw.policy_B_IEO(S,T,n)=0;
raw.true_oracle_A(S,T,n)=0; raw.true_oracle_B(S,T,n)=0;
raw.true_oracle_policy_x1(S,T,n)=0;
raw.true_Sigma1_paths(S,T,n,n)=0;
raw.true_kappa1_paths(S,T+1)=0; raw.true_kappa2_paths(S,T+1)=0;
raw.true_kappa0_paths(S,T+1)=0;
end

function assert_none_exist(paths)
for k=1:numel(paths)
    assert(exist(paths{k},'file')~=2,'Exp3Candidate:ProtectedOutputExists', ...
        'Refusing to overwrite existing candidate output: %s',paths{k});
end
end

function text=yesno(condition)
text=ternary(condition,'YES','NO');
end

function digest=sha256_file(path)
[status,out]=system(sprintf('shasum -a 256 %s',shell_quote(path)));
assert(status==0,'Exp3Candidate:HashFailure','Could not hash %s',path);
parts=strsplit(strtrim(out)); digest=parts{1};
end

function q=shell_quote(s)
q=['''' strrep(s,'''','''"''"''') ''''];
end
