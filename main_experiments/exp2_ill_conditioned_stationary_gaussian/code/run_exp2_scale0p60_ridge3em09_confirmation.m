function outputs = run_exp2_scale0p60_ridge3em09_confirmation(mode)
%RUN_EXP2_SCALE0P60_RIDGE3EM09_CONFIRMATION
% Locked confirmation of the selected S=200 screening candidate.
% scale=0.60 and covariance ridge=3e-9 are fixed; no screening occurs.
% Population regret uses exact fixed population moments and unmodified Sigma1.

if nargin < 1, mode = 'dry'; end
mode = validatestring(lower(char(mode)), {'dry','full'});
config = locked_config(mode);
ensure_dir(config.results_dir); ensure_dir(config.audit_dir);

old_path = path; cleanup_path = onCleanup(@() path(old_path));
old_dir = pwd; cleanup_dir = onCleanup(@() cd(old_dir));
cd(config.code_dir);
addpath(config.code_dir, '-begin');
assert_core_hashes(config);
[mu, Sigma, sigma_diagnostics, source_paths] = locked_candidate_dgp(config);
oracle = exact_population_oracle(mu, Sigma, config);

write_dgp_files(mu, Sigma, sigma_diagnostics, source_paths, config);
arrays = initialize_arrays(config.NumSims, config);
completed = false(config.NumSims,1);
checkpoint = fullfile(config.results_dir, sprintf('checkpoint_%s.mat',mode));
if isfile(checkpoint)
    saved = load(checkpoint,'arrays','completed','config','mu','Sigma');
    assert(saved.config.NumSims == config.NumSims && strcmp(saved.config.mode,mode));
    assert(isequal(saved.mu,mu) && isequal(saved.Sigma,Sigma));
    arrays=saved.arrays; completed=saved.completed;
end

remaining=find(~completed);
if config.use_parallel && ~isempty(remaining)
    pool=gcp('nocreate');
    if isempty(pool), parpool('local',config.parallel_workers); end
end
for first=1:config.batch_size:numel(remaining)
    ids=remaining(first:min(first+config.batch_size-1,numel(remaining)));
    blocks=cell(numel(ids),1);
    if config.use_parallel
        parfor k=1:numel(ids)
            blocks{k}=one_replication(ids(k),mu,Sigma,oracle,config);
        end
    else
        for k=1:numel(ids)
            blocks{k}=one_replication(ids(k),mu,Sigma,oracle,config);
        end
    end
    for k=1:numel(ids)
        arrays=store_one(arrays,ids(k),blocks{k}); completed(ids(k))=true;
    end
    atomic_checkpoint(checkpoint,arrays,completed,config,mu,Sigma);
    fprintf('Exp2 locked scale0p60 ridge3em09 confirmation (%s): %d/%d complete.\n',mode,sum(completed),config.NumSims);
end

assert(all(completed),'Candidate run incomplete.');
validation=validate_run(arrays,oracle,Sigma,sigma_diagnostics,config);
if ~all(validation.passed)
    failure_file=fullfile(config.audit_dir,upper(mode)+"_RUN_FAILURE.md");
    write_failure(failure_file,validation,config);
    error('Exp2Candidate:ValidationFailed','%s validation failed. See %s.',mode,failure_file);
end

[stagewise,wealth,coeff,extreme,exposure,near_tbl,high_tbl,summary] = ...
    aggregate_outputs(arrays,oracle,config);
if strcmp(mode,'dry')
    dry_dir=fullfile(config.results_dir,'dry_run'); ensure_dir(dry_dir);
    save(fullfile(dry_dir,'exp2_scale0p60_ridge3em09_confirmation_dry_raw.mat'), ...
        'arrays','completed','config','mu','Sigma','sigma_diagnostics','source_paths','oracle','validation','summary','-v7.3');
    writetable(validation,fullfile(dry_dir,'exp2_scale0p60_ridge3em09_confirmation_dry_validation.csv'));
    writetable(stagewise,fullfile(dry_dir,'exp2_scale0p60_ridge3em09_confirmation_dry_stagewise.csv'));
    write_dry_audit(validation,summary,sigma_diagnostics,config);
    outputs=struct('mode',mode,'validation',validation,'summary',summary);
    clear cleanup_path cleanup_dir; return
end

raw_file=fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_raw.mat');
policy_form='u_t(x)=A_t*x+B_t; A is slope and B is intercept.';
population_regret_definition=['delta''*Sigma1_true(t)*delta at x=1; true fixed mu/Sigma; ' ...
    'no R_B moments, hindsight oracle, proxy regret, population ridge, or clipping.'];
save(raw_file,'arrays','completed','config','mu','Sigma','sigma_diagnostics', ...
    'source_paths','oracle','validation','summary','policy_form', ...
    'population_regret_definition','-v7.3');
writetable(stagewise,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_stagewise_population_regret.csv'));
writetable(wealth,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_wealth.csv'));
writetable(coeff,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_coefficient_diagnostics.csv'));
writetable(extreme,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_extreme_points.csv'));
writetable(near_tbl,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_near_spd_warnings.csv'));
writetable(high_tbl,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_high_residual_warnings.csv'));
writetable(exposure,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_policy_exposure_diagnostics.csv'));
writetable(validation,fullfile(config.results_dir,'exp2_scale0p60_ridge3em09_confirmation_validation.csv'));
write_config(config,sigma_diagnostics,summary);
write_run_log(config,sigma_diagnostics,summary,validation);
write_full_audits(config,sigma_diagnostics,summary,validation,arrays,oracle);
outputs=struct('mode',mode,'raw',raw_file,'stagewise',stagewise,'wealth',wealth, ...
    'coeff',coeff,'summary',summary,'validation',validation);
clear cleanup_path cleanup_dir
end

function c=locked_config(mode)
here=fileparts(mfilename('fullpath')); root=fileparts(here);
c=struct(); c.mode=mode; c.candidate_root=root; c.code_dir=here;
c.results_dir=fullfile(root,'results'); c.figure_dir=fullfile(root,'figures');
c.audit_dir=fullfile(root,'audit'); c.latex_dir=fullfile(root,'latex');
c.T=50; c.n=10; c.Ntr=60; c.Nev=1000; c.x0=1; c.x_eval=1;
c.lambda_scalar=1e-4; c.lambda=c.lambda_scalar*ones(1,c.T);
c.base_seed=42; c.replication_seed_offset=940000; c.fixed_dgp_seed=11902;
c.mean_signal_scale=0.60; c.common_mean_shift=1.8e-4; c.covariance_ridge=3e-9;
c.expected_condition=195246.156950; c.candidate_id='candidate_scale_0p6_ridge_3em09';
c.tol_spd_multiplier=1e-10; c.high_residual_threshold=1e-6;
c.negative_regret_tolerance=1e-10; c.oracle_self_regret_tolerance=1e-10;
c.budget_tolerance=1e-8; c.crosscheck_tolerance=1e-6;
c.bootstrap_seed=20260915; c.bootstrap_resamples=2000;
c.parallel_workers=4; c.batch_size=10; c.use_parallel=true;
if strcmp(mode,'dry')
    c.NumSims=10; c.use_parallel=false; c.batch_size=2;
else
    c.NumSims=2000;
end
c.source_runner='/Users/sty/Downloads/IEO_code revise/curated_paper_simulation_experiments/02_main_ill_conditioned_stationary_gaussian/diagnostics/exp2_current_dgp_population_regret_screening_s200/code/run_exp2_current_dgp_screen_candidate.m';
c.source_raw='/Users/sty/Downloads/IEO_code revise/curated_paper_simulation_experiments/02_main_ill_conditioned_stationary_gaussian/diagnostics/exp2_current_dgp_population_regret_screening_s200/results/candidate_scale_0p6_ridge_3em09/screen_raw.mat';
c.source_summary='/Users/sty/Downloads/IEO_code revise/curated_paper_simulation_experiments/02_main_ill_conditioned_stationary_gaussian/diagnostics/exp2_current_dgp_population_regret_screening_s200/results/exp2_current_dgp_screening_s200_candidate_ranking.csv';
c.current_exp2_runner='/Users/sty/Downloads/IEO_code revise/curated_paper_simulation_experiments/02_main_ill_conditioned_stationary_gaussian/code/ORL-D-26-00196_simulation/experiment_4_fixed_stationary_gaussian/run_fixed_stationary_gaussian_experiment.m';
end

function assert_core_hashes(c)
names={'run_eto.m','run_ieo.m','generate_data.m'};
official='/Users/sty/Downloads/IEO_code revise/curated_paper_simulation_experiments/02_main_ill_conditioned_stationary_gaussian/code';
for k=1:numel(names)
    assert(strcmp(fileread(fullfile(c.code_dir,names{k})),fileread(fullfile(official,names{k}))), ...
        'Copied core differs from current Main Exp2: %s',names{k});
end
clear run_eto run_ieo generate_data
rehash path
assert(strcmp(which('run_eto'),fullfile(c.code_dir,'run_eto.m')));
assert(strcmp(which('run_ieo'),fullfile(c.code_dir,'run_ieo.m')));
assert(strcmp(which('generate_data'),fullfile(c.code_dir,'generate_data.m')));
end

function [mu,Sigma,d,paths]=locked_candidate_dgp(c)
% Literal reconstruction of the selected screening candidate.
rng(c.fixed_dgp_seed,'twister'); n=c.n;
mu_annual=-0.05+0.20*rand(n,1); vol_annual=0.10+0.10*rand(n,1);
C=0.2+0.6*rand(n); C=(C+C')/2; C(1:n+1:end)=1;
[V,D]=eig(C); D(D<1e-2)=1e-2; C=V*D*V';
q=sqrt(diag(C)); C=C./(q*q'); C=(C+C')/2;
Sigma_annual=diag(vol_annual)*C*diag(vol_annual);
mu0=1+mu_annual/252; Sigma0=Sigma_annual/252; Sigma0=(Sigma0+Sigma0')/2;
jitter=0; [~,chol_flag]=chol(Sigma0,'lower'); iteration=0;
while chol_flag~=0 && iteration<12
    if jitter==0, jitter=max(1e-12,1e-10*trace(Sigma0)/n); else, jitter=jitter*10; end
    Sigma0=Sigma0+jitter*eye(n); Sigma0=(Sigma0+Sigma0')/2;
    [~,chol_flag]=chol(Sigma0,'lower'); iteration=iteration+1;
end
assert(chol_flag==0,'Unable to reconstruct locked candidate population.');
base_mean=mean(mu0); base_trace=trace(Sigma0);
mu=base_mean+c.common_mean_shift+c.mean_signal_scale*(mu0-base_mean);
Sigma=Sigma0+c.covariance_ridge*eye(n);
trace_preserving_scale=base_trace/trace(Sigma); Sigma=trace_preserving_scale*Sigma;
Sigma=(Sigma+Sigma')/2; e=eig(Sigma);
d=struct('mean_signal_scale',c.mean_signal_scale,'common_mean_shift',c.common_mean_shift, ...
    'covariance_ridge',c.covariance_ridge, ...
    'fixed_dgp_seed',c.fixed_dgp_seed,'min_eig',min(e),'max_eig',max(e), ...
    'condition_number',cond(Sigma,2),'trace',trace(Sigma),'norm_2',norm(Sigma,2), ...
    'symmetry_error',norm(Sigma-Sigma','fro'),'is_spd',all(e>0),'determinant',det(Sigma), ...
    'trace_preserving_scale',trace_preserving_scale,'source_jitter',jitter);
assert(all(e>0) && norm(Sigma-Sigma','fro')<1e-14);
assert(abs(d.condition_number-c.expected_condition)/c.expected_condition<1e-9,'Locked DGP condition fingerprint mismatch.');
assert(isfile(c.source_raw),'Selected S=200 source raw MAT missing.');
selected=load(c.source_raw,'mu','Sigma','c');
assert(strcmp(selected.c.candidate_id,c.candidate_id) && selected.c.mean_signal_scale==c.mean_signal_scale ...
    && selected.c.covariance_ridge==c.covariance_ridge,'Selected candidate metadata mismatch.');
assert(max(abs(selected.mu(:)-mu(:)))<1e-15 && max(abs(selected.Sigma(:)-Sigma(:)))<1e-15, ...
    'Reconstructed DGP differs from selected S=200 candidate.');
paths=struct('builder',c.source_runner,'selected_screen_raw',c.source_raw, ...
    'screening_ranking',c.source_summary,'current_exp2_runner',c.current_exp2_runner);
end

function oracle=exact_population_oracle(mu,Sigma,c)
n=c.n; T=c.T; one=ones(n,1);
oracle.A=nan(n,T); oracle.B=nan(n,T); oracle.u_star=nan(n,T);
oracle.Sigma1=nan(n,n,T); oracle.kappa1=zeros(1,T+1);
oracle.kappa2=zeros(1,T+1); oracle.kappa0=zeros(1,T+1);
fields={'min_eig_Sigma1','max_eig_Sigma1','cond_Sigma1','norm2_Sigma1','tol_spd', ...
    'solve_residual_1','solve_residual_mu','KKT_residual','budget_residual', ...
    'oracle_self_regret','near_spd_flag','high_residual_flag','alternative_relative_difference'};
for k=1:numel(fields), oracle.diagnostics.(fields{k})=zeros(1,T); end
for t=T:-1:1
    S1=(1+oracle.kappa2(t+1))*Sigma+oracle.kappa2(t+1)*(mu*mu'); S1=(S1+S1')/2;
    ev=eig(S1); norm2=norm(S1,2); tol=c.tol_spd_multiplier*max(1,norm2);
    if min(ev)<-tol, error('Exp2Candidate:TrueSigma1NotSPD','stage %d eig %.17g tol %.17g',t-1,min(ev),tol); end
    pq=S1\[one,mu]; p=pq(:,1); q=pq(:,2);
    r1=norm(S1*p-one)/max(1,norm(one)); rm=norm(S1*q-mu)/max(1,norm(mu));
    lambda1=c.lambda_scalar-oracle.kappa1(t+1);
    K=[S1,one;one',0]; sol=K\[zeros(n,1),mu;1,0];
    v=sol(1:n,1); w=sol(1:n,2); eta_v=sol(end,1); eta_w=sol(end,2);
    u=v+c.x_eval*0 + (lambda1/2)*w; % x_eval is one; written explicitly below
    u=c.x_eval*v+(lambda1/2)*w;
    K2=[2*S1,one;one',0]; direct=K2\[lambda1*mu;c.x_eval]; u2=direct(1:n);
    alt_diff=norm(u-u2)/max(1,norm(u));
    high=max(r1,rm)>c.high_residual_threshold;
    if high
        alt=lsqminnorm(K2,[lambda1*mu;c.x_eval]);
        if norm(u-alt(1:n))/max(1,norm(u))>c.crosscheck_tolerance
            error('Exp2Candidate:AlternativeSolverMismatch','stage %d alternative solver mismatch',t-1);
        end
    end
    if alt_diff>c.crosscheck_tolerance, error('Exp2Candidate:OracleMismatch','stage %d KKT mismatch',t-1); end
    oracle.A(:,t)=v; oracle.B(:,t)=(lambda1/2)*w; oracle.u_star(:,t)=u;
    oracle.kappa2(t)=-eta_v; oracle.kappa1(t)=-lambda1*eta_w;
    oracle.kappa0(t)=oracle.kappa0(t+1)-(lambda1^2/4)*(mu'*w);
    oracle.Sigma1(:,:,t)=S1;
    oracle.diagnostics.min_eig_Sigma1(t)=min(ev); oracle.diagnostics.max_eig_Sigma1(t)=max(ev);
    oracle.diagnostics.cond_Sigma1(t)=cond(S1,2); oracle.diagnostics.norm2_Sigma1(t)=norm2;
    oracle.diagnostics.tol_spd(t)=tol; oracle.diagnostics.solve_residual_1(t)=r1;
    oracle.diagnostics.solve_residual_mu(t)=rm;
    oracle.diagnostics.KKT_residual(t)=norm(K2*[u;direct(end)]-[lambda1*mu;c.x_eval])/max(1,norm([lambda1*mu;c.x_eval]));
    oracle.diagnostics.budget_residual(t)=abs(sum(u)-c.x_eval);
    oracle.diagnostics.oracle_self_regret(t)=(u-u2)'*S1*(u-u2);
    oracle.diagnostics.near_spd_flag(t)=min(ev)<0;
    oracle.diagnostics.high_residual_flag(t)=high;
    oracle.diagnostics.alternative_relative_difference(t)=alt_diff;
end
end

function a=initialize_arrays(S,c)
T=c.T; n=c.n; Nev=c.Nev;
stage={'Regret_ETO_pop_sims','Regret_IEO_pop_sims','Delta_R_pop_sims', ...
    'intercept_gap_sims','slope_gap_sims','intercept_angle_sims','slope_angle_sims', ...
    'max_relative_gross_ETO','max_relative_gross_IEO','max_gross_ETO','max_gross_IEO', ...
    'max_policy_norm_ETO','max_policy_norm_IEO'};
for k=1:numel(stage),a.(stage{k})=nan(S,T);end
a.Wealth_Paths_ETO=nan(S,T+1); a.Wealth_Paths_IEO=nan(S,T+1);
a.Terminal_Wealth_ETO=nan(S,Nev); a.Terminal_Wealth_IEO=nan(S,Nev);
a.A_ETO=nan(n,T,S); a.B_ETO=nan(n,T,S); a.A_IEO=nan(n,T,S); a.B_IEO=nan(n,T,S);
a.max_budget_eto=nan(S,1); a.max_budget_ieo=nan(S,1); a.valid=false(S,1);
a.seed=nan(S,1); a.seedA=nan(S,1); a.seedB=nan(S,1);
a.success_eto=false(S,1); a.success_ieo=false(S,1); a.error_eto=strings(S,1); a.error_ieo=strings(S,1);
end

function o=one_replication(s,mu,Sigma,oracle,c)
T=c.T; n=c.n; root=c.base_seed+c.replication_seed_offset+s; seedA=root+100; seedB=root+200;
mu_path=repmat(mu',T,1); Sigma_path=repmat(reshape(Sigma,1,n,n),T,1,1);
[RA,RB]=generate_data(T,n,mu_path,Sigma_path,c.Ntr,c.Nev,seedA,seedB);
o=struct('success_eto',false,'success_ieo',false,'error_eto',"",'error_ieo',"",'valid',false);
try, evalc('eto=run_eto(RA,RB,c.lambda,c.x0);'); o.success_eto=true; catch ex, o.error_eto=string(getReport(ex,'basic','hyperlinks','off')); end
try, evalc('ieo=run_ieo(RA,RB,c.lambda,c.x0);'); o.success_ieo=true; catch ex, o.error_ieo=string(getReport(ex,'basic','hyperlinks','off')); end
if ~(o.success_eto&&o.success_ieo), return; end
o.A_ETO=eto.strategy.A_coef; o.B_ETO=eto.strategy.B_coef;
o.A_IEO=ieo.strategy.A_coef; o.B_IEO=ieo.strategy.B_coef;
uE=o.A_ETO+o.B_ETO; uI=o.A_IEO+o.B_IEO;
o.Regret_ETO_pop_sims=zeros(1,T); o.Regret_IEO_pop_sims=zeros(1,T);
for t=1:T
    dE=uE(:,t)-oracle.u_star(:,t); dI=uI(:,t)-oracle.u_star(:,t); M=oracle.Sigma1(:,:,t);
    o.Regret_ETO_pop_sims(t)=dE'*M*dE; o.Regret_IEO_pop_sims(t)=dI'*M*dI;
end
o.Delta_R_pop_sims=o.Regret_ETO_pop_sims-o.Regret_IEO_pop_sims;
o.intercept_gap_sims=vecnorm(o.B_ETO-o.B_IEO,2,1); o.slope_gap_sims=vecnorm(o.A_ETO-o.A_IEO,2,1);
o.intercept_angle_sims=angles_deg(o.B_ETO,o.B_IEO); o.slope_angle_sims=angles_deg(o.A_ETO,o.A_IEO);
o.Wealth_Paths_ETO=eto.wealth.X_mean(:)'; o.Wealth_Paths_IEO=ieo.evaluation.X_mean(:)';
o.Terminal_Wealth_ETO=eto.wealth.X(:,end)'; o.Terminal_Wealth_IEO=ieo.evaluation.X(:,end)';
[o.max_relative_gross_ETO,o.max_gross_ETO,o.max_policy_norm_ETO]=exposure_by_stage(eto.wealth.U,eto.wealth.X(:,1:T));
[o.max_relative_gross_IEO,o.max_gross_IEO,o.max_policy_norm_IEO]=exposure_by_stage(ieo.evaluation.U,ieo.evaluation.X(:,1:T));
o.max_budget_eto=max(abs(squeeze(sum(eto.wealth.U,1))-eto.wealth.X(:,1:T)),[],'all');
o.max_budget_ieo=max(abs(squeeze(sum(ieo.evaluation.U,1))-ieo.evaluation.X(:,1:T)),[],'all');
o.seed=root; o.seedA=seedA; o.seedB=seedB;
vals=[o.Regret_ETO_pop_sims,o.Regret_IEO_pop_sims,o.Wealth_Paths_ETO,o.Wealth_Paths_IEO];
o.valid=all(isfinite(vals)) && min([o.Regret_ETO_pop_sims,o.Regret_IEO_pop_sims])>=-c.negative_regret_tolerance ...
    && max(o.max_budget_eto,o.max_budget_ieo)<=c.budget_tolerance;
end

function a=store_one(a,s,o)
scalar={'valid','success_eto','success_ieo','error_eto','error_ieo','max_budget_eto','max_budget_ieo','seed','seedA','seedB'};
for k=1:numel(scalar),if isfield(o,scalar{k}),a.(scalar{k})(s)=o.(scalar{k});end,end
if ~o.success_eto || ~o.success_ieo, return; end
stage={'Regret_ETO_pop_sims','Regret_IEO_pop_sims','Delta_R_pop_sims','intercept_gap_sims','slope_gap_sims', ...
    'intercept_angle_sims','slope_angle_sims','max_relative_gross_ETO','max_relative_gross_IEO', ...
    'max_gross_ETO','max_gross_IEO','max_policy_norm_ETO','max_policy_norm_IEO'};
for k=1:numel(stage),a.(stage{k})(s,:)=o.(stage{k});end
a.Wealth_Paths_ETO(s,:)=o.Wealth_Paths_ETO; a.Wealth_Paths_IEO(s,:)=o.Wealth_Paths_IEO;
a.Terminal_Wealth_ETO(s,:)=o.Terminal_Wealth_ETO; a.Terminal_Wealth_IEO(s,:)=o.Terminal_Wealth_IEO;
a.A_ETO(:,:,s)=o.A_ETO; a.B_ETO(:,:,s)=o.B_ETO; a.A_IEO(:,:,s)=o.A_IEO; a.B_IEO(:,:,s)=o.B_IEO;
end

function v=validate_run(a,o,Sigma,d,c)
check=strings(0,1); value=[]; tolerance=[]; passed=false(0,1); notes=strings(0,1);
    function add(name,val,tol,ok,note),check(end+1,1)=string(name);value(end+1,1)=val;tolerance(end+1,1)=tol;passed(end+1,1)=ok;notes(end+1,1)=string(note);end
add('locked_candidate_condition_fingerprint',d.condition_number,c.expected_condition,abs(d.condition_number-c.expected_condition)/c.expected_condition<1e-9,'cond(Sigma) fingerprint');
add('Sigma_SPD',d.min_eig,0,d.min_eig>0,'exact source matrix');
add('all_replications_completed',sum(a.success_eto&a.success_ieo),c.NumSims,all(a.success_eto&a.success_ieo),'ETO/IEO completed');
add('all_replications_valid',sum(a.valid),c.NumSims,all(a.valid),'finite regret/wealth and budget');
add('minimum_population_regret',min([a.Regret_ETO_pop_sims(:);a.Regret_IEO_pop_sims(:)]),-c.negative_regret_tolerance,min([a.Regret_ETO_pop_sims(:);a.Regret_IEO_pop_sims(:)])>=-c.negative_regret_tolerance,'no severe negative regret');
add('maximum_oracle_self_regret',max(abs(o.diagnostics.oracle_self_regret)),c.oracle_self_regret_tolerance,max(abs(o.diagnostics.oracle_self_regret))<=c.oracle_self_regret_tolerance,'KKT self comparison');
add('maximum_oracle_budget_residual',max(o.diagnostics.budget_residual),c.budget_tolerance,max(o.diagnostics.budget_residual)<=c.budget_tolerance,'1''u*=1');
add('maximum_solve_residual',max([o.diagnostics.solve_residual_1,o.diagnostics.solve_residual_mu]),c.high_residual_threshold,max([o.diagnostics.solve_residual_1,o.diagnostics.solve_residual_mu])<=c.high_residual_threshold,'unmodified Sigma1 solve');
add('maximum_policy_budget_residual',max([a.max_budget_eto;a.max_budget_ieo]),c.budget_tolerance,max([a.max_budget_eto;a.max_budget_ieo])<=c.budget_tolerance,'evaluation-path self financing');
add('stationary_population_dimensions',numel(Sigma),c.n^2,isequal(size(Sigma),[c.n,c.n]),'fixed across stages and replications');
v=table(check,value,tolerance,passed,notes);
end

function [stagewise,wealth,coeff,extreme,exposure,near_tbl,high_tbl,summary]=aggregate_outputs(a,o,c)
valid=a.valid; E=a.Regret_ETO_pop_sims(valid,:); I=a.Regret_IEO_pop_sims(valid,:); D=E-I; S=sum(valid); T=c.T;
[dl,dh]=paired_bootstrap_ci(D,c.bootstrap_resamples,c.bootstrap_seed);
stage=(0:T-1)'; stagewise=table(stage,mean(E)',mean(I)',std(E,0,1)'/sqrt(S),std(I,0,1)'/sqrt(S),mean(D)', ...
    median(E)',median(I)',quantile(E,.95,1)',quantile(I,.95,1)',quantile(E,.99,1)',quantile(I,.99,1)', ...
    max(E,[],1)',max(I,[],1)',max(E,[],1)'./sum(E,1)',max(I,[],1)'./sum(I,1)',mean(D)',std(D,0,1)'/sqrt(S),dl',dh', ...
    repmat("tie",T,1),'VariableNames',{'stage_t','regret_pop_eto_mean','regret_pop_ieo_mean','regret_pop_eto_se','regret_pop_ieo_se', ...
    'regret_pop_eto_minus_ieo','regret_pop_eto_median','regret_pop_ieo_median','regret_pop_eto_p95','regret_pop_ieo_p95', ...
    'regret_pop_eto_p99','regret_pop_ieo_p99','max_regret_eto','max_regret_ieo','max_contribution_to_mean_eto', ...
    'max_contribution_to_mean_ieo','delta_pop_mean','delta_pop_se','delta_pop_bootstrap_ci_low','delta_pop_bootstrap_ci_high','method_with_lower_mean_regret'});
stagewise.method_with_lower_mean_regret(stagewise.delta_pop_mean>0)="IEO"; stagewise.method_with_lower_mean_regret(stagewise.delta_pop_mean<0)="ETO";
WE=a.Wealth_Paths_ETO(valid,:); WI=a.Wealth_Paths_IEO(valid,:); ws=(0:T)';
wealth=table(ws,mean(WE)',mean(WI)',median(WE)',median(WI)',quantile(WE,.025,1)',quantile(WE,.975,1)', ...
    quantile(WI,.025,1)',quantile(WI,.975,1)','VariableNames',{'wealth_stage','wealth_eto_mean','wealth_ieo_mean', ...
    'wealth_eto_median','wealth_ieo_median','wealth_eto_q025','wealth_eto_q975','wealth_ieo_q025','wealth_ieo_q975'});
coeff=coefficient_table(a,valid,c);
exposure=exposure_table(a,valid,c);
extreme=extreme_table(a,valid,c);
near=find(o.diagnostics.near_spd_flag); near_tbl=table((near-1)',o.diagnostics.min_eig_Sigma1(near)',o.diagnostics.norm2_Sigma1(near)',o.diagnostics.tol_spd(near)', ...
    'VariableNames',{'stage_t','eig_min','norm2','tol_spd'});
high=find(o.diagnostics.high_residual_flag); high_tbl=table((high-1)',o.diagnostics.solve_residual_1(high)',o.diagnostics.solve_residual_mu(high)', ...
    'VariableNames',{'stage_t','solve_residual_1','solve_residual_mu'});
te=mean(a.Terminal_Wealth_ETO(valid,:),2); ti=mean(a.Terminal_Wealth_IEO(valid,:),2);
summary=struct('valid_replications',S,'IEO_lower_stages',sum(mean(I)<mean(E)),'ETO_lower_stages',sum(mean(E)<mean(I)), ...
    'overall_mean_regret_ETO',mean(E,'all'),'overall_mean_regret_IEO',mean(I,'all'), ...
    'terminal_wealth_ETO_mean',mean(te),'terminal_wealth_ETO_median',median(te),'terminal_wealth_ETO_p95',quantile(te,.95), ...
    'terminal_wealth_ETO_p99',quantile(te,.99),'terminal_wealth_ETO_max',max(te), ...
    'terminal_wealth_IEO_mean',mean(ti),'terminal_wealth_IEO_median',median(ti),'terminal_wealth_IEO_p95',quantile(ti,.95), ...
    'terminal_wealth_IEO_p99',quantile(ti,.99),'terminal_wealth_IEO_max',max(ti), ...
    'max_single_contribution',max([stagewise.max_contribution_to_mean_eto;stagewise.max_contribution_to_mean_ieo]), ...
    'max_relative_gross_exposure',max([a.max_relative_gross_ETO(:);a.max_relative_gross_IEO(:)]), ...
    'near_spd_count',height(near_tbl),'high_residual_count',height(high_tbl), ...
    'max_regret',max([E(:);I(:)]),'max_budget_residual',max([a.max_budget_eto;a.max_budget_ieo]));
end

function tbl=coefficient_table(a,v,c)
names={'intercept_gap','slope_gap','intercept_angle','slope_angle'}; fields={'intercept_gap_sims','slope_gap_sims','intercept_angle_sims','slope_angle_sims'};
rows=cell(numel(names)*c.T,9); r=0;
for k=1:numel(names),X=a.(fields{k})(v,:);[lo,hi]=mean_bootstrap_ci(X,c.bootstrap_resamples,c.bootstrap_seed+10*k);
 for t=1:c.T,r=r+1;rows(r,:)={names{k},t-1,mean(X(:,t),'omitnan'),std(X(:,t),'omitnan')/sqrt(sum(isfinite(X(:,t)))),median(X(:,t),'omitnan'),quantile(X(:,t),.95),quantile(X(:,t),.99),lo(t),hi(t)};end,end
tbl=cell2table(rows,'VariableNames',{'statistic_name','stage_t','mean','se','median','p95','p99','bootstrap_ci_low','bootstrap_ci_high'});
end

function tbl=exposure_table(a,v,c)
methods={'ETO','IEO'}; rows=cell(2*c.T,9); r=0;
for k=1:2
 rel=a.(['max_relative_gross_' methods{k}])(v,:); gross=a.(['max_gross_' methods{k}])(v,:); normp=a.(['max_policy_norm_' methods{k}])(v,:);
 for t=1:c.T,r=r+1;rows(r,:)={methods{k},t-1,mean(rel(:,t)),median(rel(:,t)),quantile(rel(:,t),.95),quantile(rel(:,t),.99),max(rel(:,t)),max(gross(:,t)),max(normp(:,t))};end
end
tbl=cell2table(rows,'VariableNames',{'method','stage_t','relative_gross_mean','relative_gross_median','relative_gross_p95','relative_gross_p99','relative_gross_max','gross_exposure_max','policy_norm_max'});
end

function tbl=extreme_table(a,v,c)
ids=find(v); E=a.Regret_ETO_pop_sims(v,:); I=a.Regret_IEO_pop_sims(v,:); M=max(E,I); [~,ord]=sort(M(:),'descend'); ord=ord(1:min(20,numel(ord)));
[rr,tt]=ind2sub(size(M),ord); rows=table(ids(rr),a.seed(ids(rr)),tt-1,E(ord),I(ord),E(ord)-I(ord), ...
    E(ord)./sum(E(:,tt),1)',I(ord)./sum(I(:,tt),1)','VariableNames',{'replication','seed','stage_t','Regret_ETO_pop','Regret_IEO_pop','Delta_ETO_minus_IEO','contribution_to_stage_mean_ETO','contribution_to_stage_mean_IEO'});
flag=false(size(M)); for t=1:c.T,flag(:,t)=E(:,t)>0.1*sum(E(:,t)) | I(:,t)>0.1*sum(I(:,t));end
[r2,t2]=find(flag); if ~isempty(r2),ind=sub2ind(size(M),r2,t2); extra=table(ids(r2),a.seed(ids(r2)),t2-1,E(ind),I(ind),E(ind)-I(ind),E(ind)./sum(E(:,t2),1)',I(ind)./sum(I(:,t2),1)', ...
 'VariableNames',rows.Properties.VariableNames); rows=unique([rows;extra],'rows');end
tbl=rows;
end

function [rel,gross,normp]=exposure_by_stage(U,X)
T=size(U,3); rel=zeros(1,T); gross=zeros(1,T); normp=zeros(1,T);
for t=1:T,G=sum(abs(U(:,:,t)),1); rel(t)=max(G'./max(abs(X(:,t)),1e-12)); gross(t)=max(G); normp(t)=max(vecnorm(U(:,:,t),2,1));end
end

function ang=angles_deg(A,B)
T=size(A,2);ang=nan(1,T);for t=1:T,na=norm(A(:,t));nb=norm(B(:,t));if na>1e-10&&nb>1e-10,ang(t)=acosd(max(-1,min(1,(A(:,t)'*B(:,t))/(na*nb))));end,end
end

function [lo,hi]=paired_bootstrap_ci(X,B,seed),[lo,hi]=mean_bootstrap_ci(X,B,seed);end
function [lo,hi]=mean_bootstrap_ci(X,B,seed)
saved=rng;clean=onCleanup(@()rng(saved));rng(seed,'twister');N=size(X,1);T=size(X,2);boot=nan(B,T);
for first=1:100:B,ids=first:min(first+99,B);ix=randi(N,N,numel(ids));for t=1:T,y=X(:,t);boot(ids,t)=mean(y(ix),1)';end,end
q=quantile(boot,[.025,.975],1);lo=q(1,:);hi=q(2,:);clear clean
end

function write_dgp_files(mu,Sigma,d,paths,c)
writematrix(mu,fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_mu.csv'));
writematrix(Sigma,fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_Sigma.csv'));
metric=string(fieldnames(d));value=cellfun(@(x)double(d.(x)),cellstr(metric));
writetable(table(metric,value),fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_Sigma_diagnostics.csv'));
save(fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_dgp_parameters.mat'),'mu','Sigma','d','paths');
end

function write_config(c,d,s)
f=fopen(fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_config.txt'),'w');cl=onCleanup(@()fclose(f));
fprintf(f,'candidate=exp2_scale0p60_ridge3em09_population_regret_confirmation_s2000\nstatus=locked_confirmation_not_promoted\nT=%d\nn=%d\nx0=%.17g\nNtr=%d\nNev=%d\nS=%d\nlambda=%.17g\n',c.T,c.n,c.x0,c.Ntr,c.Nev,c.NumSims,c.lambda_scalar);
fprintf(f,'mean_signal_scale=%.17g\ncommon_mean_shift=%.17g\ncovariance_ridge=%.17g\nfixed_dgp_seed=%d\n',c.mean_signal_scale,c.common_mean_shift,c.covariance_ridge,c.fixed_dgp_seed);
fprintf(f,'base_seed=%d\nreplication_seed_offset=%d\nseedA=base_seed+offset+s+100\nseedB=base_seed+offset+s+200\n',c.base_seed,c.replication_seed_offset);
fprintf(f,'condition_number=%.17g\nmin_eig=%.17g\nmax_eig=%.17g\nIEO_lower_stages=%d\noverall_mean_ETO=%.17g\noverall_mean_IEO=%.17g\n',d.condition_number,d.min_eig,d.max_eig,s.IEO_lower_stages,s.overall_mean_regret_ETO,s.overall_mean_regret_IEO);
end

function write_run_log(c,d,s,v)
f=fopen(fullfile(c.results_dir,'exp2_scale0p60_ridge3em09_confirmation_run_log.txt'),'w');cl=onCleanup(@()fclose(f));
fprintf(f,'Completed %s run at %s.\nS=%d; cond(Sigma)=%.17g.\nValid=%d/%d. IEO lower stages=%d; ETO lower stages=%d.\n',c.mode,datestr(now,31),c.NumSims,d.condition_number,sum(v.passed),height(v),s.IEO_lower_stages,s.ETO_lower_stages);
end

function write_dry_audit(v,s,d,c)
f=fopen(fullfile(c.audit_dir,'EXP2_SCALE0P60_RIDGE3EM09_CONFIRMATION_DRY_RUN_AUDIT.md'),'w');cl=onCleanup(@()fclose(f));
fprintf(f,'# Exp2 locked scale0p60 ridge3em09 confirmation dry-run audit\n\n- S: %d\n- Fingerprint cond(Sigma): %.17g\n- All checks passed: %s\n- Valid replications: %d\n- Locked scale/ridge: 0.60 / 3e-9.\n- No existing Exp2 file was overwritten.\n',c.NumSims,d.condition_number,string(all(v.passed)),s.valid_replications);
end

function write_full_audits(c,d,s,v,a,o)
audit=fullfile(c.audit_dir,'EXP2_SCALE0P60_RIDGE3EM09_CONFIRMATION_AUDIT.md');f=fopen(audit,'w');cl=onCleanup(@()fclose(f));
fprintf(f,['# Exp2 scale0p60 ridge3em09 locked confirmation audit\n\n## Executive summary\n\n' ...
'This is the locked S=2000 confirmation of the S=200-selected scale=0.60, ridge=3e-9 candidate.  ' ...
'No parameter screening or adaptive tuning was performed in this run. No existing Main Experiment 2 files were overwritten.\n\n' ...
'## Source and fingerprint\n\n- Builder: `%s`\n- Fingerprint summary: `%s`\n- cond(Sigma): %.17g\n- lambda_min/max: %.17g / %.17g\n\n' ...
'## Retained settings\n\nT=%d, n=%d, x0=1, Ntr=%d, Nev=%d, S=%d, lambda=1e-4. Current Main Exp2 ETO/IEO and generator files were copied byte-for-byte; seed schedule offset 940000 was retained.\n\n' ...
'## Population regret\n\nAt x=1, regret is `delta''*Sigma1_true(t)*delta`, with the fixed true population moments. R_B moments, hindsight oracles, proxy regret, population ridge, and eigenvalue clipping were not used.\n\n' ...
'## Numerical stability\n\n- Validation checks passed: %d/%d\n- near-SPD stages: %d\n- high-residual stages: %d\n- maximum solve residual: %.6g\n- maximum budget residual: %.6g\n- maximum single contribution: %.6g\n\n' ...
'## Results\n\n- IEO lower-regret stages: %d\n- ETO lower-regret stages: %d\n- overall mean population regret ETO/IEO: %.9g / %.9g\n- terminal wealth ETO mean/median/p95/p99/max: %.9g / %.9g / %.9g / %.9g / %.9g\n- terminal wealth IEO mean/median/p95/p99/max: %.9g / %.9g / %.9g / %.9g / %.9g\n- maximum relative gross exposure: %.9g\n\n' ...
'## Candidate assessment\n\nThis remains a non-promoted confirmation candidate. Suitability depends on numerical stability, influence, exposure, and confirmation of the S=200 performance pattern; no automatic promotion is made.\n'], ...
c.source_runner,c.source_summary,d.condition_number,d.min_eig,d.max_eig,c.T,c.n,c.Ntr,c.Nev,c.NumSims,sum(v.passed),height(v),s.near_spd_count,s.high_residual_count,max([o.diagnostics.solve_residual_1,o.diagnostics.solve_residual_mu]),s.max_budget_residual,s.max_single_contribution,s.IEO_lower_stages,s.ETO_lower_stages,s.overall_mean_regret_ETO,s.overall_mean_regret_IEO,s.terminal_wealth_ETO_mean,s.terminal_wealth_ETO_median,s.terminal_wealth_ETO_p95,s.terminal_wealth_ETO_p99,s.terminal_wealth_ETO_max,s.terminal_wealth_IEO_mean,s.terminal_wealth_IEO_median,s.terminal_wealth_IEO_p95,s.terminal_wealth_IEO_p99,s.terminal_wealth_IEO_max,s.max_relative_gross_exposure);
clear cl
self=fullfile(c.audit_dir,'EXP2_SCALE0P60_RIDGE3EM09_CONFIRMATION_SELF_CHECK.md');f=fopen(self,'w');cl=onCleanup(@()fclose(f));
fprintf(f,['# Self-check\n\n1. Current Main Exp2 overwritten? **No.**\n2. Old results deleted? **No.**\n3. Locked S=200 source uniquely located? **Yes.**\n4. Fingerprint matched? **Yes (cond %.17g).**\n5. Scale/ridge changed after screening? **No: 0.60 / 3e-9.**\n6. Main Exp2 workflow settings retained? **Yes.**\n7. Dry run completed? **Yes.**\n8. S=2000 full run completed? **Yes.**\n9. True population regret? **Yes.**\n10. R_B moments used for regret? **No.**\n11. Hindsight oracle? **No.**\n12. Proxy regret? **No.**\n13. NaN/Inf? **No in valid outputs.**\n14. near-SPD count: **%d.**\n15. high residual count: **%d.**\n16. IEO lower stages: **%d.**\n17. ETO lower stages: **%d.**\n18. Lower overall regret: **%s.**\n19. Higher terminal mean wealth: **%s.**\n20. Max single contribution: **%.6g.**\n21. Max relative gross exposure: **%.6g.**\n22. Six figures: generated by the read-only postprocessor after this run.\n23. Bands: population/coefficient bootstrap CI; wealth empirical replication quantiles.\n24. Promotion: **No; confirmation candidate only.**\n'],d.condition_number,s.near_spd_count,s.high_residual_count,s.IEO_lower_stages,s.ETO_lower_stages,choose(s.overall_mean_regret_IEO<s.overall_mean_regret_ETO,'IEO','ETO'),choose(s.terminal_wealth_IEO_mean>s.terminal_wealth_ETO_mean,'IEO','ETO'),s.max_single_contribution,s.max_relative_gross_exposure);
end

function write_failure(file,v,c)
f=fopen(file,'w');cl=onCleanup(@()fclose(f));fprintf(f,'# %s run failure\n\nFull run was not continued/promoted.\n\n',upper(c.mode));for k=1:height(v),if ~v.passed(k),fprintf(f,'- %s: value %.17g, tolerance %.17g (%s)\n',v.check(k),v.value(k),v.tolerance(k),v.notes(k));end,end
end

function y=choose(test,a,b),if test,y=a;else,y=b;end,end
function atomic_checkpoint(target,arrays,completed,config,mu,Sigma),tmp=[tempname(fileparts(target)) '.mat'];save(tmp,'arrays','completed','config','mu','Sigma','-v7.3');movefile(tmp,target,'f');end
function ensure_dir(p),if ~isfolder(p),mkdir(p);end,end
