function outputs = run_exp2_main_experiment(mode)
%RUN_EXP2_MAIN_EXPERIMENT Formal Exp2 reproduction entry point.
% Formal Exp2 reproduction entry point for the main text and supplement.
% Mean-signal scale=0.60 and covariance ridge=3e-9 are fixed inputs.
% Population regret uses exact fixed population moments and unmodified Sigma1.

if nargin < 1, mode = 'dry'; end
mode = validatestring(lower(char(mode)), {'dry','full'});
paths = experiment_paths(mode);
config = formal_config(mode);
ensure_dir(paths.results_dir); ensure_dir(paths.table_dir); ensure_dir(paths.figure_dir);

old_path = path; cleanup_path = onCleanup(@() path(old_path));
old_dir = pwd; cleanup_dir = onCleanup(@() cd(old_dir));
cd(paths.code_dir);
addpath(paths.code_dir, '-begin');
verify_local_dependencies(paths.code_dir);
[mu, Sigma, sigma_diagnostics] = formal_dgp(config, paths.tables_dir);
oracle = exact_population_oracle(mu, Sigma, config);

write_dgp_tables(mu, Sigma, paths.table_dir);
arrays = initialize_arrays(config.NumSims, config);
completed = false(config.NumSims,1);
checkpoint = fullfile(paths.results_dir, sprintf('checkpoint_%s.mat',mode));
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
    fprintf('Exp2 formal scale0p60 ridge3em09 run (%s): %d/%d complete.\n',mode,sum(completed),config.NumSims);
end

assert(all(completed),'Formal Exp2 run incomplete.');
% Normalize checkpoint metadata to the current path-free formal config,
% including when a completed checkpoint is reused for verification.
atomic_checkpoint(checkpoint,arrays,completed,config,mu,Sigma);
validation=validate_run(arrays,oracle,Sigma,sigma_diagnostics,config);
if ~all(validation.passed)
    failure_file=fullfile(paths.results_dir,'validation_failed.csv');
    writetable(validation,failure_file);
    error('Exp2Formal:ValidationFailed','%s validation failed. See %s.',mode,failure_file);
end

[stagewise,wealth,coeff,summary] = ...
    aggregate_outputs(arrays,oracle,config);
if strcmp(mode,'dry')
    save(fullfile(paths.results_dir,'exp2_dry_raw.mat'), ...
        'arrays','completed','config','mu','Sigma','sigma_diagnostics','oracle','validation','summary','-v7.3');
    writetable(validation,fullfile(paths.results_dir,'validation.csv'));
    writetable(stagewise,fullfile(paths.results_dir,'stagewise_population_regret.csv'));
    outputs=struct('mode',mode,'validation',validation,'summary',summary);
    clear cleanup_path cleanup_dir; return
end

raw_file=fullfile(paths.results_dir,'exp2_scale0p60_ridge3em09_confirmation_raw.mat');
policy_form='u_t(x)=A_t*x+B_t; A is slope and B is intercept.';
population_regret_definition=['delta''*Sigma1_true(t)*delta at x=1; true fixed mu/Sigma; ' ...
    'no R_B moments, hindsight oracle, proxy regret, population ridge, or clipping.'];
save(raw_file,'arrays','completed','config','mu','Sigma','sigma_diagnostics', ...
    'oracle','validation','summary','policy_form', ...
    'population_regret_definition','-v7.3');
writetable(stagewise,fullfile(paths.table_dir,'exp2_scale0p60_ridge3em09_confirmation_stagewise_population_regret.csv'));
writetable(wealth,fullfile(paths.table_dir,'exp2_scale0p60_ridge3em09_confirmation_wealth.csv'));
writetable(coeff,fullfile(paths.table_dir,'exp2_scale0p60_ridge3em09_confirmation_coefficient_diagnostics.csv'));
writetable(validation,fullfile(paths.results_dir,'exp2_scale0p60_ridge3em09_confirmation_validation.csv'));
create_main_figures(stagewise,wealth,coeff,arrays,config,paths.figure_dir);
outputs=struct('mode',mode,'raw',raw_file,'stagewise',stagewise,'wealth',wealth, ...
    'coeff',coeff,'summary',summary,'validation',validation);
clear cleanup_path cleanup_dir
end

function p=experiment_paths(mode)
p.code_dir=fileparts(mfilename('fullpath'));
p.experiment_dir=fileparts(p.code_dir);
p.results_dir=fullfile(p.experiment_dir,'results',mode);
p.figure_dir=fullfile(p.experiment_dir,'figures');
p.tables_dir=fullfile(p.experiment_dir,'tables');
if strcmp(mode,'full'), p.table_dir=p.tables_dir; else, p.table_dir=p.results_dir; end
end

function c=formal_config(mode)
c=struct(); c.mode=mode;
c.T=50; c.n=10; c.Ntr=60; c.Nev=1000; c.x0=1; c.x_eval=1;
c.lambda_scalar=1e-4; c.lambda=c.lambda_scalar*ones(1,c.T);
c.base_seed=42; c.replication_seed_offset=940000; c.fixed_dgp_seed=11902;
c.mean_signal_scale=0.60; c.common_mean_shift=1.8e-4; c.covariance_ridge=3e-9;
c.expected_condition=195246.156950;
c.tol_spd_multiplier=1e-10; c.high_residual_threshold=1e-6;
c.negative_regret_tolerance=1e-10; c.oracle_self_regret_tolerance=1e-10;
c.budget_tolerance=1e-8; c.crosscheck_tolerance=1e-6;
c.bootstrap_seed=20260915; c.bootstrap_resamples=2000;
% The accepted method-specific population-regret bands were re-exported
% with the common paper-figure PCG64 stream below. Keep this post-processing
% stream separate from the paired-difference and coefficient diagnostics.
c.population_regret_band_seed=20260914;
c.population_regret_band_resamples=2000;
c.parallel_workers=4; c.batch_size=10; c.use_parallel=true;
if strcmp(mode,'dry')
    c.NumSims=10; c.use_parallel=false; c.batch_size=2;
else
    c.NumSims=2000;
end
end

function verify_local_dependencies(code_dir)
names={'run_eto.m','run_ieo.m','generate_data.m', ...
    'numpy_pcg64_bootstrap_mean_ci.m'};
for k=1:numel(names)
    assert(isfile(fullfile(code_dir,names{k})),'Required local core is missing: %s',names{k});
end
clear run_eto run_ieo generate_data numpy_pcg64_bootstrap_mean_ci
rehash path
assert(strcmp(which('run_eto'),fullfile(code_dir,'run_eto.m')));
assert(strcmp(which('run_ieo'),fullfile(code_dir,'run_ieo.m')));
assert(strcmp(which('generate_data'),fullfile(code_dir,'generate_data.m')));
assert(strcmp(which('numpy_pcg64_bootstrap_mean_ci'), ...
    fullfile(code_dir,'numpy_pcg64_bootstrap_mean_ci.m')));
end

function [mu,Sigma,d]=formal_dgp(c,tables_dir)
% Reconstruct the fixed population used by the formal Exp2 results.
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
assert(chol_flag==0,'Unable to reconstruct the formal Exp2 population.');
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
assert(abs(d.condition_number-c.expected_condition)/c.expected_condition<1e-9,'Formal DGP condition fingerprint mismatch.');
packaged_mu_file=fullfile(tables_dir,'exp2_scale0p60_ridge3em09_confirmation_mu.csv');
packaged_sigma_file=fullfile(tables_dir,'exp2_scale0p60_ridge3em09_confirmation_Sigma.csv');
assert(isfile(packaged_mu_file) && isfile(packaged_sigma_file), ...
    'Packaged Exp2 DGP tables are missing.');
packaged_mu=readmatrix(packaged_mu_file);
packaged_sigma=readmatrix(packaged_sigma_file);
assert(max(abs(packaged_mu(:)-mu(:)))<1e-12 && max(abs(packaged_sigma(:)-Sigma(:)))<1e-12, ...
    'Reconstructed DGP differs from the packaged DGP tables.');
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
    if min(ev)<-tol, error('Exp2Formal:TrueSigma1NotSPD','stage %d eig %.17g tol %.17g',t-1,min(ev),tol); end
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
            error('Exp2Formal:AlternativeSolverMismatch','stage %d alternative solver mismatch',t-1);
        end
    end
    if alt_diff>c.crosscheck_tolerance, error('Exp2Formal:OracleMismatch','stage %d KKT mismatch',t-1); end
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
add('formal_dgp_condition_fingerprint',d.condition_number,c.expected_condition,abs(d.condition_number-c.expected_condition)/c.expected_condition<1e-9,'cond(Sigma) fingerprint');
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

function [stagewise,wealth,coeff,summary]=aggregate_outputs(a,o,c)
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
te=mean(a.Terminal_Wealth_ETO(valid,:),2); ti=mean(a.Terminal_Wealth_IEO(valid,:),2);
summary=struct('valid_replications',S,'IEO_lower_stages',sum(mean(I)<mean(E)),'ETO_lower_stages',sum(mean(E)<mean(I)), ...
    'overall_mean_regret_ETO',mean(E,'all'),'overall_mean_regret_IEO',mean(I,'all'), ...
    'terminal_wealth_ETO_mean',mean(te),'terminal_wealth_ETO_median',median(te),'terminal_wealth_ETO_p95',quantile(te,.95), ...
    'terminal_wealth_ETO_p99',quantile(te,.99),'terminal_wealth_ETO_max',max(te), ...
    'terminal_wealth_IEO_mean',mean(ti),'terminal_wealth_IEO_median',median(ti),'terminal_wealth_IEO_p95',quantile(ti,.95), ...
    'terminal_wealth_IEO_p99',quantile(ti,.99),'terminal_wealth_IEO_max',max(ti), ...
    'max_single_contribution',max([stagewise.max_contribution_to_mean_eto;stagewise.max_contribution_to_mean_ieo]), ...
    'max_relative_gross_exposure',max([a.max_relative_gross_ETO(:);a.max_relative_gross_IEO(:)]), ...
    'near_spd_count',sum(o.diagnostics.near_spd_flag),'high_residual_count',sum(o.diagnostics.high_residual_flag), ...
    'max_regret',max([E(:);I(:)]),'max_budget_residual',max([a.max_budget_eto;a.max_budget_ieo]));
end

function tbl=coefficient_table(a,v,c)
names={'intercept_gap','slope_gap','intercept_angle','slope_angle'}; fields={'intercept_gap_sims','slope_gap_sims','intercept_angle_sims','slope_angle_sims'};
rows=cell(numel(names)*c.T,9); r=0;
for k=1:numel(names),X=a.(fields{k})(v,:);[lo,hi]=mean_bootstrap_ci(X,c.bootstrap_resamples,c.bootstrap_seed+10*k);
 for t=1:c.T,r=r+1;rows(r,:)={names{k},t-1,mean(X(:,t),'omitnan'),std(X(:,t),'omitnan')/sqrt(sum(isfinite(X(:,t)))),median(X(:,t),'omitnan'),quantile(X(:,t),.95),quantile(X(:,t),.99),lo(t),hi(t)};end,end
tbl=cell2table(rows,'VariableNames',{'statistic_name','stage_t','mean','se','median','p95','p99','bootstrap_ci_low','bootstrap_ci_high'});
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

function create_main_figures(stagewise,wealth,coeff,arrays,config,figure_dir)
blue=[0 0.4470 0.7410]; red=[0.8500 0.3250 0.0980];
f=figure('Visible','off','Color','w','Position',[100 100 1031 770]);
ax=axes('Parent',f); hold(ax,'on');
[band,band_audit]=approved_population_regret_band(arrays,config);
assert(max(abs(band.eto_mean-stagewise.regret_pop_eto_mean))<1e-14);
assert(max(abs(band.ieo_mean-stagewise.regret_pop_ieo_mean))<1e-14);
patch(ax,[stagewise.stage_t;flipud(stagewise.stage_t)], ...
    [band.eto_high;flipud(band.eto_low)],blue,'FaceAlpha',0.18, ...
    'EdgeColor','none','HandleVisibility','off');
patch(ax,[stagewise.stage_t;flipud(stagewise.stage_t)], ...
    [band.ieo_high;flipud(band.ieo_low)],red,'FaceAlpha',0.18, ...
    'EdgeColor','none','HandleVisibility','off');
eto_line=plot(ax,stagewise.stage_t,band.eto_mean,'--','Color',blue, ...
    'LineWidth',2.5,'DisplayName','ETO mean');
ieo_line=plot(ax,stagewise.stage_t,band.ieo_mean,'-','Color',red, ...
    'LineWidth',2.5,'DisplayName','IEO mean');
band_legend=patch(ax,nan,nan,[0.5 0.5 0.5],'FaceAlpha',0.22, ...
    'EdgeColor','none','DisplayName','Pointwise 95% CI');
xlabel(ax,'Stage (t)'); ylabel(ax,'Population regret'); grid(ax,'on'); box(ax,'on');
xlim(ax,[0 config.T-1]); legend(ax,[eto_line,ieo_line,band_legend],'Location','northeast');
all_limits=[band.eto_low;band.eto_high;band.ieo_low;band.ieo_high];
limit_range=max(all_limits)-min(all_limits);
ylim(ax,[min(all_limits)-0.05*limit_range,max(all_limits)+0.05*limit_range]);
set(ax,'FontName','Helvetica','FontSize',11,'LineWidth',0.8);
setappdata(f,'population_regret_band_audit',band_audit);
exportgraphics(f,fullfile(figure_dir,'exp2_population_regret.png'),'Resolution',300);
exportgraphics(f,fullfile(figure_dir,'exp2_population_regret.pdf'),'ContentType','vector'); close(f);

f=figure('Visible','off','Color','w','Position',[100 100 700 460]);
x=wealth.wealth_stage';
patch([x fliplr(x)],[wealth.wealth_eto_q025' fliplr(wealth.wealth_eto_q975')],blue, ...
    'FaceAlpha',0.16,'EdgeColor','none','HandleVisibility','off'); hold on;
patch([x fliplr(x)],[wealth.wealth_ieo_q025' fliplr(wealth.wealth_ieo_q975')],red, ...
    'FaceAlpha',0.16,'EdgeColor','none','HandleVisibility','off');
plot(x,wealth.wealth_eto_mean','--','Color',blue,'LineWidth',2,'DisplayName','ETO');
plot(x,wealth.wealth_ieo_mean','-','Color',red,'LineWidth',2,'DisplayName','IEO');
xlabel('Stage t'); ylabel('Out-of-sample wealth'); grid on; box on; legend('Location','best');
set(gca,'FontName','Helvetica','FontSize',11,'LineWidth',0.8);
exportgraphics(f,fullfile(figure_dir,'exp2_wealth.png'),'Resolution',300);
exportgraphics(f,fullfile(figure_dir,'exp2_wealth.pdf'),'ContentType','vector'); close(f);

create_coefficient_figures(coeff,figure_dir);
end

function [band,audit]=approved_population_regret_band(arrays,config)
valid=arrays.valid; E=arrays.Regret_ETO_pop_sims(valid,:); I=arrays.Regret_IEO_pop_sims(valid,:);
[lo,hi,center,audit]=numpy_pcg64_bootstrap_mean_ci([E I], ...
    config.population_regret_band_resamples,config.population_regret_band_seed);
T=config.T;
band=struct('eto_mean',center(1:T)','ieo_mean',center(T+1:2*T)', ...
    'eto_low',lo(1:T)','eto_high',hi(1:T)', ...
    'ieo_low',lo(T+1:2*T)','ieo_high',hi(T+1:2*T)');
audit.definition=['method-specific Monte Carlo mean; complete replication-row ' ...
    'resampling; pointwise 2.5%/97.5% percentile-bootstrap CI'];
end

function create_coefficient_figures(tbl,figure_dir)
% The four formal coefficient panels are generated from the same table that
% is saved by this runner. Bands are pointwise percentile-bootstrap CIs for
% the Monte Carlo mean, not mean +/- 1.96 SE and not simultaneous bands.
names={'intercept_gap','slope_gap','intercept_angle','slope_angle'};
ylabs={'Intercept gap','Slope gap','Intercept angle (degrees)','Slope angle (degrees)'};
blue=[0 0.4470 0.7410];
for k=1:numel(names)
    rows=strcmp(string(tbl.statistic_name),names{k}); x=tbl.stage_t(rows)';
    mu=tbl.mean(rows)'; lo=tbl.bootstrap_ci_low(rows)'; hi=tbl.bootstrap_ci_high(rows)';
    f=figure('Visible','off','Color','w','Position',[100 100 700 460]);
    patch([x fliplr(x)],[lo fliplr(hi)],blue,'FaceAlpha',0.18, ...
        'EdgeColor','none','DisplayName','Pointwise 95% percentile-bootstrap CI'); hold on;
    plot(x,mu,'-','Color',blue,'LineWidth',2.0,'DisplayName','Monte Carlo mean');
    xlabel('Stage t'); ylabel(ylabs{k}); grid on; box on;
    set(gca,'FontName','Helvetica','FontSize',11,'LineWidth',0.8);
    legend('Location','best');
    exportgraphics(f,fullfile(figure_dir,['exp2_' names{k} '.png']),'Resolution',300);
    exportgraphics(f,fullfile(figure_dir,['exp2_' names{k} '.pdf']),'ContentType','vector');
    close(f);
end
end

function write_dgp_tables(mu,Sigma,table_dir)
writematrix(mu,fullfile(table_dir,'exp2_scale0p60_ridge3em09_confirmation_mu.csv'));
writematrix(Sigma,fullfile(table_dir,'exp2_scale0p60_ridge3em09_confirmation_Sigma.csv'));
end

function atomic_checkpoint(target,arrays,completed,config,mu,Sigma),tmp=[tempname(fileparts(target)) '.mat'];save(tmp,'arrays','completed','config','mu','Sigma','-v7.3');movefile(tmp,target,'f');end
function ensure_dir(p),if ~isfolder(p),mkdir(p);end,end
