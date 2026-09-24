function outputs=run_block_gaussian_j300_variable_k_mc2000_wealth(mode)
%RUN_BLOCK_GAUSSIAN_J300_VARIABLE_K_MC2000_WEALTH Block-DGP MC2000.


if nargin<1,mode='smoke';end
mode=lower(char(mode));assert(ismember(mode,{'smoke','mc2000'}));
here=fileparts(mfilename('fullpath'));
module_dir=fileparts(here);
project_root=fileparts(fileparts(module_dir));
original_path=path;
path_cleanup=onCleanup(@()path(original_path));
addpath(here,fullfile(here,'dependencies'),'-begin');
[population,config]=build_block_gaussian_population(here);
config.N_B=1000;config.x0=1;
switch mode
    case 'smoke'
        config.NumSims=2;config.replication_seed_offset=495000;
    case 'mc2000'
        config.NumSims=2000;config.replication_seed_offset=500000;
end
config.mode=mode;config.parallel_workers=min(10,config.NumSims);
config.batch_size=min(10,config.NumSims);
config.wealth_definition=['Main-experiment own recursive Universe-B path: ', ...
    'u_t=A_t X_t+B_t; X_{t+1}=R_t''u_t; X_0=1; N_B=1000.'];
config.wealth_ci_definition='mean +/- 1.96*MC standard deviation/sqrt(valid replications)';
if strcmp(mode,'mc2000')
    run_dir='full_run_candidate';
else
    run_dir='dry_run';
end
config.results_dir=fullfile(project_root,'outputs','sensitivity_asset_dimension',run_dir);
if ~exist(config.results_dir,'dir'),mkdir(config.results_dir);end
validate_design(population,config);

N=numel(config.n_grid);S=config.NumSims;T=config.T;
data.regret_eto=nan(N,S,T);data.regret_ieo=nan(N,S,T);
data.wealth_eto=nan(N,S,T+1);data.wealth_ieo=nan(N,S,T+1);
data.l1_eto=nan(N,S);data.l1_ieo=nan(N,S);
data.max_budget_eto=nan(N,S);data.max_budget_ieo=nan(N,S);
data.max_space_residual=nan(N,S);data.max_space_dimension=nan(N,S);
data.fallbacks=nan(N,S,3);data.valid=false(N,S);
data.error_message=strings(N,S);completed=false(1,S);
checkpoint=fullfile(tempdir,sprintf('ieo_asset_dimension_%s_checkpoint.mat',mode));
if exist(checkpoint,'file')
    saved=load(checkpoint,'data','completed','config','population');
    validate_checkpoint(saved,config,population);
    data=saved.data;completed=saved.completed;
end

pool=gcp('nocreate');
if isempty(pool),parpool('local',config.parallel_workers);
elseif pool.NumWorkers~=config.parallel_workers
    delete(pool);parpool('local',config.parallel_workers);
end
remaining=find(~completed);
for first=1:config.batch_size:numel(remaining)
    ids=remaining(first:min(first+config.batch_size-1,numel(remaining)));
    blocks=cell(numel(ids),1);
    parfor bb=1:numel(ids)
        blocks{bb}=one_replication(ids(bb),population,config);
    end
    for bb=1:numel(ids)
        s=ids(bb);one=blocks{bb};
        data.regret_eto(:,s,:)=reshape(one.regret_eto,N,1,T);
        data.regret_ieo(:,s,:)=reshape(one.regret_ieo,N,1,T);
        data.wealth_eto(:,s,:)=reshape(one.wealth_eto,N,1,T+1);
        data.wealth_ieo(:,s,:)=reshape(one.wealth_ieo,N,1,T+1);
        data.l1_eto(:,s)=one.l1_eto;data.l1_ieo(:,s)=one.l1_ieo;
        data.max_budget_eto(:,s)=one.max_budget_eto;
        data.max_budget_ieo(:,s)=one.max_budget_ieo;
        data.max_space_residual(:,s)=one.max_space_residual;
        data.max_space_dimension(:,s)=one.max_space_dimension;
        data.fallbacks(:,s,:)=reshape(one.fallbacks,N,1,3);
        data.valid(:,s)=one.valid;data.error_message(:,s)=one.error_message;
        completed(s)=true;
    end
    save(checkpoint,'population','config','data','completed','-v7.3');
    fprintf('Block Gaussian K(n)=n/5 J=300 MC2000 wealth %s: %d/%d complete.\n', ...
        mode,sum(completed),S);
end

[summary,~,~,~,~,validation]= ...
    aggregate_outputs(data,population,config,completed);
assert(all(validation.passed),'AssetDimension:ValidationFailed', ...
    'Asset-dimension simulation failed an internal consistency check.');
paper_table=make_paper_table(summary);
validate_paper_table(paper_table,config);
outputs=struct('mode',mode,'simulation_executed',true, ...
    'paper_table',fullfile(config.results_dir, ...
    'asset_dimension_growing_block_table.csv'),'rows',height(paper_table));
writetable(paper_table,outputs.paper_table);
if exist(checkpoint,'file'),delete(checkpoint);end
disp(paper_table);
clear path_cleanup
end

function paper_table=make_paper_table(summary)
paper_table=table(summary.n,1e6*summary.eto_mean,1e6*summary.ieo_mean, ...
    1e6*summary.eto_minus_ieo_mean,1e6*summary.ci_lower, ...
    1e6*summary.ci_upper,'VariableNames', ...
    {'n','R0_ETO_star','R0_IEO_star','Delta_R','CI_low','CI_high'});
end

function validate_paper_table(paper_table,config)
assert(height(paper_table)==10&&isequal(paper_table.n',config.n_grid), ...
    'AssetDimension:PaperGrid','Paper table must contain the locked 10-point grid.');
assert(max(abs(paper_table.Delta_R-(paper_table.R0_ETO_star- ...
    paper_table.R0_IEO_star)))<1e-12,'AssetDimension:DeltaDirection', ...
    'Delta_R must equal R0_ETO_star minus R0_IEO_star.');
assert(all(paper_table.CI_low<=paper_table.Delta_R& ...
    paper_table.Delta_R<=paper_table.CI_high), ...
    'AssetDimension:CIOrder','Delta_R must lie inside each ordered interval.');
end

function one=one_replication(replication_id,population,config)
N=numel(config.n_grid);T=config.T;
one.regret_eto=nan(N,T);one.regret_ieo=nan(N,T);
one.wealth_eto=nan(N,T+1);one.wealth_ieo=nan(N,T+1);
one.l1_eto=nan(N,1);one.l1_ieo=nan(N,1);
one.max_budget_eto=nan(N,1);one.max_budget_ieo=nan(N,1);
one.max_space_residual=nan(N,1);one.max_space_dimension=nan(N,1);
one.fallbacks=nan(N,3);one.valid=false(N,1);one.error_message=strings(N,1);
seed=config.base_seed+config.replication_seed_offset+replication_id;
saved=rng;cleanup=onCleanup(@()rng(saved));
rng(seed+100,'twister');z_A=randn(max(config.n_grid),config.J_max,T);
rng(seed+200,'twister');z_B=randn(max(config.n_grid),config.N_B,T);
for nn=1:N
    n=config.n_grid(nn);J=config.J_grid(nn);K=config.factor_count_grid(nn);
    pop=population(nn);
    try
        R_A=pop.mu+pagemtimes(pop.cholesky,z_A(1:n,1:J,:));
        R_B=pop.mu+pagemtimes(pop.cholesky,z_B(1:n,:,:));
        Nstage=J*ones(1,T);
        eto=run_eto_factor_cov(R_A,config.lambda,Nstage,K);
        ieo=run_structured_ieo_bridge(R_A,config.lambda,Nstage, ...
            K,'unanchored',[]);
        truth=pop.oracle.A_coef+pop.oracle.B_coef;
        ue=eto.strategy.A_coef+eto.strategy.B_coef;
        ui=ieo.strategy.A_coef+ieo.strategy.B_coef;
        for t=1:T
            H=squeeze(pop.oracle_debug.Sigma_t1(:,:,t));
            de=ue(:,t)-truth(:,t);di=ui(:,t)-truth(:,t);
            one.regret_eto(nn,t)=de'*H*de;
            one.regret_ieo(nn,t)=di'*H*di;
        end
        [one.wealth_eto(nn,:),~,one.max_budget_eto(nn)]= ...
            evaluate_recursive_wealth(R_B,eto.strategy,config.x0);
        [one.wealth_ieo(nn,:),~,one.max_budget_ieo(nn)]= ...
            evaluate_recursive_wealth(R_B,ieo.strategy,config.x0);
        one.l1_eto(nn)=sum(abs(ue(:,1)));one.l1_ieo(nn)=sum(abs(ui(:,1)));
        one.max_space_residual(nn)=max([ieo.factor_policy.policy_space_residual_A; ...
            ieo.factor_policy.policy_space_residual_B]);
        one.max_space_dimension(nn)=max(ieo.factor_policy.dimension);
        one.fallbacks(nn,:)=[ieo.validation.fallback_qp_count, ...
            ieo.validation.fallback_benchmark_count,ieo.validation.fallback_value_count];
        all_values=[one.regret_eto(nn,:),one.regret_ieo(nn,:), ...
            one.wealth_eto(nn,:),one.wealth_ieo(nn,:)];
        one.valid(nn)=all(isfinite(all_values))&& ...
            min([one.regret_eto(nn,:),one.regret_ieo(nn,:)])>= ...
            -config.negative_regret_tolerance&& ...
            max(one.max_budget_eto(nn),one.max_budget_ieo(nn))<=config.budget_tolerance;
        if ~one.valid(nn),one.error_message(nn)="invalid_diagnostic";end
    catch exception
        one.error_message(nn)=string(getReport(exception,'basic','hyperlinks','off'));
    end
end
end

function validate_design(population,config)
assert(isequal(config.n_grid,5:5:50));
assert(isequal(config.J_grid,300*ones(size(config.n_grid))));
assert(config.T==50&&config.N_B==1000&&config.x0==1);
assert(isequal(config.factor_count_grid,config.n_grid/5));
assert(all(abs([population.condition_number]-20)<1e-10));
assert(all(abs([population.oracle_l1_stage0]-1.15)<1e-8));
end

function validate_checkpoint(saved,config,population)
assert(saved.config.NumSims==config.NumSims);
assert(saved.config.replication_seed_offset==config.replication_seed_offset);
assert(saved.config.N_B==config.N_B);
assert(isequal(saved.config.J_grid,config.J_grid));
assert(isequal(saved.config.factor_count_grid,config.factor_count_grid));
assert(isequal(saved.config.n_grid,config.n_grid));
for nn=1:numel(population)
    assert(isequal(saved.population(nn).Sigma,population(nn).Sigma));
    assert(isequal(saved.population(nn).mu,population(nn).mu));
end
end

function [summary,regret_paths,wealth_paths,terminal,stagewise,validation]= ...
        aggregate_outputs(data,population,config,completed)
N=numel(config.n_grid);summary_rows=repmat(stage0_template(),N,1);
terminal_rows=repmat(terminal_template(),N,1);
stage_rows=repmat(stage_template(),N,1);
regret_rows=repmat(path_template('regret'),N*config.T,1);
wealth_rows=repmat(path_template('wealth'),N*(config.T+1),1);
rr=0;wr=0;
for nn=1:N
    valid=data.valid(nn,:)';assert(any(valid));nv=sum(valid);
    er=squeeze(data.regret_eto(nn,valid,:));ir=squeeze(data.regret_ieo(nn,valid,:));
    d0=er(:,1)-ir(:,1);ci=paired_bootstrap(d0,config.bootstrap_resamples, ...
        config.bootstrap_seed+11000+nn);
    K=config.factor_count_grid(nn);
    stage_means=mean(er,1)-mean(ir,1);
    stage_rows(nn)=struct('Design',"Block Gaussian variable K", ...
        'n',config.n_grid(nn),'J',config.J_grid(nn),'K',K, ...
        'number_of_stages_with_lower_ETO_mean',sum(stage_means<0), ...
        'number_of_stages_with_lower_IEO_mean',sum(stage_means>0), ...
        'number_of_tied_stages',sum(stage_means==0));
    summary_rows(nn)=struct('n',config.n_grid(nn),'J',config.J_grid(nn),'K',K, ...
        'attempted_replications',config.NumSims,'valid_replications',nv, ...
        'eto_mean',mean(er(:,1)),'ieo_mean',mean(ir(:,1)), ...
        'eto_minus_ieo_mean',mean(d0),'ci_lower',ci(1),'ci_upper',ci(2), ...
        'eto_win_rate',mean(d0<0),'eto_q95',quantile(er(:,1),.95), ...
        'ieo_q95',quantile(ir(:,1),.95),'eto_l1_median',median(data.l1_eto(nn,valid)), ...
        'ieo_l1_median',median(data.l1_ieo(nn,valid)), ...
        'population_condition_number',population(nn).condition_number, ...
        'oracle_l1',population(nn).oracle_l1_stage0);
    for t=1:config.T
        rr=rr+1;regret_rows(rr)=make_path_row(config.n_grid(nn),config.J_grid(nn), ...
            K,t-1,nv,er(:,t),ir(:,t));
    end
    ew=squeeze(data.wealth_eto(nn,valid,:));iw=squeeze(data.wealth_ieo(nn,valid,:));
    for t=1:config.T+1
        wr=wr+1;wealth_rows(wr)=make_path_row(config.n_grid(nn),config.J_grid(nn), ...
            K,t-1,nv,ew(:,t),iw(:,t));
    end
    e=ew(:,end);i=iw(:,end);d=i-e;half=1.96*std(d)/sqrt(nv);
    terminal_rows(nn)=struct('n',config.n_grid(nn),'J',config.J_grid(nn),'K',K, ...
        'attempted_replications',config.NumSims,'valid_replications',nv, ...
        'failure_rate',1-nv/config.NumSims,'eto_mean',mean(e), ...
        'eto_mc_se',std(e)/sqrt(nv),'eto_median',median(e), ...
        'ieo_mean',mean(i),'ieo_mc_se',std(i)/sqrt(nv),'ieo_median',median(i), ...
        'ieo_minus_eto_mean',mean(d),'ieo_minus_eto_mc_se',std(d)/sqrt(nv), ...
        'ieo_minus_eto_ci_lower',mean(d)-half,'ieo_minus_eto_ci_upper',mean(d)+half, ...
        'ieo_terminal_win_rate',mean(d>0));
end
summary=struct2table(summary_rows);regret_paths=struct2table(regret_rows);
wealth_paths=struct2table(wealth_rows);terminal=struct2table(terminal_rows);
stagewise=struct2table(stage_rows);
checks=["completed";"all_cells_valid";"unique_seeds";"initial_wealth_eto"; ...
    "initial_wealth_ieo";"budget_eto";"budget_ieo";"space_residual"; ...
    "space_dimension";"fallbacks";"population_condition";"oracle_l1_calibration"];
values=[sum(completed);sum(data.valid,'all'); ...
    numel(unique(config.base_seed+config.replication_seed_offset+(1:config.NumSims))); ...
    max(abs(data.wealth_eto(:,:,1)-config.x0),[],'all','omitnan'); ...
    max(abs(data.wealth_ieo(:,:,1)-config.x0),[],'all','omitnan'); ...
    max(data.max_budget_eto,[],'all','omitnan');max(data.max_budget_ieo,[],'all','omitnan'); ...
    max(data.max_space_residual,[],'all','omitnan'); ...
    max(data.max_space_dimension,[],'all','omitnan');sum(data.fallbacks,'all','omitnan'); ...
    max(abs([population.condition_number]-20)); ...
    max(abs([population.oracle_l1_stage0]-1.15))];
tolerances=[config.NumSims;numel(data.valid);config.NumSims;0;0;1e-8;1e-8; ...
    1e-8;max(config.factor_count_grid)+2;0;1e-14;1e-8];
passed=[all(completed);all(data.valid,'all');values(3)==config.NumSims; ...
    values(4)==0;values(5)==0;values(6)<=1e-8;values(7)<=1e-8; ...
    values(8)<=1e-8;values(9)<=max(config.factor_count_grid)+2; ...
    values(10)==0;values(11)<=1e-14;values(12)<=1e-8];
validation=table(checks,values,tolerances,passed);
end

function row=stage0_template()
row=struct('n',[],'J',[],'K',[],'attempted_replications',[], ...
    'valid_replications',[],'eto_mean',[],'ieo_mean',[], ...
    'eto_minus_ieo_mean',[],'ci_lower',[],'ci_upper',[],'eto_win_rate',[], ...
    'eto_q95',[],'ieo_q95',[],'eto_l1_median',[],'ieo_l1_median',[], ...
    'population_condition_number',[],'oracle_l1',[]);
end

function row=terminal_template()
row=struct('n',[],'J',[],'K',[],'attempted_replications',[],'valid_replications',[], ...
    'failure_rate',[],'eto_mean',[],'eto_mc_se',[],'eto_median',[], ...
    'ieo_mean',[],'ieo_mc_se',[],'ieo_median',[],'ieo_minus_eto_mean',[], ...
    'ieo_minus_eto_mc_se',[],'ieo_minus_eto_ci_lower',[], ...
    'ieo_minus_eto_ci_upper',[],'ieo_terminal_win_rate',[]);
end

function row=stage_template()
row=struct('Design',string.empty,'n',[],'J',[],'K',[], ...
    'number_of_stages_with_lower_ETO_mean',[], ...
    'number_of_stages_with_lower_IEO_mean',[],'number_of_tied_stages',[]);
end

function row=path_template(~)
row=struct('n',[],'J',[],'K',[],'stage',[],'valid_replications',[], ...
    'eto_mean',[],'eto_ci_lower',[],'eto_ci_upper',[], ...
    'ieo_mean',[],'ieo_ci_lower',[],'ieo_ci_upper',[]);
end

function row=make_path_row(n,J,K,t,nv,e,i)
eh=1.96*std(e)/sqrt(nv);ih=1.96*std(i)/sqrt(nv);
row=struct('n',n,'J',J,'K',K,'stage',t,'valid_replications',nv, ...
    'eto_mean',mean(e),'eto_ci_lower',mean(e)-eh,'eto_ci_upper',mean(e)+eh, ...
    'ieo_mean',mean(i),'ieo_ci_lower',mean(i)-ih,'ieo_ci_upper',mean(i)+ih);
end

function ci=paired_bootstrap(x,B,seed)
saved=rng;cleanup=onCleanup(@()rng(saved));
rng(seed,'twister');x=x(:);N=numel(x);means=zeros(B,1);
for first=1:500:B
    ids=first:min(first+499,B);ix=randi(N,N,numel(ids));
    means(ids)=mean(x(ix),1)';
end
ci=quantile(means,[.025,.975]);
end
