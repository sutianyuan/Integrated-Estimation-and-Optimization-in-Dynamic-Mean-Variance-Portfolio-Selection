function [population,config]=build_block_gaussian_population(experiment_dir)
%BUILD_BLOCK_GAUSSIAN_POPULATION 
% Five-asset equicorrelation blocks have condition number 20. 


if nargin<1,experiment_dir=fileparts(mfilename('fullpath'));end
addpath(fullfile(experiment_dir,'dependencies'));
config=struct('experiment_dir',experiment_dir,'n_grid',5:5:50,'T',50, ...
    'J_grid',300*ones(size(5:5:50)),'J_max',300,'n_max',50,'N_B',1000, ...
    'factor_count_grid',(5:5:50)/5,'factor_count_rule','K(n)=n/5', ...
    'lambda_scalar',1e-4,'lambda',1e-4*ones(1,50), ...
    'target_variance',8.103335969935516e-5,'target_condition_number',20, ...
    'mu_bar',1.0002370503416058,'oracle_l1_target',1.15,'x0',1, ...
    'base_seed',42,'bootstrap_resamples',5000,'bootstrap_seed',271828, ...
    'parallel_workers',10,'batch_size',10,'budget_tolerance',1e-8, ...
    'negative_regret_tolerance',1e-10);

rho=(config.target_condition_number-1)/(config.target_condition_number+4);
block=config.target_variance*((1-rho)*eye(5)+rho*ones(5));
Sigma_max=kron(eye(config.n_max/5),block);
direction=zeros(config.n_max,1);
direction(1:5)=[-2;-1;0;1;2]/sqrt(10);
population=struct([]);
for nn=1:numel(config.n_grid)
    n=config.n_grid(nn);Sigma=Sigma_max(1:n,1:n);d=direction(1:n);
    delta=calibrate_delta(Sigma,d,config);
    mu=config.mu_bar*ones(n,1)+delta*d;
    record=make_record(Sigma,mu,delta,config);
    if isempty(population)
        population=repmat(record,1,numel(config.n_grid));
    else
        population(nn)=record;
    end
end
assert(all(abs([population.condition_number]-20)<1e-10));
assert(all(abs([population.mean_marginal_variance]-config.target_variance)<1e-16));
assert(all(abs([population.oracle_l1_stage0]-config.oracle_l1_target)<1e-8));
end

function delta=calibrate_delta(Sigma,direction,config)
objective=@(x) oracle_l1(Sigma,config.mu_bar*ones(size(direction))+x*direction,config)-config.oracle_l1_target;
lower=0;upper=0.0007109728400377762;
while objective(upper)<0&&upper<1,upper=2*upper;end
assert(objective(upper)>=0,'Unable to bracket fixed-L1 signal calibration.');
for iteration=1:70
    middle=.5*(lower+upper);
    if objective(middle)>=0,upper=middle;else,lower=middle;end
end
delta=.5*(lower+upper);assert(abs(objective(delta))<=1e-8);
end

function value=oracle_l1(Sigma,mu,config)
n=numel(mu);strategy=calculate_oracle(repmat(mu',config.T,1), ...
    repmat(reshape(Sigma,1,n,n),config.T,1,1),config.lambda);
value=sum(abs(strategy.A_coef(:,1)+strategy.B_coef(:,1)));
end

function p=make_record(Sigma,mu,delta,config)
n=numel(mu);mu_path=repmat(mu',config.T,1);
Sigma_path=repmat(reshape(Sigma,1,n,n),config.T,1,1);
[oracle,debug]=calculate_oracle(mu_path,Sigma_path,config.lambda);
p=struct('n',n,'Sigma',Sigma,'mu',mu,'mu_path',mu_path, ...
    'Sigma_path',Sigma_path,'oracle',oracle,'oracle_debug',debug, ...
    'calibrated_delta',delta,'condition_number',cond(Sigma), ...
    'mean_marginal_variance',mean(diag(Sigma)), ...
    'oracle_l1_stage0',sum(abs(oracle.A_coef(:,1)+oracle.B_coef(:,1))), ...
    'cholesky',chol(Sigma,'lower'));
end
