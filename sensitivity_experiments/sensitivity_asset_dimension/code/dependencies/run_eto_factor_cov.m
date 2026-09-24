function results = run_eto_factor_cov(R_A, lambda, N_by_stage, factor_count)
%RUN_ETO_FACTOR_COV ETO formulas with a locked PCA-factor covariance input.
% Numerical ridges and recursion safeguards provide stable factor-covariance
% estimation and backward recursion.

[n, Jmax, T] = size(R_A);
assert(numel(lambda)==T && numel(N_by_stage)==T);
assert(all(N_by_stage<=Jmax) && all(N_by_stage>n));

mu_hat=zeros(T,n); Sigma_factor_raw=zeros(n,n,T);
Sigma_hat=zeros(n,n,T); ER_t=zeros(n,n,T);
factor_residual_min=zeros(T,1);
for t=1:T
    J=N_by_stage(t); X=R_A(:,1:J,t);
    mu_hat(t,:)=mean(X,2)';
    S=cov(X'); S=(S+S')/2;
    [Sf,residual_min]=pca_factor_covariance(S,factor_count);
    Sigma_factor_raw(:,:,t)=Sf;
    factor_residual_min(t)=residual_min;
    Sigma_hat(:,:,t)=Sf+1e-8*eye(n);
    chol(Sigma_hat(:,:,t),'lower');
    ER_t(:,:,t)=Sigma_hat(:,:,t)+mu_hat(t,:)'*mu_hat(t,:);
end

lambda_t1=zeros(T,1); Sigma_t1=zeros(n,n,T); a_t=zeros(T,1);
b_t=zeros(T,1); q_t=zeros(T,1); alpha_t=zeros(T,1);
A_coef=zeros(n,T); B_coef=zeros(n,T); used_reg=zeros(T,1);
ones_vec=ones(n,1); base_reg=1e-8; min_a_val=1e-6; max_reg_try=10;
for t=T:-1:1
    if t==T
        lambda_t1(t)=lambda(t); Current_Sigma=Sigma_hat(:,:,t);
    else
        lambda_t1(t)=lambda(t)+lambda_t1(t+1)*q_t(t+1);
        Current_Sigma=Sigma_hat(:,:,t)+(1/a_t(t+1))*ER_t(:,:,t);
    end
    current_reg=base_reg; success=false;
    for attempt=1:max_reg_try
        M=(Current_Sigma+Current_Sigma')/2+current_reg*eye(n);
        [L,flag]=chol(M,'lower');
        if flag==0
            xa=L'\(L\ones_vec); xb=L'\(L\mu_hat(t,:)');
            va=ones_vec'*xa; vb=ones_vec'*xb;
            if va>min_a_val && all(isfinite([xa;xb;va;vb]))
                Sigma_t1(:,:,t)=M; a_t(t)=va; b_t(t)=vb;
                used_reg(t)=current_reg; success=true; break;
            end
        end
        current_reg=current_reg*10;
    end
    if ~success, error('FactorETO:RecursiveSolve','ETO solve failed at t=%d.',t); end
    q_t(t)=b_t(t)/a_t(t);
    alpha_t(t)=mu_hat(t,:)*xb-q_t(t)*b_t(t);
    A_coef(:,t)=xa/a_t(t);
    B_coef(:,t)=(lambda_t1(t)/2)*(xb-q_t(t)*xa);
end

results.parameters=struct('mu_hat',mu_hat,'Sigma_factor_raw',Sigma_factor_raw, ...
    'Sigma_hat',Sigma_hat,'ER_t',ER_t,'factor_residual_min',factor_residual_min);
results.recursion=struct('lambda_t1',lambda_t1,'Sigma_t1',Sigma_t1, ...
    'a_t',a_t,'b_t',b_t,'q_t',q_t,'alpha_t',alpha_t,'used_reg',used_reg, ...
    'kappa_2',1./a_t,'kappa_1',-lambda_t1.*q_t);
results.strategy=struct('A_coef',A_coef,'B_coef',B_coef);
results.validation=struct('max_budget_A',max(abs(sum(A_coef,1)-1)), ...
    'max_budget_B',max(abs(sum(B_coef,1))));
end

function [Sf,residual_min]=pca_factor_covariance(S,K)
n=size(S,1); K=min(K,n-1);
[V,D]=eig((S+S')/2,'vector');
[d,order]=sort(real(D),'descend'); V=real(V(:,order));
common=V(:,1:K)*diag(max(d(1:K),0))*V(:,1:K)';
residual=diag(S-common);
roundoff_floor=1e-12*max(mean(diag(S)),eps);
residual=max(real(residual),roundoff_floor);
Sf=common+diag(residual); Sf=(Sf+Sf')/2;
residual_min=min(residual);
end
