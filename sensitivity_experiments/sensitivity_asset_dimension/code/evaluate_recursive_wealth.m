function [mean_path,terminal_path,max_budget_error]=evaluate_recursive_wealth(R_B,strategy,x0)
%EVALUATE_RECURSIVE_WEALTH Main-experiment own-policy wealth recursion.
%   u_t = A_t X_t + B_t,  X_{t+1} = R_t' u_t,  X_0 = x0.

[n,N_B,T]=size(R_B);
assert(isequal(size(strategy.A_coef),[n,T]));
assert(isequal(size(strategy.B_coef),[n,T]));
validateattributes(x0,{'numeric'},{'scalar','finite'});

X=x0*ones(N_B,1);
mean_path=zeros(1,T+1);
mean_path(1)=mean(X);
max_budget_error=0;
for t=1:T
    U=strategy.A_coef(:,t)*X'+strategy.B_coef(:,t);
    max_budget_error=max(max_budget_error,max(abs(sum(U,1)'-X)));
    X=sum(R_B(:,:,t).*U,1)';
    mean_path(t+1)=mean(X);
end
terminal_path=X;
end
