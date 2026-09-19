function space=build_policy_space_bridge(Rstage,factor_count,anchor_mode,eto_A,eto_B)
%BUILD_POLICY_SPACE_BRIDGE Common structured space with optional ETO anchors.

[n,J]=size(Rstage);
assert(J>n && factor_count>=1 && isfinite(factor_count));
anchor_mode=lower(char(anchor_mode));
assert(ismember(anchor_mode,{'anchored','unanchored'}));
K=min(floor(factor_count),n-1);

mu_hat=mean(Rstage,2);
S=cov(Rstage');S=(S+S')/2;
[V,D]=eig(S,'vector');
[eigenvalues,order]=sort(real(D),'descend');V=real(V(:,order));
factor_vectors=V(:,1:K);
common=factor_vectors*diag(max(eigenvalues(1:K),0))*factor_vectors';
residual=diag(S-common);
roundoff_floor=1e-12*max(mean(diag(S)),eps);
residual=max(real(residual),roundoff_floor);
residual_used=residual+1e-8;

benchmark=ones(n,1)/n;
if strcmp(anchor_mode,'anchored')
    assert(isequal(size(eto_A),[n,1])&&isequal(size(eto_B),[n,1]));
    assert(abs(sum(eto_A)-1)<=1e-8&&abs(sum(eto_B))<=1e-8);
    raw=[ones(n,1)./residual_used,factor_vectors./residual_used, ...
        mu_hat./residual_used,eto_A-benchmark,eto_B];
else
    raw=[ones(n,1)./residual_used,factor_vectors./residual_used, ...
        mu_hat./residual_used];
end
neutral=raw-ones(n,1)*(sum(raw,1)/n);
[U,singular_values]=svd(neutral,'econ','vector');
if isempty(singular_values)
    rank_tolerance=0;Z=zeros(n,0);
else
    rank_tolerance=max(size(neutral))*eps(max(singular_values));
    Z=U(:,singular_values>rank_tolerance);
end
if ~isempty(Z)
    Z=Z-ones(n,1)*(sum(Z,1)/n);
    [Z,~]=qr(Z,0);
end

space=struct('Z',Z,'benchmark',benchmark,'dimension',size(Z,2), ...
    'factor_count',K,'factor_vectors',factor_vectors, ...
    'factor_eigenvalues',eigenvalues(1:K),'residual_variance',residual, ...
    'residual_variance_used',residual_used,'mu_hat',mu_hat, ...
    'sample_covariance',S,'raw_singular_values',singular_values, ...
    'rank_tolerance',rank_tolerance,'anchor_mode',anchor_mode, ...
    'maximum_dimension',K+2+2*strcmp(anchor_mode,'anchored'));
end
