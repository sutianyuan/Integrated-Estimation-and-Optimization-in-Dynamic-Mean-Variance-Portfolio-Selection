function results=run_structured_ieo_bridge(R_A,lambda,N_by_stage,factor_count,anchor_mode,eto_anchor)
%RUN_STRUCTURED_IEO_BRIDGE Structured IEO with or without F-ETO anchors.

[n,Jmax,T]=size(R_A);
assert(numel(lambda)==T&&numel(N_by_stage)==T);
assert(all(N_by_stage<=Jmax)&&all(N_by_stage>n));
anchor_mode=lower(char(anchor_mode));
assert(ismember(anchor_mode,{'anchored','unanchored'}));
if strcmp(anchor_mode,'anchored')
    assert(nargin==6&&isfield(eto_anchor,'strategy'));
end

A_coef=zeros(n,T);B_coef=zeros(n,T);
kappa_2=zeros(T,1);kappa_1=zeros(T,1);kappa_0=zeros(T,1);
x_grid=linspace(.5,1.5,10)';
options_qp=optimoptions('quadprog','Display','off');
options_lsqlin=optimoptions('lsqlin','Display','off');
fallback_qp_count=0;fallback_benchmark_count=0;fallback_value_count=0;
spaces=cell(T,1);space_dimension=zeros(T,1);
policy_space_residual_A=zeros(T,1);policy_space_residual_B=zeros(T,1);

for t=T:-1:1
    J=N_by_stage(t);Rstage=R_A(:,1:J,t);
    if strcmp(anchor_mode,'anchored')
        eto_A=eto_anchor.strategy.A_coef(:,t);eto_B=eto_anchor.strategy.B_coef(:,t);
    else
        eto_A=[];eto_B=[];
    end
    space=build_policy_space_bridge(Rstage,factor_count,anchor_mode,eto_A,eto_B);
    spaces{t}=space;space_dimension(t)=space.dimension;
    rng(20260115+t,'twister');order=randperm(J);split=floor(J/2);
    halves={order(1:split),order(split+1:end)};
    if t==T,nk2=0;nk1=0;nk0=0;
    else,nk2=kappa_2(t+1);nk1=kappa_1(t+1);nk0=kappa_0(t+1);end
    [A1,B1,k21,k11,k01,flags1]=run_half(Rstage,halves{1},lambda(t),nk2,nk1,nk0,x_grid,space,options_qp,options_lsqlin);
    [A2,B2,k22,k12,k02,flags2]=run_half(Rstage,halves{2},lambda(t),nk2,nk1,nk0,x_grid,space,options_qp,options_lsqlin);
    A_coef(:,t)=.5*(A1+A2);B_coef(:,t)=.5*(B1+B2);
    kappa_2(t)=.5*(k21+k22);kappa_1(t)=.5*(k11+k12);kappa_0(t)=.5*(k01+k02);
    fallback_qp_count=fallback_qp_count+flags1.qp+flags2.qp;
    fallback_benchmark_count=fallback_benchmark_count+flags1.benchmark+flags2.benchmark;
    fallback_value_count=fallback_value_count+flags1.value+flags2.value;
    assert(abs(sum(A_coef(:,t))-1)<1e-8&&abs(sum(B_coef(:,t)))<1e-8);
    if isempty(space.Z)
        policy_space_residual_A(t)=norm(A_coef(:,t)-space.benchmark);
        policy_space_residual_B(t)=norm(B_coef(:,t));
    else
        policy_space_residual_A(t)=norm((eye(n)-space.Z*space.Z')*(A_coef(:,t)-space.benchmark));
        policy_space_residual_B(t)=norm((eye(n)-space.Z*space.Z')*B_coef(:,t));
    end
    if kappa_2(t)<1e-8,kappa_2(t)=1e-8;end
end

results.strategy=struct('A_coef',A_coef,'B_coef',B_coef);
results.value_function=struct('kappa_2',kappa_2,'kappa_1',kappa_1,'kappa_0',kappa_0);
results.factor_policy=struct('spaces',{spaces},'dimension',space_dimension, ...
    'policy_space_residual_A',policy_space_residual_A, ...
    'policy_space_residual_B',policy_space_residual_B,'anchor_mode',anchor_mode);
results.validation=struct('max_budget_A',max(abs(sum(A_coef,1)-1)), ...
    'max_budget_B',max(abs(sum(B_coef,1))), ...
    'max_policy_space_residual_A',max(policy_space_residual_A), ...
    'max_policy_space_residual_B',max(policy_space_residual_B), ...
    'fallback_qp_count',fallback_qp_count, ...
    'fallback_benchmark_count',fallback_benchmark_count, ...
    'fallback_value_count',fallback_value_count);
end

function [A,B,k2,k1,k0,flags]=run_half(R_all,idx,lam,nk2,nk1,nk0,x_grid,space,options_qp,options_lsqlin)
R=R_all(:,idx);N=size(R,2);Rnet=R-1;
m1=mean(R,2);M2=(R*R')/N;mnet=mean(Rnet,2);M2net=(Rnet*Rnet')/N;
Cov_R=M2-m1*m1';H=2*(Cov_R+nk2*M2net);H=(H+H')/2;
flags=struct('qp',0,'benchmark',0,'value',0);
[u0,f0]=solve_point(H,mnet*(nk1-lam),0,space,options_qp);
[u1,f1]=solve_point(H,mnet*(nk1-lam+2*nk2),1,space,options_qp);
flags.qp=f0.qp+f1.qp;flags.benchmark=f0.benchmark+f1.benchmark;
B=u0;A=u1-u0;
V=zeros(numel(x_grid),1);
for k=1:numel(x_grid)
    x=x_grid(k);f=mnet*(nk1-lam+2*nk2*x);
    [u,ff]=solve_point(H,f,x,space,options_qp);
    flags.qp=flags.qp+ff.qp;flags.benchmark=flags.benchmark+ff.benchmark;
    V(k)=u'*(.5*H)*u+f'*u+nk2*x^2+(nk1-lam)*x+nk0;
end
X=[x_grid.^2,x_grid,ones(numel(x_grid),1)];Aineq=[-1,0,0];bineq=-1e-8;
try
    coefs=lsqlin(X,V,Aineq,bineq,[],[],[],[],[],options_lsqlin);
catch
    coefs=[];
end
if isempty(coefs)
    flags.value=flags.value+1;coefs=quadprog(2*(X'*X),-2*(X'*V),Aineq,bineq,[],[],[],[],[],options_qp);
end
if isempty(coefs)
    flags.value=flags.value+1;coefs=X\V;coefs(1)=max(coefs(1),1e-8);
end
k2=coefs(1);k1=coefs(2);k0=coefs(3);
end

function [u,flags]=solve_point(H,f,x,space,options)
Z=space.Z;w=space.benchmark;r=size(Z,2);
flags=struct('qp',0,'benchmark',0);
if r==0,u=x*w;return;end
Hred=Z'*H*Z;Hred=(Hred+Hred')/2;
fred=Z'*(H*(x*w)+f);
[gamma,~,exitflag]=quadprog(Hred,fred,[],[],[],[],[],[],[],options);
if exitflag<0||isempty(gamma)
    flags.qp=1;[gamma,~,exitflag]=quadprog(Hred+1e-8*eye(r),fred,[],[],[],[],[],[],[],options);
end
if exitflag<0||isempty(gamma),flags.benchmark=1;gamma=zeros(r,1);end
u=x*w+Z*gamma;
end
