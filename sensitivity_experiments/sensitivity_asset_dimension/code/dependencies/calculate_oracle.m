function [strategy, debug] = calculate_oracle(mu_true, Sigma_true, lambda)
% CALCULATE_ORACLE Calculates the theoretical optimal strategy (Oracle)
% based on true parameters using the ETO recursion logic.
%
% Inputs:
%   mu_true    - True mean vectors (T x n)
%   Sigma_true - True covariance matrices (T x n x n)
%   lambda     - Risk tolerance parameter vector (1 x T)
%
% Outputs:
%   strategy   - Struct containing A_coef, B_coef
%   debug      - Struct containing Sigma_t1 (for Hessian calculation)

    [T, n] = size(mu_true);
    
    % Initialize recursion variables
    lambda_t1 = zeros(T, 1);
    Sigma_t1 = zeros(n, n, T);
    a_t = zeros(T, 1);
    b_t = zeros(T, 1);
    q_t = zeros(T, 1);
    
    A_coef = zeros(n, T);
    B_coef = zeros(n, T);
    
    ones_vec = ones(n, 1);
    
    % Backward Recursion
    for t = T:-1:1
        % 1. Construct Sigma_t1
        if t == T
            lambda_t1(t) = lambda(t);
            Current_Sigma = squeeze(Sigma_true(t, :, :));
        else
            lambda_t1(t) = lambda(t) + lambda_t1(t+1) * q_t(t+1);
            
            % E[R R'] = Sigma + mu mu'
            mu_next = mu_true(t, :)';
            Sigma_next = squeeze(Sigma_true(t, :, :));
            ER_next = Sigma_next + mu_next * mu_next';
            
            % Note: In ETO, Sigma_t1 depends on a_{t+1}
            Current_Sigma = Sigma_next + (1/a_t(t+1)) * ER_next;
        end
        
        Sigma_t1(:, :, t) = Current_Sigma;
        
        % 2. Solve Linear Systems (Inversion)
        % Using Cholesky for stability (Oracle should be stable, but safe is better)
        % Sigma * x_a = 1
        % Sigma * x_b = mu
        
        % Add tiny jitter if needed for numerical stability of inversion
        jitter = 1e-10;
        L = chol(Current_Sigma + jitter * eye(n), 'lower');
        
        x_a = L' \ (L \ ones_vec);
        x_b = L' \ (L \ mu_true(t, :)');
        
        val_a = ones_vec' * x_a;
        val_b = ones_vec' * x_b;
        
        a_t(t) = val_a;
        b_t(t) = val_b;
        q_t(t) = val_b / val_a;
        
        % 3. Calculate Strategy Coefficients
        % A_t = x_a / a_t
        A_coef(:, t) = x_a / a_t(t);
        
        % B_t = (lambda_t1 / 2) * (x_b - q_t * x_a)
        B_coef(:, t) = (lambda_t1(t) / 2) * (x_b - q_t(t) * x_a);
    end
    
    strategy.A_coef = A_coef;
    strategy.B_coef = B_coef;
    
    debug.Sigma_t1 = Sigma_t1;
    debug.a_t = a_t;
    debug.lambda_t1 = lambda_t1;

end
