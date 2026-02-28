clc,clear,close all

N = 32;         
K = 4;          
M = 96;        
SNR=10;

x0 = zeros(N, 1);
x0(3:K+2) = exp(j*2*pi*rand(K,1)); 
x0 = x0 / norm(x0);     

A = (1/sqrt(2)) * (randn(M, N) + 1i * randn(M, N)); 
b = A * x0;     
y = abs(AWGN(b,SNR));

x_est_gctf = GCTF(A,y);
x_est_swf = SWF(A, y);
x_est_cprime = C_PRIME(A, y,1e-2,500);
x_est_saltmin = SparseAltMinPhase(A, y,2*K);
x_est_taf = truncate_amplitude_flow(A, y,2*K);
x_est_phaseliftoff = phaseliftoff(y,A);
x_est_hwf = hwf(A, y, K);
x_est_phasecut = phasecut(A, y);
x_est_gs = Gerchberg_Saxton(A, y);

plot(abs(x_est_gctf))
hold on;
plot(abs(x_est_swf))
plot(abs(x_est_cprime))
plot(abs(x_est_saltmin))
plot(abs(x_est_taf))
plot(abs(x_est_phaseliftoff))
plot(abs(x_est_hwf))
plot(abs(x_est_phasecut))
plot(abs(x_est_gs))

legend("GCTF","SWF","CPRIME","SALTMIN","TAF","PhaseLiftOff","HWF","PhaseCut","GS")



figure

x_est_original = GCTF(A,y);
x_est_module3_phaseliftoff = GCTF_module3_phaseliftoff(A,y);
x_est_module3_swf = GCTF_module3_swf(A,y);
x_est_module3_cprime = GCTF_module3_cprime(A,y);
x_est_module3_gs = GCTF_module3_gs(A,y);
x_est_module1_phaseliftoff = GCTF_module1_phaseliftoff(A,y);
x_est_module1_swf = GCTF_module1_swf(A,y);
x_est_module1_cprime = GCTF_module1_cprime(A,y);
x_est_module1_gs = GCTF_module1_gs(A,y);

plot(abs(x_est_original))
hold on;
plot(abs(x_est_module3_phaseliftoff))
plot(abs(x_est_module3_swf))
plot(abs(x_est_module3_cprime))
plot(abs(x_est_module3_gs))
plot(abs(x_est_module1_phaseliftoff))
plot(abs(x_est_module1_swf))
plot(abs(x_est_module1_cprime))
plot(abs(x_est_module1_gs))

legend("GCTF","Module III PhaseLiftOff","Module III SWF","Module III C-PRIME","Module III GS","Module I PhaseLiftOff","Module I SWF","Module I C-PRIME","Module I GS")


function x_hat = truncate_amplitude_flow(A, y,K, opts)
if nargin < 4
    opts = [];
end

[m, n] = size(A);

if ~isfield(opts, 'max_iter'), opts.max_iter = 1000; end
if ~isfield(opts, 'tol'), opts.tol = 1e-8; end
if ~isfield(opts, 'step_size'), opts.step_size = 1; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'reg_param'), opts.reg_param = 0.1; end

k=K;

Y = zeros(n, n);
for i = 1:m
    Y = Y + y(i)^2 * (A(i, :)' * A(i, :));
end
Y = Y / m;

if isscalar(Y)
    V = 1;
    D = Y;
else
    [V, D] = eigs(Y, 1);
end
x0 = sqrt(D) * V;
x0 = x0 / norm(x0);

scale = sqrt(sum(y.^2) / m);
x_init = scale * x0;

x_hat = x_init;
obj_vals = zeros(opts.max_iter, 1);

for iter = 1:opts.max_iter
    current_obj = compute_objective(A, y, x_hat);
    grad = compute_gradient(A, y, x_hat);
    obj_vals(iter) = current_obj;

    step_size = 1;
    backtrack_count = 0;

    for backtrack_iter = 1:30
        x_candidate = x_hat - step_size * grad;
        x_candidate = hard_threshold(x_candidate, k);

        candidate_obj = compute_objective(A, y, x_candidate);

        if candidate_obj < current_obj
            break;
        else
            step_size = 0.5 * step_size;
            backtrack_count = backtrack_count + 1;
        end
    end

    x_new = x_hat - step_size * grad;
    x_new = hard_threshold(x_new, k);

    step_sizes(iter) = step_size;
    backtrack_iters(iter) = backtrack_count;

    if iter > 1
        rel_change = norm(x_new - x_hat) / norm(x_hat);
        if rel_change < opts.tol
            break;
        end
    end

    x_hat = x_new;

    if opts.verbose && mod(iter, 100) == 0
    end
end

end

function x_thresh = hard_threshold(x, k)
[~, idx] = sort(abs(x), 'descend');
x_thresh = zeros(size(x));
x_thresh(idx(1:k)) = x(idx(1:k));
end

function x_est = SparseAltMinPhase(A, y,K)
[m, n] = size(A);

S = support_recovery(A, y,K);

A_S = A(:, S);
x_S_est = AltMin(A_S, y);

x_est = zeros(n, 1);
x_est(S) = x_S_est;

end

function S = support_recovery(A, y,K)
[n, m] = size(A);
k=K;
scores = zeros(n, 1);
for j = 1:m
    scores(j) = sum(abs(A(:, j)' * y));
end
[~, idx] = sort(scores, 'descend');
S = idx(1:k);
end

function x_est = GCTF(A,y)
[m,d]=size(A);
x_est_shangyici = zeros(d,1);
for itertimes = 1:8

    if itertimes == 1
        x_est = gradient_descent(A, y);
    else
        x_est =gradient_descent(A, y, x_est);
    end

    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate;
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2;
        else
            break;
        end
    end

    C_truncated = A(:,set_index_source);
    x_subspace = gradient_descent(C_truncated,y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end

function x_noise=AWGN(x,SNR)
[m n]=size(x);
Eb=norm(x,2)^2/m/n;
sigma=Eb/10^(SNR/10)/2;
[size_x size_y]=size(x);
x_noise=x+sqrt(sigma)*randn(size_x,size_y);
x_noise=x_noise+j*sqrt(sigma)*randn(size_x,size_y);
end

function x_est = C_PRIME(A, y, rho, max_iter, x_init)
[M, N] = size(A);

if any(y < 0)
    error('y must be nonnegative.');
end

y_sqrt = y;

C = max(eig(A' * A));

if nargin < 5 || isempty(x_init)
    x = randn(N, 1) + 1i * randn(N, 1);
else
    x = x_init;
end

A_H = A';

for iter = 1:max_iter
    Ax = A * x;
    phase_Ax = exp(1i * angle(Ax));
    c1 = x - (1/C) * A_H * (Ax - y_sqrt .* phase_Ax);

    x1 = exp(1i * angle(c1)) .* max(abs(c1) - rho/(2*C), 0);

    Ax1 = A * x1;
    phase_Ax1 = exp(1i * angle(Ax1));
    c2 = x1 - (1/C) * A_H * (Ax1 - y_sqrt .* phase_Ax1);

    x2 = exp(1i * angle(c2)) .* max(abs(c2) - rho/(2*C), 0);

    r = x1 - x;
    v = x2 - x1 - r;
    alpha = -norm(r) / norm(v);

    x3 = x - 2 * alpha * r + alpha^2 * v;

    while true
        Ax3 = A * x3;
        f3 = norm(y_sqrt - abs(Ax3))^2 + rho * norm(x3, 1);
        Ax2 = A * x2;
        f2 = norm(y_sqrt - abs(Ax2))^2 + rho * norm(x2, 1);

        if f3 <= f2
            break;
        else
            alpha = (alpha - 1) / 2;
            x3 = x - 2 * alpha * r + alpha^2 * v;
        end
    end

    Ax3 = A * x3;
    phase_Ax3 = exp(1i * angle(Ax3));
    c3 = x3 - (1/C) * A_H * (Ax3 - y_sqrt .* phase_Ax3);
    x_next = exp(1i * angle(c3)) .* max(abs(c3) - rho/(2*C), 0);

    if norm(x_next - x) / norm(x) < 1e-6
        fprintf('C_Prime Converged at iteration %d\n', iter);
        break;
    end

    x = x_next;
end

x_est = x;

end

function [set_index_source,set_index_sink] = zuixiaoge(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei)
n_edge=2*length(cap_sink)+length(edge_tou);
l=length(edge_tou);
n_node=length(cap_sink);
bianji=zeros(n_edge,3);
for u = 1:n_node
    bianji(u,1)=n_node+1;
    bianji(u,2)=u;
    bianji(u,3)=cap_source(u);
    bianji(u+n_node,1)=u;
    bianji(u+n_node,2)=n_node+2;
    bianji(u+n_node,3)=cap_sink(u);
end
for u=1:l
    bianji(u+2*n_node,1)=edge_tou(u);
    bianji(u+2*n_node,2)=edge_wei(u);
    bianji(u+2*n_node,3)=cap_xianglin;
end
yizhaodaozuidaliu=0;
m=n_node+2;
A=zeros(m);
for u = 1:length(bianji)
    A(bianji(u,1),bianji(u,2))=bianji(u,3);
end
maxflow=zeros(m);
while 1
    flag=[];
    flag=[flag n_node+1];
    head=n_node+1;
    tail=n_node+1;
    queue=[];
    queue(n_node+1)=n_node+1;
    head=1;
    pa=zeros(1,m);
    pa(n_node+1)=n_node+1;
    while tail~=head
        u=queue(tail);
        for v=1:m
            if A(u,v)>0 && isempty(find(flag==v,1))
                queue(head)=v;
                if head<n_node
                    head=head+1;
                else
                    head=n_node+2;
                end
                flag=[flag v];
                pa(v)=u;
            end
        end
        if tail==n_node+1
            tail=1;
        else
            if tail==n_node
                tail=n_node+2;
            else
                tail=tail+1;
            end
        end
    end
    if pa(m)==0
        break;
    end
    path=[];
    u=m;
    k=0;
    while u ~= n_node+1
        path=[path;pa(u) u A(pa(u),u)];
        u=pa(u);
        k=k+1;
    end
    Mi=min(path(:,3));
    for u=1:k
        A(path(u,1),path(u,2))=A(path(u,1),path(u,2))-Mi;
        maxflow(path(u,1),path(u,2))=maxflow(path(u,1),path(u,2))+Mi;
    end
end
set_index_source=[];
set_index_sink=[];
for u = 1:n_node
    if isempty(find(flag==u))
        set_index_sink=[set_index_sink,u];
    else
        set_index_source=[set_index_source,u];
    end
end
end

function x_hat = gradient_descent(A, y, x_initial)
opts=[];
if ~isfield(opts, 'max_iter'), opts.max_iter = 1e4; end
if ~isfield(opts, 'tol'), opts.tol = 1e-8; end
if ~isfield(opts, 'step_size'), opts.step_size = 1; end
if ~isfield(opts, 'sparsity'), opts.sparsity = 0.5; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'reg_param'), opts.reg_param = 0.1; end

[m, n] = size(A);

Y = zeros(n, n);
for i = 1:m
    Y = Y + y(i)^2 * (A(i, :)' * A(i, :));
end
Y = Y / m;

[V, D] = eigs(Y, 1);
x0 = sqrt(D) * V;
x0 = x0 / norm(x0);

if nargin < 3
    scale = sqrt(sum(y.^2) / m);
    x_init = scale * x0;
else
    x_init = x_initial;
end

x_hat = x_init;
obj_vals = zeros(opts.max_iter, 1);

if opts.verbose
end

for iter = 1:opts.max_iter
    current_obj = compute_objective(A, y, x_hat);
    grad = compute_gradient(A, y, x_hat);
    obj_vals(iter) = current_obj;

    step_size = 1;
    backtrack_count = 0;

    for backtrack_iter = 1:30
        x_candidate = x_hat - step_size * grad;

        candidate_obj = compute_objective(A, y, x_candidate);

        if candidate_obj < current_obj
            break;
        else
            step_size = 0.5 * step_size;
            backtrack_count = backtrack_count + 1;
        end
    end

    x_new = x_hat - step_size * grad;

    step_sizes(iter) = step_size;
    backtrack_iters(iter) = backtrack_count;

    if iter > 1
        rel_change = norm(x_new - x_hat) / norm(x_hat);
        if rel_change < opts.tol
            if opts.verbose
                fprintf('Gradient_Descent Converge at Iteration %d\n', iter);
            end
            break;
        end
    end

    x_hat = x_new;

    if opts.verbose && mod(iter, 100) == 0
    end
end

end

function grad = compute_gradient(A, y, x)
    m = length(y);
    Ax = A * x;
    measurement = abs(Ax).^2;
    error = measurement - y.^2;
    grad_sum = A' * (error .* Ax);
    grad = (2/m) * grad_sum;
end

function obj_val = compute_objective(A, y, x)
    measurements = abs(A * x).^2;
    obj_val = sum((measurements - y.^2).^2) / (2*length(y));
end

function x_est = phaseliftoff(y,C,xinitial)
[m,d]=size(C);
b = abs(y).^2;
A=C;
k = d;
mu = 1e-3;     
lambda = mu * k / (sqrt(2) - 1); 

max_iter_dca = 25;      
max_iter_admm = 25;   
tol_dca = 1e-4;       
delta = 1;          

if nargin < 3
    X = zeros(d, d);        
    X_prev = X;
    Y = zeros(d, d);        
else
    X = xinitial*xinitial';
    X_prev = X;
end

for iter_dca = 1:max_iter_dca
    if norm(X, 'fro') > 0
        Y = X / norm(X, 'fro');
    else
        Y = zeros(d, d);
    end

    W = lambda * (eye(d) - Y); 

    X1 = zeros(d, d);
    X2 = zeros(d, d);
    X3 = zeros(d, d);
    Y1 = zeros(d, d);
    Y2 = zeros(d, d);

    A_mat = A;
    I_d = eye(d);
    I_m = eye(m);
    A_adj_A = zeros(d^2, d^2);
    M = zeros(d^2, d^2);
    for i = 1:d
        for j = 1:d
            e_ij = zeros(d, d);
            e_ij(i, j) = 1;
            vec_ij = A_adj_A_func(e_ij, A_mat);
            M((j-1)*d+i, :) = vec_ij(:)';
        end
    end
    M = M + delta * eye(d^2);
    Minv = pinv(M);

    X3_prev = X3;

    for iter_admm = 1:max_iter_admm
        rhs_X1 = A_adj_b(A_mat, b) - W + delta * X3 - Y1;
        vec_rhs = rhs_X1(:);
        vec_X1 = Minv * vec_rhs;
        X1 = reshape(vec_X1, [d, d]);

        Z = X3 - Y2 / delta;
        X2 = soft_threshold(Z, mu / delta);
        Z_avg = (X1 + X2 + (Y1 + Y2) / delta) / 2;
        X3 = proj_psd(Z_avg);

        Y1 = Y1 + delta * (X1 - X3);
        Y2 = Y2 + delta * (X2 - X3);

        prim_res = norm(X1 - X3, 'fro') + norm(X2 - X3, 'fro');
        dual_res = delta * norm(X3 - X3_prev, 'fro'); 
        if prim_res < 1e-4 && dual_res < 1e-4
            break;
        end
        X3_prev = X3;
    end

    X_next = X3; 

    diff = norm(X_next - X, 'fro') / max(norm(X, 'fro'), 1);
    if diff < tol_dca
        %fprintf('DCA converged at iteration %d.\n', iter_dca);
        break;
    end

    X_prev = X;
    X = X_next;
end

[V, D] = eig(X);
[~, idx] = max(diag(D));
x_est = sqrt(D(idx, idx)) * V(:, idx);

end

function res = A_adj_b(A, b)
    [m, n] = size(A);
    res = A' * (diag(b) * A);
end

function res = soft_threshold(Z, tau)
res = sign(Z) .* max(abs(Z) - tau, 0);
end

function X_psd = proj_psd(Z)
[V, D] = eig(Z);
D = diag(D);
D(D < 0) = 0;
X_psd = V * diag(D) * V';
end

function res = A_adj_A_func(X, A)
    [m, n] = size(A);
    AXA = A * X * A';  
    q = diag(AXA);     
    res = A' * (diag(q) * A);
end

function x_hat = AltMin(A, y, xinitial)
[M,N]=size(A);
if nargin < 3
    theta = rand(M,1)*2*pi;
else
    theta = angle(A*xinitial);
end
for iter = 1:100
    x_hat = pinv(A)*(y.*exp(j*theta));
    theta = angle(A*x_hat);
end
end

function x_hat = SWF(A, y, x_initial)
opts=[];
if ~isfield(opts, 'max_iter'), opts.max_iter = 1e4; end
if ~isfield(opts, 'tol'), opts.tol = 1e-8; end
if ~isfield(opts, 'step_size'), opts.step_size = 1; end
if ~isfield(opts, 'sparsity'), opts.sparsity = 0.5; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'reg_param'), opts.reg_param = 0.1; end


[m, n] = size(A);

Y = zeros(n, n);
for i = 1:m
    Y = Y + y(i)^2 * (A(i, :)' * A(i, :));
end
Y = Y / m;

[V, D] = eigs(Y, 1);
x0 = sqrt(D) * V;
x0 = x0 / norm(x0);

if nargin < 3
    scale = sqrt(sum(y.^2) / m);
    x_init = scale * x0;
else
    x_init = x_initial;
end

x_hat = x_init;
obj_vals = zeros(opts.max_iter, 1);


for iter = 1:opts.max_iter
    current_obj = compute_objective_swf(A, y, x_hat, opts.reg_param);
    grad = compute_gradient_swf(A, y, x_hat, opts.reg_param);
    obj_vals(iter) = current_obj;

    step_size = 1;
    backtrack_count = 0;

    for backtrack_iter = 1:30
        x_candidate = x_hat - step_size * grad;

        candidate_obj = compute_objective_swf(A, y, x_candidate, opts.reg_param);

        if candidate_obj < current_obj
            break; 
        else
            step_size = 0.5 * step_size;
            backtrack_count = backtrack_count + 1;
        end
    end

    x_new = x_hat - step_size * grad;

    step_sizes(iter) = step_size;
    backtrack_iters(iter) = backtrack_count;

    if iter > 1
        rel_change = norm(x_new - x_hat) / norm(x_hat);
        if rel_change < opts.tol
            if opts.verbose
                fprintf('SWF Converge at Iteration %d\n', iter);
            end
            break;
        end
    end

    x_hat = x_new;
end

end

function grad = compute_gradient_swf(A, y, x, reg_param)
    m = length(y);
    Ax = A * x; 
    measurement = abs(Ax).^2; 
    error = measurement - y.^2; 
    grad_sum = A' * (error .* Ax);
    grad = (2/m) * grad_sum;
    sparse_grad = reg_param * sign(x);
    grad = grad + sparse_grad;
end

function obj_val = compute_objective_swf(A, y, x, reg_param)
    m = length(y);
    Ax = A * x; 
    measurement = abs(Ax).^2; 
    error_squared_sum = sum((measurement - y.^2).^2);
    obj_val = error_squared_sum / (2*m);
    sparse_penalty = reg_param * norm(x, 1);
    obj_val = obj_val + sparse_penalty;
end

function x_hat = Gerchberg_Saxton(A, y, xinitial)
[M,N]=size(A);
if nargin < 3
    theta = rand(M,1)*2*pi;
else
    theta = angle(A*xinitial);
end
for iter = 1:100
    x_hat = pinv(A)*(y.*exp(j*theta));
    theta = angle(A*x_hat);
end
end


function x_est = phasecut(A, y, options)

if nargin < 3
    options = struct();
end

[M, N] = size(A);

max_iter = get_option_phasecut(options, 'max_iter', 2000);
tol = get_option_phasecut(options, 'tol', 1e-8);
verbose = get_option_phasecut(options, 'verbose', false);

G = A' * bsxfun(@times, A, y.^2);
G = (G + G') / 2;

u = phasecut_power_iteration(G, max_iter, tol, verbose);

U = diag(u);

x_init = u;

x_est = refine_solution_phasecut(A, y, x_init, options);


end

function x = refine_solution_phasecut(A, y, x_init, options)

max_iter = get_option_phasecut(options, 'refine_iter', 50);
tol = get_option_phasecut(options, 'refine_tol', 1e-6);
verbose = get_option_phasecut(options, 'verbose', false);

x = x_init;
[M, N] = size(A);

for iter = 1:max_iter
    x_old = x;
    
    Ax = A * x;
    phase_Ax = Ax ./ (abs(Ax) + eps);
    
    b_target = y .* phase_Ax;
    
    if N <= M
        x = A \ b_target;
    else
        lambda = get_option_phasecut(options, 'lambda', 1e-3);
        x = (A'*A + lambda*eye(N)) \ (A' * b_target);
    end
    
    x = x / norm(x);
    
    change = norm(x - x_old);
    if change < tol
        break;
    end
end

end

function u = phasecut_power_iteration(G, max_iter, tol, verbose)

N = size(G, 1);

try
    [V, ~] = eigs(G, 1, 'largestabs');
    u = V ./ abs(V);
catch
    theta = 2 * pi * rand(N, 1);
    u = exp(1j * theta);
end

for iter = 1:max_iter
    u_old = u;
    
    u_new = G * u;
    
    u_new = u_new ./ (abs(u_new) + eps);
    
    change = norm(u - u_new) / sqrt(N);
    u = u_new;
    
    if verbose && mod(iter, 100) == 0
        obj_val = real(u' * G * u);
    end
    
    if change < tol
        break;
    end
end

end

function value = get_option_phasecut(options, field, default)
if isfield(options, field)
    value = options.(field);
else
    value = default;
end
end


function x_est = hwf(A, y, K, params)

if nargin < 4
    params = struct();
end
eta = get_param(params, 'eta', 1e-3);
alpha = get_param(params, 'alpha', 0.001);
max_iter = get_param(params, 'max_iter', 500);
tol = get_param(params, 'tol', 1e-3);
restarts = get_param(params, 'restarts', 1);
kappa = get_param(params, 'kappa', 0.05);

[M, N] = size(A);

y2 = y.^2;  
theta_hat = sqrt(mean(y2)); 

R = zeros(N, 1);
for i = 1:N
    R(i) = real(mean(y2 .* abs(A(:,i)).^2));
end

best_grad_norm = inf;
x_best = randn(N, 1);

for b = 1:restarts
    [~, idx_sorted] = sort(R, 'descend');
    I_b = idx_sorted(b);
    
    u = alpha * (randn(N, 1) + 1j*randn(N,1));
    v = alpha * (randn(N, 1) + 1j*randn(N,1));
    u(I_b) = sqrt(theta_hat/sqrt(3) + alpha^2);  
    
    for t = 1:max_iter
        x = (u.^2) - (v.^2);
        
        Ax = A * x;
        grad = zeros(N, 1);
        for j = 1:M
            grad = grad + (abs(Ax(j))^2 - y2(j)) * conj(Ax(j)) * A(j,:)';
        end
        grad = grad / M;
        
        u = u .* (1 - 2*eta * grad);
        v = v .* (1 + 2*eta * grad);
        
        if norm(grad) < tol
            break;
        end
    end
    
    x = (u.^2) - (v.^2);
    
    x_abs = abs(x);
    [~, sort_idx] = sort(x_abs, 'descend');
    support = sort_idx(1:K);
    
    x_sparse = zeros(N, 1);
    x_sparse(support) = x(support);
    
    x_sparse = align_phase(x_sparse, A, y);
    
    Ax = A * x_sparse;
    grad_norm = norm((abs(Ax).^2 - y2) .* conj(Ax) .* (A * x_sparse));
    
    if grad_norm < best_grad_norm
        best_grad_norm = grad_norm;
        x_best = x_sparse;
    end
end

x_est = x_best;
end

function val = get_param(params, field, default)
if isfield(params, field)
    val = params.(field);
else
    val = default;
end
end

function x_aligned = align_phase(x, A, y)
Ax = A * x;
phi = angle(Ax' * y);
x_aligned = x * exp(-1i * phi);
end

function x_est = GCTF_module3_phaseliftoff(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = gradient_descent(A, y);
    else
        x_est = gradient_descent(A, y, x_est);
    end

    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = phaseliftoff(y,C_truncated);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end



function x_est = GCTF_module3_swf(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = gradient_descent(A, y);
    else
        x_est = gradient_descent(A, y, x_est);
    end


    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = SWF(C_truncated,y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end





function x_est = GCTF_module3_cprime(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = gradient_descent(A, y);
    else
        x_est = gradient_descent(A, y, x_est);
    end

    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = C_PRIME(C_truncated, y,1e-2,500);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end





function x_est = GCTF_module3_gs(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = gradient_descent(A, y);
    else
        x_est = gradient_descent(A, y, x_est);
    end

    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = AltMin(C_truncated, y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end

function x_est = GCTF_module1_phaseliftoff(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = phaseliftoff(y,A);
    else
        x_est = phaseliftoff(y, A, x_est);
    end


    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = AltMin(C_truncated,y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end



function x_est = GCTF_module1_swf(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = SWF(A,y);
    else
        x_est = SWF(A,y,x_est);
    end

    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = AltMin(C_truncated,y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end





function x_est = GCTF_module1_cprime(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = C_PRIME(A, y, 1e-2, 500);
    else
        x_est = C_PRIME(A, y, 1e-2, 500, x_est);
    end


    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = AltMin(C_truncated, y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end





function x_est = GCTF_module1_gs(A,y)
[m,d]=size(A);
cishu = 8;
x_est_shangyici = zeros(d,1);
for itertimes = 1:cishu

    if itertimes == 1
        x_est = AltMin(A, y);
    else
        x_est = AltMin(A, y, x_est);
    end


    n=d;
    xtemp=abs(x_est)/norm(x_est,2)*sqrt(n);
    concentrationrate=sum(xtemp.^2+xtemp.^(-2));
    for iter = 1:100
        cap_source=xtemp.^2;
        cap_sink=1./cap_source;
        cap_xianglin=concentrationrate; 
        edge_tou=[1:n-1,n:-1:2];
        edge_wei=[2:n,n-1:-1:1];
        [set_index_source,set_index_sink]=min_cut_maxflow(cap_source,cap_sink,cap_xianglin,edge_tou,edge_wei);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2; 
        else
            break;
        end
    end


    C_truncated = A(:,set_index_source);
    x_subspace = AltMin(C_truncated, y);
    x_est = zeros(d,1);
    x_est(set_index_source) = x_subspace;

    if norm(abs(x_est)-abs(x_est_shangyici),2)^2/norm(x_est,2)^2 < 1e-8
        break;
    end
    x_est_shangyici = x_est;
end
end

function [set_index_source, set_index_sink] = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_tou, edge_wei)
    n_node = length(cap_sink);
    s = n_node + 1; 
    t = n_node + 2; 
    
    all_s = [];
    all_t = [];
    all_weights = [];
    
    all_s = [all_s, repmat(s, 1, n_node)];
    all_t = [all_t, 1:n_node];
    all_weights = [all_weights, cap_source'];
    
    all_s = [all_s, 1:n_node];
    all_t = [all_t, repmat(t, 1, n_node)];
    all_weights = [all_weights, cap_sink'];
    
    all_s = [all_s, edge_tou];
    all_t = [all_t, edge_wei];
    all_weights = [all_weights, repmat(cap_xianglin, 1, length(edge_tou))];
    
    G = digraph(all_s, all_t, all_weights);
    [~, ~, cs, ct] = maxflow(G, s, t);
    
    set_index_source = cs(cs <= n_node);
    set_index_sink = ct(ct <= n_node);
    
    set_index_source = sort(set_index_source);
    set_index_sink = sort(set_index_sink);
end