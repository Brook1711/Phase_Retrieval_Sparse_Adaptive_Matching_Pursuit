# SPR.py phase retrieval algorithms converted from MATLAB to Python.
# This module implements multiple phase retrieval solvers and supporting utilities.
# The code uses numpy for linear algebra and pure Python for graph flow operations.

"""%The default graph topology is the linear topology. 
You can change the graph topology by adjusting "edge_start" and "edge_end"."""

import numpy as np
import matplotlib.pyplot as plt


# Compute spectral initialization for phase retrieval.
def spectral_initialization(A, y):
    m, n = A.shape
    Y = np.zeros((n, n), dtype=np.complex128)
    for i in range(m):
        ai = A[i, :]
        Y += (y[i] ** 2) * np.outer(np.conjugate(ai), ai)
    Y = Y / m

    if Y.size == 1:
        V = np.array([1.0], dtype=np.complex128)
        D = Y.item()
    else:
        V, D = largest_eig(Y)

    x0 = np.sqrt(D) * V
    x0 = x0 / np.linalg.norm(x0)
    return x0


# Return the principal eigenvector and eigenvalue of a Hermitian matrix.
def largest_eig(Y):
    if Y.size == 1:
        return np.array([1.0], dtype=np.complex128), Y.item()
    evals, evecs = np.linalg.eigh(Y)
    idx = np.argmax(evals)
    return evecs[:, idx], evals[idx]


# Keep the largest k entries by magnitude and zero out the rest.
def hard_threshold(x, k):
    idx = np.argsort(-np.abs(x))
    x_thresh = np.zeros_like(x)
    x_thresh[idx[:k]] = x[idx[:k]]
    return x_thresh


# Compute the gradient of the phase retrieval objective.
def compute_gradient(A, y, x):
    Ax = A @ x
    measurement = np.abs(Ax) ** 2
    error = measurement - y ** 2
    grad_sum = A.conj().T @ (error * Ax)
    grad = (2.0 / len(y)) * grad_sum
    return grad


# Evaluate the phase retrieval loss for a candidate x.
def compute_objective(A, y, x):
    measurements = np.abs(A @ x) ** 2
    obj_val = np.sum((measurements - y ** 2) ** 2) / (2.0 * len(y))
    return obj_val


# Compute the SWF gradient with sparsity penalty.
def compute_gradient_swf(A, y, x, reg_param):
    Ax = A @ x
    measurement = np.abs(Ax) ** 2
    error = measurement - y ** 2
    grad_sum = A.conj().T @ (error * Ax)
    grad = (2.0 / len(y)) * grad_sum
    sparse_grad = reg_param * np.sign(x)
    return grad + sparse_grad


# Compute the SWF objective including l1 regularization.
def compute_objective_swf(A, y, x, reg_param):
    Ax = A @ x
    measurement = np.abs(Ax) ** 2
    error_squared_sum = np.sum((measurement - y ** 2) ** 2)
    obj_val = error_squared_sum / (2.0 * len(y))
    sparse_penalty = reg_param * np.linalg.norm(x, 1)
    return obj_val + sparse_penalty


# Select support indices from measurement correlations.
def support_recovery(A, y, K):
    m, n = A.shape
    scores = np.zeros(n)
    for j in range(n):
        scores[j] = np.abs(np.vdot(A[:, j], y))
    idx = np.argsort(-scores)
    return idx[:K]


# Compute A^H * diag(b) * A efficiently.
def A_adj_b(A, b):
    return A.conj().T @ (A * b[:, np.newaxis])


# Compute A^H * diag(diag(A X A^H)) * A for a matrix X.
def A_adj_A_func(X, A):
    AXA = A @ X @ A.conj().T
    q = np.diag(AXA)
    return A.conj().T @ (q[:, np.newaxis] * A)


# Apply elementwise soft thresholding for sparsity.
def soft_threshold(Z, tau):
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0)


# Project a matrix onto the positive semidefinite cone.
def proj_psd(Z):
    evals, evecs = np.linalg.eigh(Z)
    evals = np.maximum(evals, 0)
    return evecs @ np.diag(evals) @ evecs.conj().T


# Align the global phase of x to the measurements y.
def align_phase(x, A, y):
    Ax = A @ x
    phi = np.angle(np.vdot(Ax, y))
    return x * np.exp(-1j * phi)


# Compute a min-cut partition from source/sink capacities.
def min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end):
    n_node = len(cap_sink)
    s = n_node
    t = n_node + 1
    max_node = n_node + 2

    graph = Dinic(max_node)
    for i in range(n_node):
        if cap_source[i] > 0:
            graph.add_edge(s, i, cap_source[i])
        if cap_sink[i] > 0:
            graph.add_edge(i, t, cap_sink[i])
    for u, v in zip(edge_start, edge_end):
        graph.add_edge(u, v, cap_xianglin)

    graph.max_flow(s, t)
    reachable = graph.min_cut(s)

    set_index_source = np.sort(np.array([i for i in range(n_node) if reachable[i]], dtype=int))
    set_index_sink = np.sort(np.array([i for i in range(n_node) if not reachable[i]], dtype=int))
    return set_index_source, set_index_sink


# Dinic max-flow implementation for min-cut computation.
class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        if cap <= 0:
            return
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])

    def bfs(self, s, t, level):
        for i in range(len(level)):
            level[i] = -1
        queue = [s]
        level[s] = 0
        for u in queue:
            for v, cap, _ in self.adj[u]:
                if cap > 1e-15 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level[t] >= 0

    def dfs(self, u, t, flow, level, it):
        if u == t:
            return flow
        for i in range(it[u], len(self.adj[u])):
            v, cap, rev = self.adj[u][i]
            if cap > 1e-15 and level[v] == level[u] + 1:
                pushed = self.dfs(v, t, min(flow, cap), level, it)
                if pushed > 0:
                    self.adj[u][i][1] -= pushed
                    self.adj[v][rev][1] += pushed
                    return pushed
            it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0.0
        level = [-1] * self.n
        while self.bfs(s, t, level):
            it = [0] * self.n
            pushed = self.dfs(s, t, np.inf, level, it)
            while pushed > 0:
                flow += pushed
                pushed = self.dfs(s, t, np.inf, level, it)
        return flow

    def min_cut(self, s):
        visited = [False] * self.n
        stack = [s]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            for v, cap, _ in self.adj[u]:
                if cap > 1e-15 and not visited[v]:
                    stack.append(v)
        return visited


# Truncate amplitude flow algorithm for sparse phase retrieval.
def truncate_amplitude_flow(A, y, K, opts=None):
    if opts is None:
        opts = {}
    m, n = A.shape
    opts.setdefault('max_iter', 1000)
    opts.setdefault('tol', 1e-8)
    opts.setdefault('step_size', 1)
    opts.setdefault('verbose', True)
    opts.setdefault('reg_param', 0.1)

    k = K
    Y = np.zeros((n, n), dtype=np.complex128)
    for i in range(m):
        ai = A[i, :]
        Y += (y[i] ** 2) * np.outer(np.conjugate(ai), ai)
    Y = Y / m

    V, D = largest_eig(Y)
    x0 = np.sqrt(D) * V
    x0 = x0 / np.linalg.norm(x0)

    scale = np.sqrt(np.sum(y ** 2) / m)
    x_init = scale * x0

    x_hat = x_init
    step_sizes = np.zeros(opts['max_iter'])
    backtrack_iters = np.zeros(opts['max_iter'], dtype=int)

    for iter_idx in range(opts['max_iter']):
        current_obj = compute_objective(A, y, x_hat)
        grad = compute_gradient(A, y, x_hat)

        step_size = 1.0
        backtrack_count = 0

        for _ in range(30):
            x_candidate = x_hat - step_size * grad
            x_candidate = hard_threshold(x_candidate, k)
            candidate_obj = compute_objective(A, y, x_candidate)
            if candidate_obj < current_obj:
                break
            step_size *= 0.5
            backtrack_count += 1

        x_new = hard_threshold(x_hat - step_size * grad, k)
        step_sizes[iter_idx] = step_size
        backtrack_iters[iter_idx] = backtrack_count

        if iter_idx > 0:
            rel_change = np.linalg.norm(x_new - x_hat) / np.linalg.norm(x_hat)
            if rel_change < opts['tol']:
                break

        x_hat = x_new
    return x_hat


# Sparse alternating minimization phase retrieval.
def SparseAltMinPhase(A, y, K):
    m, n = A.shape
    S = support_recovery(A, y, K)
    A_S = A[:, S]
    x_S_est = AltMin(A_S, y)
    x_est = np.zeros(n, dtype=np.complex128)
    x_est[S] = x_S_est
    return x_est


# Gradient-based combinatorial truncation framework implementation.
def GCTF(A, y):
    m, d = A.shape
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    x_est = None
    Isub_history = []
    for itertimes in range(20):
        if itertimes == 0:
            x_est = gradient_descent(A, y)
        else:
            x_est = gradient_descent(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        existence = False
        for prev_set in Isub_history:
            if np.array_equal(set_index_source, prev_set):
                existence = True
                break

        Isub_history.append(set_index_source.copy())

        C_truncated = A[:, set_index_source]
        x_subspace = gradient_descent(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8 or existence:
            break
        x_est_shangyici = x_est
    return x_est


# Add complex Gaussian noise to simulate an AWGN channel.
def AWGN(x, SNR):
    m, n = x.shape
    Eb = np.linalg.norm(x) ** 2 / m / n
    sigma = Eb / (10 ** (SNR / 10.0)) / 2.0
    noise = np.sqrt(sigma) * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    return x + noise


# C-PRIME algorithm for robust phase retrieval with thresholding.
def C_PRIME(A, y, rho, max_iter, x_init=None):
    M, N = A.shape
    if np.any(y < 0):
        raise ValueError('y must be nonnegative.')

    y_sqrt = y
    C = np.max(np.linalg.eigvalsh(A.conj().T @ A))

    if x_init is None:
        x = np.random.randn(N) + 1j * np.random.randn(N)
    else:
        x = x_init

    for _ in range(max_iter):
        Ax = A @ x
        phase_Ax = np.exp(1j * np.angle(Ax))
        c1 = x - (1.0 / C) * A.conj().T @ (Ax - y_sqrt * phase_Ax)
        x1 = np.exp(1j * np.angle(c1)) * np.maximum(np.abs(c1) - rho / (2.0 * C), 0)

        Ax1 = A @ x1
        phase_Ax1 = np.exp(1j * np.angle(Ax1))
        c2 = x1 - (1.0 / C) * A.conj().T @ (Ax1 - y_sqrt * phase_Ax1)
        x2 = np.exp(1j * np.angle(c2)) * np.maximum(np.abs(c2) - rho / (2.0 * C), 0)

        r = x1 - x
        v = x2 - x1 - r
        alpha = -np.linalg.norm(r) / np.linalg.norm(v)
        x3 = x - 2 * alpha * r + alpha ** 2 * v

        while True:
            Ax3 = A @ x3
            f3 = np.linalg.norm(y_sqrt - np.abs(Ax3)) ** 2 + rho * np.linalg.norm(x3, 1)
            Ax2 = A @ x2
            f2 = np.linalg.norm(y_sqrt - np.abs(Ax2)) ** 2 + rho * np.linalg.norm(x2, 1)
            if f3 <= f2:
                break
            alpha = (alpha - 1.0) / 2.0
            x3 = x - 2 * alpha * r + alpha ** 2 * v

        Ax3 = A @ x3
        phase_Ax3 = np.exp(1j * np.angle(Ax3))
        c3 = x3 - (1.0 / C) * A.conj().T @ (Ax3 - y_sqrt * phase_Ax3)
        x_next = np.exp(1j * np.angle(c3)) * np.maximum(np.abs(c3) - rho / (2.0 * C), 0)

        if np.linalg.norm(x_next - x) / np.linalg.norm(x) < 1e-6:
            print(f'C_PRIME Converged at iteration {_ + 1}')
            x = x_next
            break
        x = x_next

    return x


# Gradient descent initializer used by several estimators.
def gradient_descent(A, y, x_initial=None):
    m, n = A.shape
    opts = {
        'max_iter': int(1e4),
        'tol': 1e-8,
        'step_size': 1,
        'sparsity': 0.5,
        'verbose': True,
        'reg_param': 0.1,
    }

    Y = np.zeros((n, n), dtype=np.complex128)
    for i in range(m):
        ai = A[i, :]
        Y += (y[i] ** 2) * np.outer(np.conjugate(ai), ai)
    Y /= m

    V, D = largest_eig(Y)
    x0 = np.sqrt(D) * V
    x0 = x0 / np.linalg.norm(x0)

    if x_initial is None:
        scale = np.sqrt(np.sum(y ** 2) / m)
        x_init = scale * x0
    else:
        x_init = x_initial

    x_hat = x_init
    for iter_idx in range(opts['max_iter']):
        current_obj = compute_objective(A, y, x_hat)
        grad = compute_gradient(A, y, x_hat)

        step_size = 1.0
        backtrack_count = 0
        for _ in range(30):
            x_candidate = x_hat - step_size * grad
            candidate_obj = compute_objective(A, y, x_candidate)
            if candidate_obj < current_obj:
                break
            step_size *= 0.5
            backtrack_count += 1

        x_new = x_hat - step_size * grad
        if iter_idx > 0:
            rel_change = np.linalg.norm(x_new - x_hat) / np.linalg.norm(x_hat)
            if rel_change < opts['tol']:
                print(f'Gradient_Descent Converge at Iteration {iter_idx + 1}')
                break
        x_hat = x_new
    return x_hat


# PhaseLiftOff algorithm using DCA and ADMM for phase retrieval.
def phaseliftoff(y, C, xinitial=None):
    m, d = C.shape
    b = y ** 2
    A = C
    k = d
    mu = 1e-3
    lambd = mu * k / (np.sqrt(2.0) - 1.0)
    max_iter_dca = 25
    max_iter_admm = 25
    tol_dca = 1e-4
    delta = 1.0

    if xinitial is None:
        X = np.zeros((d, d), dtype=np.complex128)
    else:
        xinitial = np.asarray(xinitial, dtype=np.complex128)
        X = np.outer(xinitial, np.conjugate(xinitial))
    X_prev = X.copy()

    for _ in range(max_iter_dca):
        if np.linalg.norm(X, 'fro') > 0:
            Y = X / np.linalg.norm(X, 'fro')
        else:
            Y = np.zeros((d, d), dtype=np.complex128)

        W = lambd * (np.eye(d, dtype=np.complex128) - Y)
        X1 = np.zeros((d, d), dtype=np.complex128)
        X2 = np.zeros((d, d), dtype=np.complex128)
        X3 = np.zeros((d, d), dtype=np.complex128)
        Y1 = np.zeros((d, d), dtype=np.complex128)
        Y2 = np.zeros((d, d), dtype=np.complex128)

        M = np.zeros((d * d, d * d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                e_ij = np.zeros((d, d), dtype=np.complex128)
                e_ij[i, j] = 1.0
                vec_ij = A_adj_A_func(e_ij, A)
                M[j * d + i, :] = vec_ij.reshape(-1, order='F').conj()
        M = M + delta * np.eye(d * d, dtype=np.complex128)
        Minv = np.linalg.pinv(M)

        X3_prev = X3.copy()
        for _ in range(max_iter_admm):
            rhs_X1 = A_adj_b(A, b) - W + delta * X3 - Y1
            vec_rhs = rhs_X1.reshape(-1, order='F')
            vec_X1 = Minv @ vec_rhs
            X1 = vec_X1.reshape((d, d), order='F')

            Z = X3 - Y2 / delta
            X2 = soft_threshold(Z, mu / delta)

            Z_avg = (X1 + X2 + (Y1 + Y2) / delta) / 2.0
            X3 = proj_psd(Z_avg)

            Y1 = Y1 + delta * (X1 - X3)
            Y2 = Y2 + delta * (X2 - X3)

            prim_res = np.linalg.norm(X1 - X3, 'fro') + np.linalg.norm(X2 - X3, 'fro')
            dual_res = delta * np.linalg.norm(X3 - X3_prev, 'fro')
            if prim_res < 1e-4 and dual_res < 1e-4:
                break
            X3_prev = X3.copy()

        X_next = X3
        diff = np.linalg.norm(X_next - X, 'fro') / max(np.linalg.norm(X, 'fro'), 1.0)
        if diff < tol_dca:
            break
        X_prev = X.copy()
        X = X_next

    evals, evecs = np.linalg.eig(X)
    idx = np.argmax(evals.real)
    x_est = np.sqrt(evals[idx]) * evecs[:, idx]
    return x_est


# Alternating minimization for phase retrieval with known phase.
def AltMin(A, y, xinitial=None):
    M, N = A.shape
    if xinitial is None:
        theta = np.random.rand(M) * 2.0 * np.pi
    else:
        theta = np.angle(A @ xinitial)
    for _ in range(100):
        x_hat = np.linalg.pinv(A) @ (y * np.exp(1j * theta))
        theta = np.angle(A @ x_hat)
    return x_hat


# Sparse Wirtinger Flow solver for phase retrieval.
def SWF(A, y, x_initial=None):
    m, n = A.shape
    opts = {
        'max_iter': int(1e4),
        'tol': 1e-8,
        'step_size': 1,
        'sparsity': 0.5,
        'verbose': True,
        'reg_param': 0.1,
    }

    Y = np.zeros((n, n), dtype=np.complex128)
    for i in range(m):
        ai = A[i, :]
        Y += (y[i] ** 2) * np.outer(np.conjugate(ai), ai)
    Y = Y / m

    V, D = largest_eig(Y)
    x0 = np.sqrt(D) * V
    x0 = x0 / np.linalg.norm(x0)

    if x_initial is None:
        scale = np.sqrt(np.sum(y ** 2) / m)
        x_hat = scale * x0
    else:
        x_hat = x_initial

    for iter_idx in range(opts['max_iter']):
        current_obj = compute_objective_swf(A, y, x_hat, opts['reg_param'])
        grad = compute_gradient_swf(A, y, x_hat, opts['reg_param'])

        step_size = 1.0
        backtrack_count = 0
        for _ in range(30):
            x_candidate = x_hat - step_size * grad
            candidate_obj = compute_objective_swf(A, y, x_candidate, opts['reg_param'])
            if candidate_obj < current_obj:
                break
            step_size *= 0.5
            backtrack_count += 1

        x_new = x_hat - step_size * grad
        if iter_idx > 0:
            rel_change = np.linalg.norm(x_new - x_hat) / np.linalg.norm(x_hat)
            if rel_change < opts['tol']:
                print(f'SWF Converge at Iteration {iter_idx + 1}')
                break
        x_hat = x_new
    return x_hat


# Classical Gerchberg-Saxton phase retrieval algorithm.
def Gerchberg_Saxton(A, y, xinitial=None):
    M, N = A.shape
    if xinitial is None:
        theta = np.random.rand(M) * 2.0 * np.pi
    else:
        theta = np.angle(A @ xinitial)
    for _ in range(100):
        x_hat = np.linalg.pinv(A) @ (y * np.exp(1j * theta))
        theta = np.angle(A @ x_hat)
    return x_hat


# Helper to read options with a default value.
def get_option_phasecut(options, field, default):
    return options.get(field, default)


# Power iteration to estimate a phase vector for PhaseCut.
def phasecut_power_iteration(G, max_iter, tol, verbose):
    N = G.shape[0]
    try:
        evals, evecs = np.linalg.eig(G)
        idx = np.argmax(np.abs(evals))
        u = evecs[:, idx]
        u = u / np.abs(u)
    except Exception:
        theta = 2.0 * np.pi * np.random.rand(N)
        u = np.exp(1j * theta)

    for iter_idx in range(max_iter):
        u_new = G @ u
        u_new = u_new / (np.abs(u_new) + np.finfo(float).eps)
        change = np.linalg.norm(u - u_new) / np.sqrt(N)
        u = u_new
        if verbose and (iter_idx + 1) % 100 == 0:
            _ = np.real(np.conjugate(u) @ (G @ u))
        if change < tol:
            break
    return u


# Refine PhaseCut estimate by alternating least squares.
def refine_solution_phasecut(A, y, x_init, options):
    max_iter = get_option_phasecut(options, 'refine_iter', 50)
    tol = get_option_phasecut(options, 'refine_tol', 1e-6)
    verbose = get_option_phasecut(options, 'verbose', False)

    x = x_init
    M, N = A.shape
    for _ in range(max_iter):
        x_old = x
        Ax = A @ x
        phase_Ax = Ax / (np.abs(Ax) + np.finfo(float).eps)
        b_target = y * phase_Ax
        if N <= M:
            x = np.linalg.lstsq(A, b_target, rcond=None)[0]
        else:
            lambd = get_option_phasecut(options, 'lambda', 1e-3)
            x = np.linalg.solve(A.conj().T @ A + lambd * np.eye(N, dtype=np.complex128), A.conj().T @ b_target)
        x = x / np.linalg.norm(x)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


# PhaseCut algorithm wrapper that refines the computed phase.
def phasecut(A, y, options=None):
    if options is None:
        options = {}
    M, N = A.shape
    G = A.conj().T @ (A * (y ** 2)[:, np.newaxis])
    G = (G + G.conj().T) / 2.0
    u = phasecut_power_iteration(G, get_option_phasecut(options, 'max_iter', 2000), get_option_phasecut(options, 'tol', 1e-8), get_option_phasecut(options, 'verbose', False))
    x_init = u
    x_est = refine_solution_phasecut(A, y, x_init, options)
    return x_est


# Hard Thresholded Wirtinger Flow algorithm for sparse phase retrieval.
def hwf(A, y, K, params=None):
    if params is None:
        params = {}
    eta = params.get('eta', 1e-3)
    alpha = params.get('alpha', 0.001)
    max_iter = params.get('max_iter', 500)
    tol = params.get('tol', 1e-3)
    restarts = params.get('restarts', 1)
    kappa = params.get('kappa', 0.05)

    M, N = A.shape
    y2 = y ** 2
    theta_hat = np.sqrt(np.mean(y2))

    R = np.zeros(N)
    for i in range(N):
        R[i] = np.real(np.mean(y2 * np.abs(A[:, i]) ** 2))

    best_grad_norm = np.inf
    x_best = np.random.randn(N) + 1j * np.random.randn(N)

    for b in range(restarts):
        idx_sorted = np.argsort(-R)
        I_b = idx_sorted[b]
        u = alpha * (np.random.randn(N) + 1j * np.random.randn(N))
        v = alpha * (np.random.randn(N) + 1j * np.random.randn(N))
        u[I_b] = np.sqrt(theta_hat / np.sqrt(3.0) + alpha ** 2)

        for _ in range(max_iter):
            x = (u ** 2) - (v ** 2)
            Ax = A @ x
            grad = np.zeros(N, dtype=np.complex128)
            for j in range(M):
                grad += (np.abs(Ax[j]) ** 2 - y2[j]) * np.conjugate(Ax[j]) * A[j, :].conj()
            grad = grad / M
            u = u * (1 - 2 * eta * grad)
            v = v * (1 + 2 * eta * grad)
            if np.linalg.norm(grad) < tol:
                break

        x = (u ** 2) - (v ** 2)
        x_abs = np.abs(x)
        sort_idx = np.argsort(-x_abs)
        support = sort_idx[:K]
        x_sparse = np.zeros(N, dtype=np.complex128)
        x_sparse[support] = x[support]
        x_sparse = align_phase(x_sparse, A, y)

        Ax = A @ x_sparse
        grad_norm = np.linalg.norm((np.abs(Ax) ** 2 - y2) * np.conjugate(Ax) * (A @ x_sparse))
        if grad_norm < best_grad_norm:
            best_grad_norm = grad_norm
            x_best = x_sparse

    return x_best


# GCTF module variant using PhaseLiftOff in the subspace.
def GCTF_module3_phaseliftoff(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = gradient_descent(A, y)
        else:
            x_est = gradient_descent(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = phaseliftoff(y, C_truncated)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# GCTF module variant using SWF in the subspace.
def GCTF_module3_swf(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = gradient_descent(A, y)
        else:
            x_est = gradient_descent(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = SWF(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# GCTF module variant using C-PRIME in the subspace.
def GCTF_module3_cprime(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = gradient_descent(A, y)
        else:
            x_est = gradient_descent(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = C_PRIME(C_truncated, y, 1e-2, 500)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# GCTF module variant using Gerchberg-Saxton in the subspace.
def GCTF_module3_gs(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = gradient_descent(A, y)
        else:
            x_est = gradient_descent(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = AltMin(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# Module I variant using PhaseLiftOff after GCTF selection.
def GCTF_module1_phaseliftoff(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = phaseliftoff(y, A)
        else:
            x_est = phaseliftoff(y, A, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = AltMin(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# Module I variant using SWF after GCTF selection.
def GCTF_module1_swf(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = SWF(A, y)
        else:
            x_est = SWF(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = AltMin(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# Module I variant using C-PRIME after GCTF selection.
def GCTF_module1_cprime(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = C_PRIME(A, y, 1e-2, 500)
        else:
            x_est = C_PRIME(A, y, 1e-2, 500, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = AltMin(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# Module I variant using GS after GCTF selection.
def GCTF_module1_gs(A, y):
    m, d = A.shape
    cishu = 8
    x_est_shangyici = np.zeros(d, dtype=np.complex128)
    for itertimes in range(cishu):
        if itertimes == 0:
            x_est = AltMin(A, y)
        else:
            x_est = AltMin(A, y, x_est)

        n = d
        xtemp = np.abs(x_est) / np.linalg.norm(x_est) * np.sqrt(n)
        concentrationrate = np.sum(xtemp ** 2 + xtemp ** (-2))
        for _ in range(100):
            cap_source = xtemp ** 2
            cap_sink = 1.0 / cap_source
            cap_xianglin = concentrationrate
            edge_start = np.concatenate((np.arange(n - 1), np.arange(n - 1, 0, -1)))
            edge_end = np.concatenate((np.arange(1, n), np.arange(n - 2, -1, -1)))
            set_index_source, _ = min_cut_maxflow(cap_source, cap_sink, cap_xianglin, edge_start, edge_end)
            if len(set_index_source) == n or len(set_index_source) <= 1:
                concentrationrate *= 0.5
            else:
                break

        C_truncated = A[:, set_index_source]
        x_subspace = AltMin(C_truncated, y)
        x_est = np.zeros(d, dtype=np.complex128)
        x_est[set_index_source] = x_subspace

        if np.linalg.norm(np.abs(x_est) - np.abs(x_est_shangyici)) ** 2 / np.linalg.norm(x_est) ** 2 < 1e-8:
            break
        x_est_shangyici = x_est
    return x_est


# Main experiment script to compare all phase retrieval estimators.
def main():
    N = 32
    K = 4
    M = 96
    SNR = 10

    x0 = np.zeros(N, dtype=np.complex128)
    x0[10:K+10] = np.exp(1j * 2.0 * np.pi * np.random.rand(K))
    x0 = x0 / np.linalg.norm(x0)

    A = (1.0 / np.sqrt(2.0)) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    b = A @ x0
    y = np.abs(AWGN(b.reshape(-1, 1), SNR)).reshape(-1)

    x_est_gctf = GCTF(A, y)
    x_est_swf = SWF(A, y)
    x_est_cprime = C_PRIME(A, y, 1e-2, 500)
    x_est_saltmin = SparseAltMinPhase(A, y, 2 * K)
    x_est_taf = truncate_amplitude_flow(A, y, 2 * K)
    x_est_phaseliftoff = phaseliftoff(y, A)
    x_est_hwf = hwf(A, y, K)
    x_est_phasecut = phasecut(A, y)
    x_est_gs = Gerchberg_Saxton(A, y)

    plt.plot(np.abs(x_est_gctf), label='GCTF')
    plt.plot(np.abs(x_est_swf), label='SWF')
    plt.plot(np.abs(x_est_cprime), label='CPRIME')
    plt.plot(np.abs(x_est_saltmin), label='SALTMIN')
    plt.plot(np.abs(x_est_taf), label='TAF')
    plt.plot(np.abs(x_est_phaseliftoff), label='PhaseLiftOff')
    plt.plot(np.abs(x_est_hwf), label='HWF')
    plt.plot(np.abs(x_est_phasecut), label='PhaseCut')
    plt.plot(np.abs(x_est_gs), label='GS')
    plt.legend()
    plt.title('Phase Retrieval Estimates')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.savefig('Figure1.pdf')
    plt.show()
    plt.close()

    plt.figure()
    x_est_original = GCTF(A, y)
    x_est_module3_phaseliftoff = GCTF_module3_phaseliftoff(A, y)
    x_est_module3_swf = GCTF_module3_swf(A, y)
    x_est_module3_cprime = GCTF_module3_cprime(A, y)
    x_est_module3_gs = GCTF_module3_gs(A, y)
    x_est_module1_phaseliftoff = GCTF_module1_phaseliftoff(A, y)
    x_est_module1_swf = GCTF_module1_swf(A, y)
    x_est_module1_cprime = GCTF_module1_cprime(A, y)
    x_est_module1_gs = GCTF_module1_gs(A, y)

    plt.plot(np.abs(x_est_original), label='GCTF')
    plt.plot(np.abs(x_est_module3_phaseliftoff), label='Module III PhaseLiftOff')
    plt.plot(np.abs(x_est_module3_swf), label='Module III SWF')
    plt.plot(np.abs(x_est_module3_cprime), label='Module III C-PRIME')
    plt.plot(np.abs(x_est_module3_gs), label='Module III GS')
    plt.plot(np.abs(x_est_module1_phaseliftoff), label='Module I PhaseLiftOff')
    plt.plot(np.abs(x_est_module1_swf), label='Module I SWF')
    plt.plot(np.abs(x_est_module1_cprime), label='Module I C-PRIME')
    plt.plot(np.abs(x_est_module1_gs), label='Module I GS')
    plt.legend()
    plt.title('GCTF Module Comparisons')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.savefig('Figure2.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
