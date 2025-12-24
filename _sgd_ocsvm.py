import numpy as np
from numba import njit
from numba.typed import List
    
# class for SONAR/OCSVM object
class Sonar():
    def __init__(self,
                 lambda_param=0.1, # bound on type 1 error
                 learning_rate=1e-3, # learning rate of SGD
                 max_iter=1000, # normal data count
                 random_state=None, 
                 reg=True, # SONAR L2 regularization
                 snapshot_iters=[], 
                 dynamic_rate=True, # time-varying learning rate
                 restarts=[], 
                 detect=False, # changepoint detection
                 base_schedules = [], 
                 detect_snapshots=[], 
                 threshold=1,
                 labels=False, 
                 adagrad=True,  
                    ):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.reg = reg
        self.snapshot_iters = snapshot_iters
        self.dynamic_rate = dynamic_rate
        self.restarts = restarts
        self.detect = detect
        self.threshold = threshold
        self.detect_snapshots = detect_snapshots
        self.base_schedules = base_schedules
        self.labels=labels
        self.adagrad=adagrad

    # train SONAR or OCSVM
    def fit(self, X, y=None):
        # hyperparameters
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # parameter initialization
        w_init = (lambda x: x / np.linalg.norm(x))(rng.randn(n_features)) #rng.randn(n_features) #* self.lambda_param 
        rho_init = 1 

        # processing inputs
        snapshot_points = np.array(self.snapshot_iters, dtype=np.int64)
        restarts = np.array(self.restarts, dtype=np.int64)
        n_samples, n_features = X.shape
        
        if y is None:
            y_int = np.zeros(n_samples, dtype=np.int64)
        else:
            y_int = (np.asarray(y) != 0).astype(np.int64)
        
        restarts = np.asarray(self.restarts, dtype=np.int64)
        snapshot_points = np.asarray(self.snapshot_iters if self.snapshot_iters is not None else np.empty(0, np.int64),
                                     dtype=np.int64)

        # run SGD training loop with numba
        self.coef_, self.offset_, self.snapshots_, self.cumulative_errors_, self.radii_, self.outlier_errors_ = numba_sgd(
            X=X.astype(np.float64, copy=False),
            w_init=w_init.astype(np.float64, copy=False),
            rho_init = float(rho_init),
            learning_rate=self.learning_rate,
            max_steps=self.max_iter,
            lambda_param=self.lambda_param,
            reg=self.reg,
            dynamic_rate=self.dynamic_rate,
            restarts=restarts,
            snapshot_points=snapshot_points,
            Y=y_int,                 
            adagrad=self.adagrad
        )
        return self

    # train SONARC on data
    def fit_cpd(self, X, y=None):

        # initialization and hyperparameters
        rng = np.random.RandomState(self.random_state)
    
        # process inputs
        X = np.asarray(X, dtype=np.float64, order="C")
        n_samples, n_features = X.shape
        if (y is None) or (not getattr(self, "labels", False)):
            Y_int = np.zeros(n_samples, dtype=np.int8)
        else:
            Y_arr = np.asarray(y)
            Y_int = (Y_arr != 0).astype(np.int8)
    
        # parameter initialization
        w_init = (lambda x: x / np.linalg.norm(x))(rng.randn(n_features)) #(rng.randn(n_features) * float(self.lambda_param)).astype(np.float64)
        rho_init = 1 #float(self.lambda_param)
    
        # -process inputs
        max_steps = int(self.max_iter)
    
        restarts = np.asarray(getattr(self, "restarts", []), dtype=np.int64)    
        base_schedules = getattr(self, "base_schedules", [])
        S, Slen, SnapIdx = pack_schedules(base_schedules, max_steps)

        detect_snapshots = getattr(self, "detect_snapshots", [])
        if getattr(self, "detect", False) and len(detect_snapshots) > 0:
            Wref, Rhoref, RefLen = pack_detect_snapshots(detect_snapshots, n_features)
        else:
            # minimal dummy arrays so Numba signature is satisfied
            Wref   = np.zeros((1, 1, n_features), dtype=np.float64)
            Rhoref = np.zeros((1, 1), dtype=np.float64)
            RefLen = np.zeros(1, dtype=np.int64)
            # also ensure S, Slen, SnapIdx are at least 1x1 so shapes are valid
            if S.size == 0:
                S      = np.zeros((1, 1), dtype=np.int64)
                Slen   = np.zeros(1, dtype=np.int64)
                SnapIdx= np.full((1, 1), -1, dtype=np.int64)
    
        # set inputs
        learning_rate = float(getattr(self, "learning_rate", 1e-3))
        lambda_param  = float(self.lambda_param)
        reg           = bool(getattr(self, "reg", True))
        dynamic_rate  = bool(getattr(self, "dynamic_rate", True))
        detect        = bool(getattr(self, "detect", False))
        threshold     = float(getattr(self, "threshold", 1.0))
        adagrad       = bool(getattr(self, "adagrad", True))
        adagrad_eps   = float(getattr(self, "adagrad_eps", 1e-8))

        snapshot_points = np.asarray(self.snapshot_iters if self.snapshot_iters is not None else np.empty(0, np.int64),
                                         dtype=np.int64)
    
        # SGD training loop
        (coef, offset, cumulative_errors, detected_changes,
         radii, outlier_errors, self.snapshots_) = sgd_cpd(
            X, Y_int,
            w_init, rho_init,
            max_steps, lambda_param,
            reg, dynamic_rate, detect,
            S, Slen, SnapIdx,
            Wref, Rhoref, RefLen,
            threshold,
            adagrad, adagrad_eps,
             snapshot_points=snapshot_points
        )
    
        # store output after training
        self.coef_ = coef
        self.offset_ = offset
        self.cumulative_errors_ = cumulative_errors
        self.detected_changepoint_steps_ = detected_changes
        self.detected_changes_ = np.atleast_1d(detected_changes).size 
        self.radii_ = radii
        self.outlier_errors_ = outlier_errors
    
        return self

    # evaluate decision function on data
    def decision_function(self, X):
        return np.dot(X, self.coef_) - self.offset_

    # compute decision function at a snapshot
    def snapshot_decision_function(self, X, snapshot_index):
        w_snap, rho_snap = self.snapshots_[snapshot_index]
        return np.dot(X, w_snap) - rho_snap
        
    # predict normal or outlier data based on classifications
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


    
# SGD training loop with numba for SONAR/OCSVM
@njit(cache=True, fastmath=False)   # cache the compiled version
def numba_sgd(X, 
              w_init, # initial w
              rho_init, # initial rho
              learning_rate=1e-3, # learning rate if fixed
              max_steps=1000, # normal data count
              lambda_param=0.1, # type 1 error bound
              reg=True, # Sonar L2 regularization
              dynamic_rate=True, # dynamic learning rate
              restarts=np.empty(0, np.int64), # restart steps
              snapshot_points=np.empty(0, np.int64), # times to take snapshots of decision boundaries
              Y=np.empty(0, np.int64), # normal/outlier labels
              adagrad=True, # adagrad normalization
              adagrad_eps=1e-8, # adagrad parameter
             ): 
    
    n_samples, n_features = X.shape

    # initialize parameters
    w = w_init.copy()
    rho = rho_init
    
    snapshots = []

    # ensure stable dtypes for Numba math
    w = w.astype(np.float64)
    rho = float(rho)

    # allocate outputs with fixed dtypes/sizes
    cumulative_errors = np.zeros(max_steps, dtype=np.float64) # cum. type 1 errors
    radii             = np.zeros(max_steps, dtype=np.float64) # iterates' margins
    cumulative_error  = 0.0 # tracking cum. type 1 error
    outlier_errors = np.zeros(Y[Y != 0].shape[0], dtype=np.float64) # outlier errors

    # other accumulators for adagrad
    g2_w  = np.zeros_like(w)   
    g2_rho = 0.0               
    grad_rho = 0.0

    # Bookkeeping for control-flow
    latest_restart   = 0
    detected_changes = 0            
    normal_step      = 0 # normal data step count
    ostep            = 0 # outlier step count
    step             = 0 # aggregate step count

    # snapshots
    if snapshot_points.size > 0:
        sp = snapshot_points
    else:
        sp = np.empty(0, np.int64)
    snapshot_idx = 0 

    # bottou heuristic
    alpha   = (lambda_param / 2.0)
    t0_b    = bottou_t0(alpha)
    t0_b_reg= bottou_t0(0.5)

    # track restarts
    restart_idx = 0
    next_restart = restarts[0] if restarts.size > 0 else -1
    

    while normal_step < max_steps:
        x = X[step]
        margin_current = np.dot(w, x)

        if Y[step]==0: # step with normal data
            
            # Check if we're at a restart point
            if restart_idx < restarts.size and (normal_step + 1) == restarts[restart_idx]:
                # reset ADAGRAD accumulators
                g2_w = np.zeros_like(w)
                g2_rho = 0.0
                
                restart_idx += 1
            
            # Compute gradients on current iterate
            if margin_current < rho:
                grad_w = -x
                grad_rho = 1.0
            else:
                grad_w = np.zeros_like(w)
                grad_rho = 0.0
                
            if reg:
                grad_w += w
                grad_rho += rho - lambda_param
            else:
                grad_w += w * lambda_param
                grad_rho -= lambda_param

            # Step size
            if not dynamic_rate:
                eta = learning_rate
            else:
                if reg:
                    eff_t = effective_step_from_restarts(normal_step + 1, restarts)
                    eta = bottou_eta(eff_t, 0.5, t0_b_reg)
                else:
                    eff_t = effective_step_from_restarts(normal_step + 1, restarts)
                    eta = bottou_eta(eff_t, alpha, t0_b)

            # Update parameters
            if adagrad:
                g2_w += grad_w * grad_w
                g2_rho += grad_rho * grad_rho

                w  -= eta * (grad_w / (np.sqrt(g2_w) + adagrad_eps))
                rho -= eta * (grad_rho / (np.sqrt(g2_rho) + adagrad_eps))
            else:
                w  -= eta * grad_w
                rho -= eta * grad_rho


            # compute metrics on current iterate
            margin = np.dot(w, x)
            
            # compute type 1 error on current iterate
            if margin < rho:
                cumulative_error += 1
            cumulative_errors[normal_step] = cumulative_error

            # store margin 
            norm_w = np.sqrt(np.dot(w, w))
            if norm_w != 0 and rho / norm_w <= 1:
                radii[normal_step] = rho / norm_w
            else:
                radii[normal_step] = 1
            
            normal_step += 1   

            # Save snapshot if we've reached snapshot point
            if (snapshot_points is not None) and (snapshot_idx < snapshot_points.shape[0]):
                if (step + 1) == snapshot_points[snapshot_idx]:
                    snapshots.append((w.copy(), rho))
                    snapshot_idx += 1
                    
        else: # step with outlier data
            margin_eval = 0.0
            for j in range(n_features):
                margin_eval += w[j] * x[j]
            margin_eval -= rho
            
            if margin_eval >= 0:
                outlier_errors[ostep] = 1.0
            ostep += 1

        step += 1 # update step

    return w, rho, snapshots, cumulative_errors, radii, outlier_errors

# sgd training loop for SONARC
@njit(cache=True, fastmath=False)
def sgd_cpd(
    X,               # data
    Y_int,           # labels
    w_init,          
    rho_init,        
    max_steps,       # normal data count
    lambda_param,    # type 1 error bound
    reg,             # SONARC L2 regularization
    dynamic_rate,    # set time-varying learning rate
    detect,          # bool to activate CPD
    S, Slen, SnapIdx,# schedules: (K,M) int64, (K,), (K,M)
    Wref, Rhoref, RefLen,  # references: (K,M,d),(K,M),(K,)
    threshold,       # threshold for CPD
    adagrad, adagrad_eps,  # adagrad
    snapshot_points=None   # array of snapshot points
):
    T = max_steps
    n_total, d = X.shape
    K = S.shape[0]

    w = w_init.copy()
    rho = rho_init

    # smoothed iterates for CPD
    smooth_window = 50 
    w_smooth = w.copy()
    rho_smooth = rho
    smooth_alpha = 2.0 / (smooth_window + 1)

    # adagrad accumulators
    g2_w = np.zeros(d) if adagrad else np.zeros(1)
    g2_rho = 0.0

    cumulative_errors = np.zeros(T)
    radii = np.zeros(T)

    # snapshots
    snapshots = []
    snapshot_idx = 0
    if snapshot_points is None:
        snapshot_points = np.empty(0, dtype=np.int64)

    # set outlier_errors array from labels
    O = 0
    for t in range(n_total):
        if Y_int[t] == 1:
            O += 1
    outlier_errors = np.zeros(O)

    detected_changes = 0
    latest_restart = 0

    alpha = 0.5 * lambda_param
    t0_b = bottou_t0(alpha)
    t0_b_reg = bottou_t0(0.5)

    total_seen = 0
    normal_step = 0
    ostep = 0

    # reusable gradient buffer
    grad_w = np.zeros(d)

    detected_changepoint_steps = []

    # set burn-in period for CPD
    # burn_in = 5000 # for synthetic
    burn_in = 50 # for real datasets or skab_all

    while normal_step < T:
        i = total_seen
        x = X[i]

        # compute margin on current iterate
        margin_current = 0.0
        for j in range(d):
            margin_current += w[j] * x[j]

        if Y_int[i] == 0: # step with normal data
            if margin_current < rho:
                for j in range(d):
                    grad_w[j] = -x[j]
                grad_rho = 1.0
            else:
                for j in range(d):
                    grad_w[j] = 0.0
                grad_rho = 0.0

            if reg:
                for j in range(d):
                    grad_w[j] += w[j]
                grad_rho += (rho - lambda_param)
            else:
                for j in range(d):
                    grad_w[j] += w[j] * lambda_param
                grad_rho -= lambda_param

            # step size
            if not dynamic_rate:
                eta = 1e-3
            else:
                eff_t = (normal_step - latest_restart + 1)
                if eff_t < 1:
                    eff_t = 1
                if reg:
                    eta = bottou_eta(eff_t, 0.5, t0_b_reg)
                else:
                    eta = bottou_eta(eff_t, alpha, t0_b)

            # Update (AdaGrad or plain)
            if adagrad:
                for j in range(d):
                    g2_w[j] += grad_w[j]*grad_w[j]
                    w[j] -= eta * (grad_w[j] / (np.sqrt(g2_w[j]) + adagrad_eps))
                g2_rho += grad_rho*grad_rho
                rho -= eta * (grad_rho / (np.sqrt(g2_rho) + adagrad_eps))
            else:
                for j in range(d):
                    w[j] -= eta * grad_w[j]
                rho -= eta * grad_rho

            # Update smoothed iterates using exponential moving average
            for j in range(d):
                w_smooth[j] = smooth_alpha * w[j] + (1 - smooth_alpha) * w_smooth[j]
            rho_smooth = smooth_alpha * rho + (1 - smooth_alpha) * rho_smooth

            # recompute margin on UPDATED iterate
            margin_updated = 0.0
            for j in range(d):
                margin_updated += w[j] * x[j]
            
            # use margin_updated instead of margin_current
            if margin_updated < rho:
                if normal_step == 0:
                    cumulative_errors[0] = 1.0
                else:
                    cumulative_errors[normal_step] = cumulative_errors[normal_step - 1] + 1.0
            else:
                if normal_step == 0:
                    cumulative_errors[0] = 0.0
                else:
                    cumulative_errors[normal_step] = cumulative_errors[normal_step - 1]


            # radii on current iterate
            norm_w = 0.0
            for j in range(d):
                norm_w += w[j]*w[j]
            norm_w = np.sqrt(norm_w)
            if norm_w != 0.0 and rho / norm_w <= 1.0:
                radii[normal_step] = rho / norm_w
            else:
                radii[normal_step] = 1.0

            # changepoint detection (uses smoothed iterate for distance computation)
            if detect and (normal_step - latest_restart > burn_in) and K > 0:
                fired = False
                for a in range(K):
                    rlen = Slen[a]
                    max_idx = -1
                    max_val = -1
                    
                    # find the restart period length for this base learner
                    period_length = Slen[a] if Slen[a] > 0 else T
                    
                    for k in range(rlen):
                        si = SnapIdx[a, k]
                        # Require that entire period is after latest_restart
                        period_start = si - period_length if si >= period_length else 0
                        
                        if si >= 0 and si < T and si < normal_step and period_start > latest_restart:
                            if si > max_val:
                                max_val = si
                                max_idx = k
                    
                    if max_idx == -1 or max_idx >= RefLen[a]:
                        continue
                    
                    # compute distance using smoothed iterate
                    dist = 0.0
                    rrho = Rhoref[a, max_idx]
                    dr = rho_smooth - rrho
                    dist += dr*dr
                    for j in range(d):
                        dw = w_smooth[j] - Wref[a, max_idx, j]
                        dist += dw*dw
                    
                    replay_len = 1
                    for k in range(rlen):
                        si0 = SnapIdx[a, k]
                        if si0 >= 0:
                            replay_len = Slen[a] if Slen[a] > 0 else 1
                            break
                    
                    if dist > threshold * np.log(T) * np.log(1/lambda_param) / replay_len:
                        detected_changes += 1
                        latest_restart = normal_step
                        detected_changepoint_steps.append(normal_step)

                        # reset ADAGRAD accumulators
                        if adagrad:
                            for j in range(d):
                                g2_w[j] = 0.0
                            g2_rho = 0.0

                        # RESET SMOOTHED iterates
                        w_smooth = w.copy()
                        rho_smooth = rho
                        
                        fired = True
                        break
                if fired:
                    pass

            total_seen  += 1
            normal_step += 1

            # save snapshot if we've reached snapshot point
            if snapshot_idx < snapshot_points.shape[0]:
                if total_seen == snapshot_points[snapshot_idx]:
                    w_snapshot = w_smooth.copy()
                    snapshots.append((w_snapshot, rho_smooth))
                    snapshot_idx += 1
                    
        else: # outlier step
            margin_eval = 0.0
            for j in range(d):
                margin_eval += w[j] * x[j]
            margin_eval -= rho
            
            if ostep < O and margin_eval >= 0:
                outlier_errors[ostep] = 1.0
            ostep += 1
            total_seen += 1
    return w, rho, cumulative_errors, detected_changepoint_steps, radii, outlier_errors, snapshots

# compute effective time elapsed since last restart
@njit
def effective_step_from_restarts(step, restarts):
    if restarts.shape[0] == 0:
        return step
    latest_restart = -1
    for i in range(restarts.shape[0]):
        if restarts[i] <= step:
            latest_restart = restarts[i]
    if latest_restart == -1:
        return step
    return step - latest_restart

# compute bottou heuristic parameter
@njit
def bottou_t0(alpha):
    typw = np.sqrt(1.0 / np.sqrt(alpha))
    initial_eta0 = typw
    return 1.0 / (alpha * initial_eta0)

# compute bottou heuristic stepsize
@njit
def bottou_eta(effective_t, alpha, t0):
    t = max(effective_t, 1)  # guard
    return 1.0 / (alpha * (t0 + t - 1.0))

# helper function for packing restart schedules of base algorithms
def pack_schedules(restart_schedules, T):
    K = len(restart_schedules)
    M = max((len(rs) for rs in restart_schedules), default=0)
    S = np.full((K, M), -1, dtype=np.int64)       # restart times
    SnapIdx = np.full((K, M), -1, dtype=np.int64) # snapshot indices (r-1 on normal timeline)
    Slen = np.zeros(K, dtype=np.int64)
    for j, rs in enumerate(restart_schedules):
        Slen[j] = len(rs)
        for k, r in enumerate(rs):
            S[j, k] = r
            si = r - 1
            if 0 <= si < T:
                SnapIdx[j, k] = si
    return S, Slen, SnapIdx

# helper function for packing snapshots of base algorithms
def pack_detect_snapshots(detect_snapshots, d):
    K = len(detect_snapshots)
    M = max((len(lst) for lst in detect_snapshots), default=0)
    Wref = np.zeros((K, M, d), dtype=np.float64)
    Rhoref = np.zeros((K, M), dtype=np.float64)
    RefLen = np.zeros(K, dtype=np.int64)
    for j, lst in enumerate(detect_snapshots):
        RefLen[j] = len(lst)
        for k, (w_ref, rho_ref) in enumerate(lst):
            Wref[j, k, :] = w_ref
            Rhoref[j, k]  = rho_ref
    return Wref, Rhoref, RefLen

# tune changepoint detection threshold for SONARC
def detect_changepoint_sgd(X, random_state, lambda_param, threshold=25.0, pruning_params=(2, 1), dyads=[1,2,4,8,16]):
    T = X.shape[0]
    restart_schedules_temp = []  # list of list of restart times
    for i in dyads:
        phase_length = T // i
        restarts_for_i = [phase_length * k for k in range(1, i + 1)]
        restart_schedules_temp.append(restarts_for_i)

    snapshots_all_temp = []  # to store snapshots from each model
    j=0
    for j, restarts in enumerate(restart_schedules_temp):
        snapshots_j = [r - 1 for r in restarts]
        clf_temp = Sonar(lambda_param=lambda_param, 
                                    random_state=random_state, 
                                    max_iter=T, 
                                    snapshot_iters=snapshots_j,
                                    restarts=restarts,
                                )
        clf_temp.fit(X)
        snapshots_all_temp.append(clf_temp.snapshots_)  # store snapshots
        j+=1
    clf_detect_temp = Sonar(lambda_param=lambda_param, 
                                random_state=random_state, 
                                max_iter=T, 
                                snapshot_iters=[],
                                threshold = threshold,
                                detect=True,
                                base_schedules = restart_schedules_temp,
                                detect_snapshots = snapshots_all_temp,
                            )
    clf_detect_temp.fit_cpd(X)
    if clf_detect_temp.detected_changes_ > 0:
        return True
    return False  # No changepoint found


    