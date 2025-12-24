import numpy as np
import sys
from numba import njit
from _sgd_ocsvm import Sonar, detect_changepoint_sgd
from joblib import Parallel, delayed
from _rbf import PairRBFSampler
from threadpoolctl import threadpool_limits

# run multiple trainings of SONAR/OCSVM and aggregate statistics
def run_full(
    X, # features
    Y, # normal/outlier labels
    normal_count, 
    outlier_counts,
    lambda_param=0.01, # type 1 error bound
    n_components=100, # number of RFF's
    N=5,  # number of RFF trials
    gamma=0.5 # rbf kernel parameter
):
    # preprocessing 
    X64 = np.asarray(X, dtype=np.float64, order='C')

    if Y is None:
        Y_local = np.zeros(X64.shape[0], dtype=np.int64)
    else:
        # If Y already 0/1 (or 0/!=0), this makes it int64 binary.
        Y_local = (np.asarray(Y) != 0).astype(np.int64)

    # accumulators
    cum_errors            = np.zeros((normal_count, N))
    cum_errors_ocsvm      = np.zeros((normal_count, N))
    outlier_errors        = np.zeros((outlier_counts, N))
    outlier_errors_ocsvm  = np.zeros((outlier_counts, N))
    radii                 = np.zeros((normal_count, N))
    radii_ocsvm           = np.zeros((normal_count, N))
    f_cum_errors = np.zeros(N)           # <-- N, not normal_count!
    f_cum_errors_ocsvm = np.zeros(N)
    f_outlier_errors = np.zeros(N)       # <-- N, not outlier_counts!
    f_outlier_errors_ocsvm = np.zeros(N)
    

    for i in range(N):
        random_state = i

        # RFF transform for this trial
        transform = PairRBFSampler(
            gamma=gamma, random_state=random_state, n_components=n_components
        )
        Z = transform.fit_transform(X64)  # shape: (T+O, m)

        # run_sonar_ocsvm returns:
        (c_e, c_n_e, r, n_r, o_e, o_n_e, f_c_e, f_c_n_e, f_o_e, f_o_n_e) = run_sonar_ocsvm(
            Z, 
            Y_local, 
            normal_count, 
            outlier_counts,
            lambda_param, 
            random_state
        )

        cum_errors[:, i]           = c_e
        cum_errors_ocsvm[:, i]     = c_n_e
        outlier_errors[:, i]       = o_e
        outlier_errors_ocsvm[:, i] = o_n_e
        radii[:, i]                = r
        radii_ocsvm[:, i]          = n_r
        f_cum_errors[i]           = f_c_e
        f_cum_errors_ocsvm[i]     = f_c_n_e
        f_outlier_errors[i]       = f_o_e
        f_outlier_errors_ocsvm[i] = f_o_n_e

        print(i, end=" ", flush=True)

    avg_cum_errors           = cum_errors.mean(axis=1)
    avg_cum_errors_ocsvm     = cum_errors_ocsvm.mean(axis=1)
    avg_radii                = radii.mean(axis=1)
    avg_radii_ocsvm          = radii_ocsvm.mean(axis=1)
    avg_out_cum_errors       = outlier_errors.mean(axis=1)
    avg_out_cum_errors_ocsvm = outlier_errors_ocsvm.mean(axis=1)
    f_avg_cum_errors           = np.mean(f_cum_errors)
    f_avg_cum_errors_ocsvm     = np.mean(f_cum_errors_ocsvm)
    f_avg_out_cum_errors       = np.mean(f_outlier_errors)
    f_avg_out_cum_errors_ocsvm = np.mean(f_outlier_errors_ocsvm)

    return (avg_cum_errors, avg_cum_errors_ocsvm,
            avg_radii, avg_radii_ocsvm,
            avg_out_cum_errors, avg_out_cum_errors_ocsvm,
           f_avg_cum_errors, f_avg_cum_errors_ocsvm,
            f_avg_out_cum_errors, f_avg_out_cum_errors_ocsvm,
           )
   
# run sonar and ocsvm for given data and random state
def run_sonar_ocsvm(
    X, Y, normal_count, outlier_counts, lambda_param, random_state
):
    # models
    clf_ocsvm = Sonar(lambda_param=lambda_param, random_state=random_state,
                      max_iter=normal_count, dynamic_rate=True,
                      reg=False, labels=True)
    clf = Sonar(lambda_param=lambda_param, random_state=random_state,
                max_iter=normal_count, dynamic_rate=True,reg=True,
                labels=True)

    clf.fit(X, Y)
    clf_ocsvm.fit(X, Y)

    # cumulative arrays
    cum_err_reg  = clf.cumulative_errors_
    cum_err_nreg = clf_ocsvm.cumulative_errors_
    out_cum_reg  = np.cumsum(clf.outlier_errors_)
    out_cum_nreg = np.cumsum(clf_ocsvm.outlier_errors_)
    radii_reg    = clf.radii_
    radii_nreg   = clf_ocsvm.radii_

    # Predictions from each pipeline
    X_normal = X[Y==0]
    yhat_reg   = clf.predict(X_normal)
    yhat_ocsvm = clf_ocsvm.predict(X_normal)
    
    # Type I error = fraction of normals misclassified as outliers
    type1_reg   = np.sum(yhat_reg   == -1) / normal_count
    type1_ocsvm = np.sum(yhat_ocsvm == -1) / normal_count

    # Type II error
    X_outlier = X[Y!=0]
    yhat_reg   = clf.predict(X_outlier)
    yhat_ocsvm = clf_ocsvm.predict(X_outlier)
    
    # Type I error = fraction of outliers misclassified as normals
    type2_reg   = np.sum(yhat_reg   == 1) / outlier_counts
    type2_ocsvm = np.sum(yhat_ocsvm == 1) / outlier_counts

    
    return (cum_err_reg, cum_err_nreg,
                radii_reg, radii_nreg,
                out_cum_reg, out_cum_nreg, 
            type1_reg, type1_ocsvm, type2_reg, type2_ocsvm)


# helper function to build dyads & restart schedules once
def _restart_schedules(T, dyads):
    schedules = []
    for d in dyads:
        phase = T // d
        schedules.append([phase * k for k in range(1, d + 1)])
    return schedules

# run SONARC over multiple RFF draws
def run_full_cpd(
    X, Y, normal_count, outlier_counts,
    lambda_param=0.01, n_components=100, N=5, gamma=0.5,
    use_float32=False, n_jobs=-1
):
    
    # preprocessing
    X = np.asarray(X, dtype=np.float32 if use_float32 else np.float64, order='C')
    if Y is None:
        Y_fit = None
    else:
        Y = np.asarray(Y, dtype=np.float64, order='C')
        Y_fit = Y
    Y = Y_fit

    T = int(normal_count)
    O = int(outlier_counts)

    dyads = dyads_up_to_T(T)                 # compute once
    restart_schedules = _restart_schedules(T, dyads)
    
    # accumulators
    cum_errors_detect      = np.zeros((T, N))
    outlier_errors_detect  = np.zeros((O, N))
    radii_detect           = np.zeros((T, N))
    f_cum_errors_detect      = np.zeros(N)
    f_outlier_errors_detect  = np.zeros(N)
    detected_changes = np.zeros(N, dtype=int)   # <-- NEW

    # RFF trials
    for i in range(N):
        random_state = i

        # draw RFF's
        rff = PairRBFSampler(gamma=gamma, random_state=random_state, n_components=n_components)
        Z = rff.fit_transform(X)  # shape: (T+O, m)


        # threshold for CPD
        chosen_threshold = 1e-3
    
        # run SONARC on precomputed features Z
        (c_d, r_d, o_d, nchg, f1_d, f2_d) = run_sonarc(
            Z, Y_fit, T, O, lambda_param, restart_schedules, chosen_threshold, random_state, n_jobs=n_jobs
        )

        detected_changes[i] = nchg            # number of detected changes
        cum_errors_detect[:, i]       = c_d
        radii_detect[:, i]            = r_d
        outlier_errors_detect[:, i]   = o_d
        f_cum_errors_detect[i]       = f1_d
        f_outlier_errors_detect[i]   = f2_d
        

        print(i, end=" ", flush=True)

    # ---- 3) averages across trials ----
    avg_cum_errors_detect        = cum_errors_detect.mean(axis=1)
    avg_radii_detect             = radii_detect.mean(axis=1)
    avg_out_cum_errors_detect    = outlier_errors_detect.mean(axis=1)
    f_avg_cum_errors_detect        = np.mean(f_cum_errors_detect)
    f_avg_out_cum_errors_detect    = np.mean(f_outlier_errors_detect)

    return (avg_cum_errors_detect,
            avg_radii_detect,
            avg_out_cum_errors_detect,
                detected_changes.mean(),
            f_avg_cum_errors_detect,
             f_avg_out_cum_errors_detect,
           )

# run sonarc for given data and random state
def run_sonarc(Z, Y, T, O, lambda_param, restart_schedules, chosen_threshold, random_state, n_jobs=-1):
    K = len(restart_schedules)

    # fit K base learners in parallel, all on Z
    def _fit_one(restarts):
        snapshots_j = [r - 1 for r in restarts]
        clf = Sonar(
            lambda_param=lambda_param,
            random_state=random_state,
            max_iter=T,
            snapshot_iters=snapshots_j,
            restarts=restarts,
            labels=True,
        )
        clf.fit(Z, Y)
        return (clf.cumulative_errors_, clf.radii_, clf.outlier_errors_, clf.snapshots_,
                clf.coef_.copy(), float(clf.offset_))

    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one)(rs) for rs in restart_schedules
        )

    # collect
    snapshots_all     = []
    for j, (cums, rad, outl, snaps, wj, rhoj) in enumerate(results):
        snapshots_all.append(snaps)

    # SONARC detect (single run) on Z
    clf_detect = Sonar(
        lambda_param=lambda_param,
        random_state=random_state,
        max_iter=T,
        threshold=chosen_threshold,
        detect=True,
        base_schedules=restart_schedules,
        detect_snapshots=snapshots_all,
        labels=True,
    )
    clf_detect.fit_cpd(Z, Y)

    num_changes = clf_detect.detected_changes_

    print(f"Detected {num_changes} changes")
    print(f"Detected changepoints at normal steps: {clf_detect.detected_changepoint_steps_}")

    # Compute final type 1 and 2 errors
    X_normal = Z[Y == 0]
    X_outlier = Z[Y != 0]
    
    actual_normal_count = len(X_normal)
    actual_outlier_count = len(X_outlier)

    # ---- SONARC predictions ----
    yhat_detect_normal = clf_detect.predict(X_normal)
    yhat_detect_outlier = clf_detect.predict(X_outlier)
    
    type1_detect = np.sum(yhat_detect_normal == -1) / actual_normal_count
    type2_detect = np.sum(yhat_detect_outlier == 1) / actual_outlier_count

    return (clf_detect.cumulative_errors_,
            clf_detect.radii_,
            np.cumsum(clf_detect.outlier_errors_),
            num_changes,
            type1_detect, type2_detect)

# helper class for online standardization of data
class OnlineStandardizer:
    def __init__(self, D, eps=1e-8):
        self.D = D
        self.eps = eps
        self.n = 0
        self.mean = np.zeros(D)
        self.M2 = np.zeros(D)  # Sum of squared differences from mean
    
    def update_and_transform(self, x):
        """
        Update statistics with new sample x and return standardized version.
        Uses Welford's online algorithm for variance.
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        if self.n > 1:
            std = np.sqrt(self.M2 / self.n)
            x_std = (x - self.mean) / (std + self.eps)
        else:
            x_std = np.zeros_like(x)  # Return zeros for first sample
        
        return x_std

# helper class for writing simultaneously to stdout and an output file
class TeeLogger:
    """Writes everything to both stdout and a file."""
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout  # keep original stdout

    def write(self, message):
        self.stdout.write(message)  # print to notebook/terminal
        self.file.write(message)    # also write to file

    def flush(self):
        # needed so Jupyter prints immediately
        self.stdout.flush()
        self.file.flush()

# Return powers of two <= T using NumPy.
def dyads_up_to_T(T):
    upper = min(T, 256)
    max_power = int(np.floor(np.log2(upper)))  # largest power s.t. 2**p <= T
    return (2 ** np.arange(0, max_power + 1)).tolist()