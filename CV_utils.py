import cv2
import pandas as pd
import os
import numpy as np
import sklearn.model_selection as sm
import sklearn.linear_model as sl
import sklearn.svm as ss
import sklearn.kernel_ridge as sk
import sklearn.tree as st
import time
from six.moves import range
import multiprocessing

from group_lasso_utils.regain.admm.lasso_ import lu_factor
from group_lasso_utils.regain.prox import soft_thresholding
from group_lasso_utils.regain.utils import flatten

#8 unitary normal vectors of reference
a_0 = [1, 0]
a_45 = [1, 1]/np.sqrt(2)
a_90 = [0, 1]
a_135 = [-1, 1]/np.sqrt(2)
a_180 = [-1, 0]
a_225 = [-1, -1]/np.sqrt(2)
a_270 = [0, -1]
a_315 = [1, -1]/np.sqrt(2)

ref_angles = [a_0, a_45, a_90, a_135, a_180, a_225, a_270, a_315]

len_ref_angles = len(ref_angles)


################ LOADING & PREPROCESSING ################

def load_marks(reduced = None):
    
    t = pd.read_csv('datasets/ICMI_ANNOTATION.csv', sep=';')
    ff = os.listdir('Video/DANCE_Platform_Tools')
    ft = list(t['Name'])
    not_to_pick = set(ff)-set(ft)
    
    t = t[~t['Name'].isin(not_to_pick)]
    
    if reduced: t = t.head(reduced)
    
    return t['Name'], t[['mean lightness', 'mean fragility']]


def cut_relevant_part(image, width = None):
    
    indexes = np.argwhere(image > 0)[:, 1] 
    left = np.min(indexes)
    right = np.max(indexes)+1
    
    if width is not None:
        
        diff = width - right + left
        
        if diff != 0:          
            right += int(np.ceil(diff/2))
            left -= int(np.floor(diff/2))
            
            
    return image[:, left:right], (left, right)


def get_binary_shape(gray_frame):
    
    # first thresholding: keep black-ish element in the frame
    ret, thresh = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY_INV)
    # gaussian blur: reduce noise of black isolated elements
    blurred = cv2.blur(thresh, (10, 10));
    # second thresholding
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY);
    
    return thresh


def get_grayscale_shape(gray_frame, binary_frame):
    
    return np.multiply(gray_frame, (binary_frame/255))

    
def read_frames(video):
    
    cap = cv2.VideoCapture(video)
    b_frames = []
    bgs_frames = []
    c_frames = []
    gs_frames = []
    
    while(True):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
            
        c_frames.append(frame)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gs_frames.append(gray_frame)
        
        binary_frame = get_binary_shape(gray_frame)
        b_frames.append(binary_frame)
        bgs_frames.append(get_grayscale_shape(gray_frame, binary_frame))
    
    cap.release()
    
    return b_frames, bgs_frames, c_frames, gs_frames


def binary_read(path_to_read_binary, n=None, batch_size=10, limit_videos=0, verbose=False):
        
    names = os.listdir(path_to_read_binary)
    names = list(set(names) & set(load_marks()[0]))
    if limit_videos == 0: limit_videos=len(names)
    names = sorted(names)[:limit_videos]
    if n: names = names[(n-1)*batch_size:n*batch_size]
    
    video_list = []
    
    # binary video loading
    for i, name in enumerate(names):
        temp_bin_frames = []
        if verbose: print('converting {} : {} out of {}'.format(name, i+1, len(names)))
        cap = cv2.VideoCapture(path_to_read_binary+name)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret: break
            temp_bin_frames.append(frame[:,:,0])

        video_list.append(temp_bin_frames)
        cap.release()
    
    return video_list


def reduce_video_len(frame_list, n_frames):
    
    l = len(frame_list)
    diff = l - n_frames
    base = l//(diff+1)
    
    while diff > 0:
        
        del frame_list[base*diff]
        diff -= 1
        
    return frame_list


def convert_save_binaryVideo (folderIn='Video/DANCE_Platform_Tools/', folderOut='Video/binary_video/', verbose=False):
    
    for i, name in enumerate(os.listdir(folderIn)):
        if verbose: print('converting {} ; #{} out of {}'.format(name, i+1, len(os.listdir(folderIn))))
        l_b_frames = read_frames(folderIn+name)[0]
        height, width =  l_b_frames[0].shape
        video = cv2.VideoWriter(folderOut+name, cv2.VideoWriter_fourcc(*'mpeg'), fps=50, frameSize=(width,height), isColor=False)

        for f in l_b_frames: video.write(f)

        video.release()

def extract_patches_mean(video, patch_size):
    # WE ARE SUPPOSING img.shape[0&1]%patch_size == 0
    
    patches_mean_list = []
    
    for img in video:
        img_shape = img.shape

        # compare img size with patch size
        if img_shape[0]<patch_size or img_shape[1]<patch_size or (img.shape[0]%patch_size)!=0 or (img.shape[1]%patch_size)!=0:
            raise Exception("Inappropriate patch size. Image shape {}, patch size ({}, {})".format(img_shape, patch_size, patch_size))

        # how many patches can fit the img height
        shape_x = img_shape[0]//patch_size

        # how many patches can fit the img length    
        shape_y = img_shape[1]//patch_size

        patches_mean = np.zeros((shape_x, shape_y))

        h = 0
        for i in range(shape_x):
            l = 0
            for j in range(shape_y):       
                patches_mean[i, j] = np.mean(img[h:h+patch_size, l:l+patch_size])
                l += patch_size            
            h += patch_size
        
        patches_mean_list.append(patches_mean.reshape(-1))
                                 
    return patches_mean_list

def normalize_preprocessing(video_frames_list, pd_density_frames, width=None, n_frames = 500, verbose=False):
    
    indexes_deleted = []
    norm_videos = []
           
    for i, video_frames in enumerate(video_frames_list):
                
        if verbose: print('\nWorking on video {}/{}'.format(i+1, len(video_frames_list)))
        
        l = len(video_frames)
        
        if n_frames and l != n_frames:
            
            if l < n_frames:                 
                if verbose: print('\t .... Video discarded')                    
                indexes_deleted.append(i)
                continue
                
            else :                
                if verbose: print('\t .... Cleaning the frame list')                
                video_frames = reduce_video_len(video_frames, n_frames) 
        
        if verbose: print('\t .... Cutting frames')            
        density_frame = pd_density_frames.iloc[i, 0]
        density_frame, edges = cut_relevant_part(density_frame, width)
        
        video_frames = [x[:, edges[0]:edges[1]] for x in video_frames]                        
        if verbose: print('\t .... Cutting complete')
                    
        norm_videos.append(video_frames)
            
    return norm_videos, indexes_deleted


def cut_non_relevant_features(X):
    
    indexes = np.where(~X.any(axis=0))[0]
    
    return indexes, X[:, ~indexes]


def preprocess_data(names, reading_path = 'Video/DANCE_Platform_Tools/', writing_path = 'Video/'):
    
    #loading and saving 10 videos at time

    videos_frames_list = []
    total_video_df_list = []

    tot = len(names)
    j = 0
    stop_flag = False

    if len(names) > 10:

        n = names[:10]
        remaining = names[10:]

    while True :

        j+=1 #used to differentiate the name of the pickles

        for i, name in enumerate(n):

            print("Getting frames of video {}. {}/{}".format(name, (i+1)+10*(j-1), tot))
            videos_frames_list.append(cvu.read_frames(reading_path+name)[0])

        print("Creating dataframe from video frames")
        df = pd.DataFrame(videos_frames_list, index = n)
        print("Creating the pickle file for video frames")
        df.to_pickle(writing_path+'videos_frames_'+str(j)+'.pkl')

        for i, video_frames in enumerate(videos_frames_list):

            print("Getting density frame of whole video {}/{}".format((i+1)+10*(j-1), tot))
            total_video_df_list.append(normalised_binary_density(video_frames, frame_size=video_frames[0].shape, batch_size=len(video_frames)))

        print("Creating dataframe from density frames")
        df = pd.DataFrame(total_video_df_list, index = n)
        print("Creating the pickle file for density frames")
        df.to_pickle(writing_path+'videos_density_'+str(j)+'.pkl')

        if len(remaining) <= 10:

            if stop_flag == False:
                n = remaining
                stop_flag = True
            else:
                break #exiting from the while

        else:
            n = remaining[:10]
            remaining = remaining[10:]

        videos_frames_list = []
        total_video_df_list = []
        
        
def merge_child(return_dict, index, path_to_read):
    
    X = pd.read_csv(path_to_read+str(index)+".csv").as_matrix()
    return_dict[index] = X
        
        
def merge_dataset(path_to_read, file_name, num_to_merge=15, verbose = False):
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    
    j = 0
    
    if verbose:
        print("Starting merging")
        start = time.time()
    
    while j<num_to_merge:
        
        j+=1
        
        process = multiprocessing.Process(target=merge_child, args=[return_dict, j, path_to_read])
        jobs.append(process)
        process.start()

    for p in jobs:
        p.join()
    
    
    for i, p in enumerate(jobs):
        
        if i == 0:
            X = return_dict[i+1]
        else:
            X = np.concatenate((X, return_dict[i+1]), axis=0)
        
    if verbose:
        print("Complete. Time spent: %s" % (time.time()-start))
        start = time.time()
        print("Saving the dataset")
        
    df = pd.DataFrame(X)
    df.to_csv(file_name, index = False)
    
    if verbose:
        print("Complete. Time spent: %s" % (time.time()-start))
        
        
        
################ DRAWING METHODS ################

def draw_centroids(centroids, colored_frame, color=[0,0,255]):
    for c in centroids:
        try: colored_frame[int(c[0])-2:int(c[0])+2, int(c[1])-2:int(c[1])+2] = color
        except: continue
    return colored_frame


def draw_clusters(labels, relevant_pixels, colored_frame):
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], 
          [0,255,255], [127,127,127], [0,0,0], [255,255,127], [255,127,255], [255,255,255]]
    
    for i, l in enumerate(labels):
        try: colored_frame[relevant_pixels[i][0],relevant_pixels[i][1]] = colors[l]
        except: continue
            
    return colored_frame



################ M-L ALGORITHMS ################

def rlsCV_regression(alphas, X_tr, X_ts, y_tr, y_ts):
    
    reg_RLS = sl.RidgeCV(alphas=alphas).fit(X_tr, y_tr)
    
    y_pred = reg_RLS.predict(X_ts)
    
    mean_err = np.mean(np.abs(y_pred - y_ts))
    var_err = np.var(np.abs(y_pred - y_ts))

    best_alpha = reg_RLS.alpha_
    coef = reg_RLS.coef_
    
    return best_alpha, mean_err, var_err, coef, y_pred


def svmCV_regression(params, X_tr, X_ts, y_tr, y_ts):

    svr = ss.SVR()
    clf = sm.GridSearchCV(svr, params)
    
    clf.fit(X_tr, y_tr)
    svr = clf.best_estimator_ 
    
    y_pred = svr.predict(X_ts)
        
    mean_err = np.mean(np.abs(y_pred - y_ts))
    var_err = np.var(np.abs(y_pred - y_ts))

    return clf.best_params_['C'], mean_err, var_err, svr.dual_coef_, y_pred


def ridgeKernelCV_regression(params, X_tr, X_ts, y_tr, y_ts, kernel='linear'):
    # kernel = ‘linear’ | ‘poly’ | ‘rbf’ | ‘sigmoid’ | ‘precomputed’ 

    kr = sk.KernelRidge(kernel=kernel)
    clf = sm.GridSearchCV(kr, params)
    
    clf.fit(X_tr, y_tr)
    kr = clf.best_estimator_ 
    
    y_pred = kr.predict(X_ts)

    mean_err = np.mean(np.abs(y_pred - y_ts))
    var_err = np.var(np.abs(y_pred - y_ts))

    return clf.best_params_['alpha'], mean_err, var_err, kr.dual_coef_, y_pred


def tree_regression(X_tr, X_ts, y_tr, y_ts):
    
    dtr = st.DecisionTreeRegressor()
    dtr.fit(X_tr, y_tr)
    y_pred = dtr.predict(X_ts)
    
    mean_err = np.mean(np.abs(y_pred - y_ts))
    var_err = np.var(np.abs(y_pred - y_ts))
    
    return mean_err, var_err, dtr.feature_importances_, y_pred


def lassoCV_regression(alphas, X_tr, X_ts, y_tr, y_ts):
    
    lasso = sl.LassoCV(alphas=alphas).fit(X_tr, y_tr)

    y_pred = lasso.predict(X_ts)
    
    mean_err = np.mean(np.abs(y_pred - y_ts))
    var_err = np.var(np.abs(y_pred - y_ts))
    
    return [lasso.alpha_, mean_err, var_err, lasso.coef_, y_pred]


#-------------GROUP LASSO-------------------------

MAX_ITER = 1000

def group_lassoCV(X, y, lambdas, groups, max_iter=MAX_ITER, rtol=1e-6):
    
    err = []
    coef_list = []
    
    X_tr, y_tr, _, X_v, y_v, _ = random_sampling(X, y, y)
    
    for l in lambdas:

        coef, _ = group_lasso(X_tr, y_tr, l, groups, max_iter=max_iter, rtol=rtol)
        coef_list.append(coef)
        err.append(sum(abs(y_v-np.dot(X_v, coef)))/len(y_v))
        

    best = np.argmin(err)
    
    return coef_list[best], lambdas[best]

"""Solve group lasso problem via ADMM.
More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""


def group_lasso(A, b, lamda=1.0, groups=None, rho=1.0, alpha=1.0, max_iter=1000, tol=1e-4, rtol=1e-2, return_history=False):
    r"""Group Lasso solver.
    Solves the following problem via ADMM
       minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.
    Parameters
    ----------
    A : array-like, 2-dimensional
        Input matrix.
    b : array-like, 1-dimensional
        Output vector.
    lamda : float, optional
        Regularisation parameter.
    groups : list
        Groups of variables.
    rho : float, optional
        Augmented Lagrangian parameter.
    alpha : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.
    Returns
    -------
    x : numpy.array
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
    """
    n_samples, n_features = A.shape

    # check valid partition
    if not np.allclose(flatten(groups), np.arange(n_features)):
        raise ValueError("Invalid partition in groups. "
                         "Groups must be non-overlapping and each variables "
                         "must be selected")

    # % save a matrix-vector multiply
    Atb = A.T.dot(b)

    # ADMM solver
    x = np.zeros(n_features)
    z = np.zeros(n_features)
    u = np.zeros(n_features)

    # % pre-factor
    L, U = lu_factor(A, rho)

    hist = []
    for _ in range(max_iter):
        # % x-update
        q = Atb + rho * (z - u)  # % temporary value
        if n_samples >= n_features:
            x = np.linalg.lstsq(U, np.linalg.lstsq(L, q)[0])[0]
        else:
            x = q - A.T.dot(
                np.linalg.lstsq(
                    U, np.linalg.lstsq(
                        L, A.dot(q))[0])[0]) / rho
            x /= rho

        # % z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        for group in groups:
            z[group] = soft_thresholding(x_hat[group] + u[group], lamda / rho)

        # % u-update
        u += (x_hat - z)

        # % diagnostics, reporting, termination checks
        history = (
            objective(A, b, lamda, groups, x, z),  # obj

            np.linalg.norm(x - z),  # r norm
            np.linalg.norm(-rho * (z - zold)),  # s norm

            np.sqrt(n_features) * tol + rtol * max(
                np.linalg.norm(x), np.linalg.norm(-z)),  # eps pri
            np.sqrt(n_features) * tol + rtol * np.linalg.norm(rho * u)  # eps dual
        )

        hist.append(history)
        if history[1] < history[3] and history[2] < history[4]:
            break

    return z, history if return_history else z


def objective(A, b, alpha, groups, x, z):
    # obj = 0
    # for i, group in enumerate(p):
    #     obj = obj + np.linalg.norm(z[group])
    penalty = np.sum([np.linalg.norm(z[g]) for g in groups])
    return .5 * np.sum((A.dot(x) - b) ** 2) + alpha * penalty

#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# License: BSD
import math
import numpy as np
from scipy import linalg


MAX_ITER_SPARSE = 1000

def soft_threshold(a, b):
    # vectorized version
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

def sparse_group_lassoCV(X, y, alphas, rhos, groups, max_iter=MAX_ITER_SPARSE, rtol=1e-6, verbose=False):

    err = []
    coef_list = []
    
    X_tr, y_tr, _, X_v, y_v, _ = random_sampling(X, y, y)
    
    for alpha in alphas:
        for rho in rhos:
        
            coef = sparse_group_lasso(X_tr, y_tr, alpha, rho, groups, max_iter=max_iter, rtol=rtol, verbose = verbose)
            coef_list.append(coef)
            err.append(sum(abs(y_v-X_v*coef)))

    best = np.argmin(err)
    alpha_best = best // len(alphas)
    rho_best = best // len(rhos)
    
    return coef_list[best], alphas[alpha_best], rhos[rho_best]

def sparse_group_lasso(X, y, alpha, rho, groups, max_iter=MAX_ITER_SPARSE, rtol=1e-6,
                verbose=False):
    """
    Linear least-squares with l2/l1 + l1 regularization solver.
    Solves problem of the form:
    (1 / (2 n_samples)) * ||Xb - y||^2_2 +
        [ (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1) ]
    where b_j is the coefficients of b in the
    j-th group. Also known as the `sparse group lasso`.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.
    y : array of shape (n_samples,)
    alpha : float or array
        Amount of penalization to use.
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution. TODO duality gap
    Returns
    -------
    x : array
        vector of coefficients
    References
    ----------
    "A sparse-group lasso", Noah Simon et al.
    """
    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))
    if groups.shape[0] != X.shape[1]:
        raise ValueError('Groups should be of shape %s got %s instead' % ((X.shape[1],), groups.shape))
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    n_samples = X.shape[0]
    alpha = alpha * n_samples

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    Xy = np.dot(X.T, y)
    K = np.dot(X.T, X)
    step_size = 1. / (linalg.norm(X, 2) ** 2)
    _K = [K[group][:, group] for group in group_labels]

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        perm = np.random.permutation(len(group_labels))
        X_residual = Xy - np.dot(K, w_new) # could be updated, but kernprof says it's peanuts
        for i in perm:
            group = group_labels[i]
            #import ipdb; ipdb.set_trace()
            p_j = math.sqrt(group.size)
            Kgg = _K[i]
            X_r_k = X_residual[group] + np.dot(Kgg, w_new[group])
            s = soft_threshold(X_r_k, alpha * rho)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha * p_j:
                w_new[group] = 0.
            else:
                # .. step 3 ..
                for _ in range(2 * group.size): # just a heuristic
                    grad_l =  - (X_r_k - np.dot(Kgg, w_new[group]))
                    tmp = soft_threshold(w_new[group] - step_size * grad_l, step_size * rho * alpha)
                    tmp *= max(1 - step_size * p_j * (1 - rho) * alpha / np.linalg.norm(tmp), 0)
                    delta = linalg.norm(tmp - w_new[group])
                    w_new[group] = tmp
                    if delta < 1e-3:
                        break

                assert np.isfinite(w_new[group]).all()

        norm_w_new = max(np.linalg.norm(w_new), 1e-10)
        if np.linalg.norm(w_new - w_old) / norm_w_new < rtol:
            #import ipdb; ipdb.set_trace()
            break
    return w_new


def check_kkt(A, b, x, penalty, groups):
    """Check KKT conditions for the group lasso
    Returns True if conditions are satisfied, False otherwise
    """
    group_labels = [groups == i for i in np.unique(groups)]
    penalty = penalty * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = 1e-1 # sort of tolerance
    for g in group_labels:
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < penalty + safety_net:
                return False
        else:
            w = - penalty * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
    return True


if __name__ == '__main__':
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    alpha = .1
    groups = np.r_[[0, 0], np.arange(X.shape[1] - 2)]
    coefs = group_lasso(X, y, alpha, groups, verbose=True)
    print('KKT conditions verified:', check_kkt(X, y, coefs, alpha, groups))

#---------------------------- END GROUP LASSO -----------------------

def split_tr_ts(X, Y_l, Y_f):
    
    splitter = sm.KFold(n_splits=2)
    tr, ts = splitter.split(X)
    tr = tr[0]
    ts = ts[0]

    return X[tr], X[ts], Y_l[tr], Y_f[tr], Y_l[ts], Y_f[ts]


def random_sampling(X, y_l, y_f):
    from sklearn.model_selection import ShuffleSplit
    
    train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.25).split(X, y_l))
    
    return X[train_idx], y_l[train_idx], y_f[train_idx], X[test_idx], y_l[test_idx], y_f[test_idx]


def print_results(model, modelResult_array):
    print('Model: {}'.format(model))
    
    print("\tregularizer alpha: {}".format(modelResult_array[0]))

    print("\tmean error: {}".format(modelResult_array[1]))

    print("\tvariance of the error: {}".format(modelResult_array[2]))

    print("\tcoefficient w:")
    print(modelResult_array[3])
    
    print("\tprediction:")
    print(modelResult_array[4])
    
    