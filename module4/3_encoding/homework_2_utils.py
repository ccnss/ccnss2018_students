import numpy as np
import npp

from itertools import combinations, chain

def make_data(N_R, N_P, P_parts, M, true_variances, noise_variance, combs, Pnoise_models,
              P_models, use_mixing=True, orthogonalize=True, noise_scale=1.0, **etc):
    # Generate timecourses for each partition
    X_parts = [np.random.randn(p, N_R + N_P) for p in P_parts]
    #print "X_parts[0].shape", X_parts[0].shape
    XX = np.corrcoef(np.vstack(X_parts))

    # Orthogonalize timecourses across and within partitions?
    if orthogonalize:
        cat_orthog_X_parts, _, _ = np.linalg.svd(np.vstack(X_parts).T, full_matrices=False)
        X_parts = np.vsplit(npp.zs(cat_orthog_X_parts).T, np.cumsum(P_parts)[:-1])
        XX_orthog = np.corrcoef(np.vstack(X_parts))

    # Generate "true" weights used to construct Y
    Btrue_parts = [np.random.randn(p, M) for p in P_parts]
    #print "Btrue_parts[0].shape", Btrue_parts[0].shape

    # Generate output timecourses for each partition
    Y_parts = [B.T.dot(X).T for X,B in zip(X_parts, Btrue_parts)]
    #print "Y_parts[0].shape", Y_parts[0].shape

    # Rescale timecourses for each partition to have appropriate variance
    scaled_Y_parts = [Y / Y.std(0) * np.sqrt(tv) for Y,tv in zip(Y_parts, true_variances)]
    #print "scaled_Y_parts[0].shape", scaled_Y_parts[0].shape

    # Generate noise timecourses scaled to have appropriate variance
    Y_noise = np.random.randn(N_R + N_P, M)
    scaled_Y_noise = Y_noise / Y_noise.std(0) * np.sqrt(noise_variance)
    #print "scaled_Y_noise.shape", scaled_Y_noise.shape

    # Construct Y from combination of partition timecourses
    Y_total = np.array(scaled_Y_parts).sum(0) + scaled_Y_noise
    zY_total = npp.zs(Y_total)
    #print "Y_total.shape", Y_total.shape

    # Generate feature timecourses
    # Stack together partition features to make "true" features for each feature space
    Xtrue_feats = [np.vstack([X_parts[c] for c in comb]) for comb in combs]
    #print "Xtrue_feats[0].shape", Xtrue_feats[0].shape

    # Generate noise features to round out each feature space
    Xnoise_feats = [noise_scale * np.random.randn(Pnoise, N_R + N_P) for Pnoise in Pnoise_models]
    #print "Xnoise_feats[0].shape", Xnoise_feats[0].shape

    # Generate matrices to mix real and noise features in each space
    mixing_mats = [np.random.randn(P, P) for P in P_models]
    #print "mixing_mats[0].shape", mixing_mats[0].shape

    # Use mixing matrices to generate feature timecourses
    if use_mixing:
        X_feats = [m.dot(np.vstack([Xt, Xn])) for m,Xt,Xn in zip(mixing_mats, Xtrue_feats, Xnoise_feats)]
    else:
        X_feats = [np.vstack([Xt, Xn]) for m,Xt,Xn in zip(mixing_mats, Xtrue_feats, Xnoise_feats)]
    #print "X_feats[0].shape", X_feats[0].shape

    return X_feats, Y_total
