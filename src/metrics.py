"""3D evaluation metrics on native-resolution masks (1 mm voxels)."""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

# BraTS convention when exactly one of prediction / ground truth is empty:
# the HD95 is set to the image diagonal (sqrt(240^2 + 240^2 + 155^2) mm).
HD95_EMPTY = 373.13


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(pred, gt).sum() / s


def _surface(m):
    return m & ~ndimage.binary_erosion(m)


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile symmetric surface distance in mm (max of both directions)."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return HD95_EMPTY
    # work on the bounding box of both masks (plus margin) for speed
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = idx.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = _surface(pred[sl]), _surface(gt[sl])
    d_to_g = ndimage.distance_transform_edt(~g, sampling=spacing)[p]
    d_to_p = ndimage.distance_transform_edt(~p, sampling=spacing)[g]
    return float(max(np.percentile(d_to_g, 95), np.percentile(d_to_p, 95)))


def volume_ml(mask, voxel_ml=0.001):
    return float(mask.sum() * voxel_ml)


def uncertainty_vs_error(unc, pred, gt, brain, rng=None, n=200_000):
    """Do uncertain voxels coincide with wrong voxels?

    Returns AUROC of the uncertainty score for detecting errors inside the brain
    (0.5 = no relation, 1 = perfect), and the error rate among the 1% most
    uncertain voxels vs. all other brain voxels.
    """
    rng = rng or np.random.default_rng(0)
    u, err = unc[brain].astype(np.float32), (pred != gt)[brain]
    if err.all() or not err.any():
        return {"unc_auroc": np.nan, "err_top1pct": np.nan, "err_rest": float(err.mean())}
    if u.size > n:
        keep = rng.choice(u.size, n, replace=False)
        u_s, e_s = u[keep], err[keep]
    else:
        u_s, e_s = u, err
    auroc = roc_auc_score(e_s, u_s) if 0 < e_s.sum() < e_s.size else np.nan
    cut = np.quantile(u, 0.99)
    top = u > cut if cut <= u.min() else u >= cut   # many voxels can tie at 0
    return {"unc_auroc": float(auroc), "err_top1pct": float(err[top].mean()),
            "err_rest": float(err[~top].mean())}
