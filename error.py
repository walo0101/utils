def iou_coef(y_true, y_pred, smooth=1.):
  intersection = backend.sum(backend.abs(y_true * y_pred))
  union = backend.sum(y_true) + backend.sum(y_pred)-intersection
  iou = backend.mean((intersection + smooth) / (union + smooth))
  return iou
    
def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    
#%% xuli torch dice not binary mask
def dice_coefficient_multiclass(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1e-6, ignore_index: int = None) -> torch.Tensor:
    # If preds are logits or probs → convert to class indices
    if preds.ndim == targets.ndim + 1:
        preds = torch.argmax(preds, dim=1)  # (N, H, W)
    
    dice_scores = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        #dice = ( intersection + smooth) / (union - intersection + smooth)
        dice_scores.append(dice)
    return torch.tensor(dice_scores).mean()  # shape: (num_classes,)

def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))

def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err) / np.linalg.norm(y_true)

def mean_l2_relative_error(y_true, y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return np.mean(
        np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    )

def _absolute_percentage_error(y_true, y_pred):
    return 100 * np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(config.real(np)).eps, None)
    )

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))

def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))

def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error
mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

rmse = math.sqrt(mean_squared_error(b_pred, bU[1:]))
print('Train Score: %.8f RMSE' % (rmse))

#%% Pearson correlation coefficient
r1 = np.sum((bU[1:] - np.mean(bU[1:]))*(b_pred - np.mean(b_pred)))
r2 = np.sqrt(np.sum(np.square(bU[1:] - np.mean(bU[1:]))))*np.sqrt(np.sum(np.square(b_pred - np.mean(b_pred))))
b_train = r1/r2
print('pearson_train= ', b_train)
