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

mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

rmse = math.sqrt(mean_squared_error(b_pred, bU[1:]))
print('Train Score: %.8f RMSE' % (rmse))

#%% Pearson correlation coefficient
r1 = np.sum((bU[1:] - np.mean(bU[1:]))*(b_pred - np.mean(b_pred)))
r2 = np.sqrt(np.sum(np.square(bU[1:] - np.mean(bU[1:]))))*np.sqrt(np.sum(np.square(b_pred - np.mean(b_pred))))
b_train = r1/r2
print('pearson_train= ', b_train)
