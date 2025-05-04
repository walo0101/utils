#%% list to torch tensor
val_pos = torch.stack(val_pos, dim=0)
val_y = torch.stack(val_y, dim=0)

#%% save torch tensor variable
torch.save(val_pred, 'val_pred_batch8.pt')
torch.save(val_loader, 'val_loader_batch8.pt')

#%% load torch tensor variable
BATCH_SIZE = 8
val_pred=torch.load('val_pred_batch8.pt', map_location=torch.device('cpu'))
pred = []
for i, data in enumerate(val_pred, 0):
  data = torch.reshape(data, (BATCH_SIZE, 2048))
  for j in range(0,len(data)):
    pred.append(data[j])

val_gt = torch.load('val_loader_batch8.pt',weights_only=False)
val_pos = []
val_y = []
for i, data in enumerate(val_gt, 0):
  for j in range(0,len(data)):
    val_pos.append(data[j].pos)
    val_y.append(data[j].y)


iou_val = []
for i in range(0,len(pred)):
  iou = compute_iou(pred[i], val_y[i])
  iou_val.append(iou.item())
print(np.mean(iou_val))
print(np.std(iou_val))
print(np.max(iou_val))
print(np.min(iou_val))
print(np.argmin(iou_val))
