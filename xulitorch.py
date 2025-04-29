#%% list to torch tensor
val_pos = torch.stack(val_pos, dim=0)
val_y = torch.stack(val_y, dim=0)
