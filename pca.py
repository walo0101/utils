from sklearn.decomposition import PCA
pca = PCA(n_components=t)
pca.fit(ds_input_pos)
P = pca.components_
P = np.transpose(P)

# Px = PCA(n_components=t)
# bX = Px.fit_transform(ds_input_pos)

#**==** construct vector b **==**
def pca_star(P, U):
    U = np.transpose(U)
    A = np.dot(np.transpose(P),P)
    B = np.dot(np.transpose(P),U)
    b = np.dot(np.linalg.inv(A),B)
    return np.transpose(b)

def pca_starX(P, X, Xm):
    X = np.transpose(X)
    A = np.dot(np.transpose(P),P)
    x_xm = X - np.transpose(Xm)
    B = np.dot(np.transpose(P),x_xm)
    b = np.dot(np.linalg.inv(A),B)
    return np.transpose(b)

bX = np.zeros([row, t])
bV = np.zeros([row, t])
bF = np.zeros([row, t])

bU = np.zeros([row, t])
for i in range(0, row):
    X = ds_input_pos[i:i+1]
    # V = ds_input_vel[i:i+1]
    F = ds_input_ext[i:i+1]
    U = u_ds_output[i:i+1,:] 
    
    bX[i:i+1] = pca_starX(P,X, ds_input_pos[0:1,:])
    # bV[i:i+1] = pca_starX(P,V, ds_input_vel[0:1,:])
    bF[i:i+1] = pca_starX(Pf,F, ds_input_ext[0:1,:])
    bU[i:i+1] = pca_star(P, U)   
