from sklearn.decomposition import PCA
pca = PCA(n_components=t)
pca.fit(ds_input_pos)
P = pca.components_
P = np.transpose(P)
