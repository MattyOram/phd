def smooth(mesh, n_iter):
    mesh1 = mesh.copy(deep=True)
    smooth_mesh = mesh1.smooth_taubin(n_iter=n_iter)
    return smooth_mesh 