import numpy as np
import skimage
from ctypes import CDLL

cfuncs = CDLL('eig3volume.dll')

def vesselness3d(image, sigmas, spacing, tau, brightondark):

    image = np.half(image)

    for n, sigma in np.ndenumerate(np.array(sigmas)):
        print('Current Filter Sigma: ', sigma)
        lambda1, lambda2, lambda3 = volumeEigenvalues(image, np.array(sigma), spacing, brightondark)
        if brightondark:
            lambda2 = -lambda2
            lambda3 = -lambda3
        lambda_rho = lambda3
        lambda_rho[lambda3 > 0 and lambda3 < tau * np.max(lambda3)] = tau * np.max(lambda3)
        lambda_rho[lambda3 <= 0] = 0
        response = lambda2 **2 * (lambda_rho - lambda2) * 27 / (lambda2 +lambda_rho)**3

        response[lambda2 >= lambda_rho/2 and lambda_rho > 0] = 1
        response[lambda2 <= 0 or lambda_rho <= 0] = 0

        if j==0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)
    vesselness = vesselness / np.max(vesselness)
    vesselness[vesselness < 1e3] = 0
    return vesselness

def volumeEigenvalues(volume, sigma, spacing, brightondark):

    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = Hessian3D(volume, sigma, spacing)

    c = sigma**2
    Hxx = c*Hxx
    Hxy = c*Hxy
    Hxz = c*Hxz
    Hyy = c*Hyy
    Hyz = c*Hyz
    Hzz = c*Hzz

    B1 = -(Hxx + Hyy + Hzz)
    B2 = Hxx * Hyy + Hxx * Hzz + Hyy * Hzz - Hxy * Hxy - Hxz * Hxz -Hyz * Hyz
    B3 = Hxx * Hyz * Hyz + Hxy * Hxy * Hzz + Hxz * Hyy * Hxz - Hxx * Hyy * Hzz - Hxy * Hyz * Hxz - Hxz * Hxy * Hyz

    T = np.ones_like(B1)

    if brightondark:
        T[B1 <= 0] = 0
        T[B2 <= 0 and B3 == 0] = 0
        T[B1 > 0 and B2 > 0 and B1 * B2 < B3] = 0
    else:
        T[B1>=0] = 0
        T[B2 >= 0 and B3 == 0] = 0
        T[B1 < 0 and B2 < 0 and (-B1) * (-B2) < (-B3)] = 0

        indices = np.nonzero(T==1)

        Hxx = Hxx[(indices)]
        Hyy = Hyy[(indices)]
        Hzz = Hzz[(indices)]
        Hxz = Hxz[(indices)]
        Hyz = Hyz[(indices)]
        Hxy = Hxy[(indices)]

        lambda1i, lambda2i, lambda3i = cfuncs.eig3volume(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

        lambda1 = np.zeros_like(T)
        lambda2 = np.zeros_like(T)
        lambda3 = np.zeros_like(T)

        for i in np.arange(indices.shape[0]):
            lambda1[indices[i]] = lambda1i
            lambda2[indices[i]] = lambda2i
            lambda3[indices[i]] = lambda3i

        lambda1[np.abs(lambda1) < 1e-4] = 0
        lambda2[np.abs(lambda2) < 1e-4] = 0
        lambda3[np.abs(lambda3) < 1e-4] = 0

        return lambda1, lambda2, lambda3

def Hessian3D(volume, sigma, spacing):

    if sigma > 0:
        F = skimage.filters.gaussian(volume, sigma)
    else:
        F = volume

    [Dx, Dy, Dz] = np.gradient(F)
    [Dxx, Dxy, Dxz] = np.gradient(Dx)
    [Dyx, Dyy, Dyz] = np.gradient(Dy)
    [Dzx, Dzy, Dzz] = np.gradient(Dz)

    return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz

