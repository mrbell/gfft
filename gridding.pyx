#cython: boundscheck=False
#cython: wraparound=False
"""
gridding.pyx

This file contains the gridding functions used in the GFFT package. We use
a gridding procedure as described in the paper

Beatty, P.J. and Nishimura, D.G. and Pauly, J.M. "Rapid gridding reconstruction
with a minimal oversampling ratio", IEEE Transactions on Medical Imaging,
Vol. 24, Num. 6, 2005
"""

"""
Copyright 2012 Michael Bell, Henrik Junklewitz

This file is part of GFFT.

GFFT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GFFT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GFFT.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

DTYPE = np.float64
CTYPE = np.complex128
ITYPE = int
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CTYPE_t

cdef extern from "gsl/gsl_sf_bessel.h":
    double gsl_sf_bessel_I0(double x)

cdef extern from "math.h":
    double exp(double theta)
    double sqrt(double x)
    double ceil(double x)
    double sin(double theta)


###############################################################################
# Wrapper functions for new GFFT classes
###############################################################################
def gridder3d(list u, np.ndarray[CTYPE_t, ndim=1] vis,
              list du, list Nu, list umin, double alpha, int W,
              list hflag):
    """Just a wrapper for the old grid_3d function with a unified interface"""

    return grid_3d(u[0], u[1], u[2], vis,
                   du[0], Nu[0], umin[0],
                   du[1], Nu[1], umin[1],
                   du[2], Nu[2], umin[2],
                   alpha, W, hflag[0], hflag[1], hflag[2])


def degridder3d(list u, np.ndarray[CTYPE_t, ndim=3] regVis, \
                list du, list Nu, list umin, double alpha, int W):
    """
    Just a wrapper around the old degrid_3d function with a unified interface
    """

    return degrid_3d(u[0], u[1], u[2], regVis,
                     du[0], Nu[0], umin[0],
                     du[1], Nu[1], umin[1],
                     du[2], Nu[2], umin[2],
                     alpha, W)


def gridcorr3d(list dx, list Nx, list xmin, list du, int W, double alpha):
    """
    A thin wrapper around the old get_grid_corr_3d function with a unified
    interface.
    """
    return get_grid_corr_3d(dx[0], Nx[0], xmin[0],
                            dx[1], Nx[1], xmin[1],
                            dx[2], Nx[2], xmin[2],
                            du[0], du[1], du[2], W, alpha)


def gridder2d(list u, np.ndarray[CTYPE_t, ndim=1] vis,
              list du, list Nu, list umin, double alpha, int W,
              list hflag):
    """Just a wrapper for the old grid_2d function with a unified interface"""

    return grid_2d(u[0], u[1], vis,
                   du[0], Nu[0], umin[0],
                   du[1], Nu[1], umin[1],
                   alpha, W, hflag[0], hflag[1])


def degridder2d(list u, np.ndarray[CTYPE_t, ndim=2] regVis, \
                list du, list Nu, list umin, double alpha, int W):
    """
    Just a wrapper around the old degrid_2d function with a unified interface
    """

    return degrid_2d(u[0], u[1], regVis,
                     du[0], Nu[0], umin[0],
                     du[1], Nu[1], umin[1],
                     alpha, W)

def gridcorr2d(list dx, list Nx, list xmin, list du, int W, double alpha):
    """
    A thin wrapper around the old get_grid_corr_2d function with a unified
    interface.
    """
    return get_grid_corr_2d(dx[0], Nx[0], xmin[0],
                            dx[1], Nx[1], xmin[1],
                            du[0], du[1], W, alpha)


def gridder1d(list u, np.ndarray[CTYPE_t, ndim=1] vis,
              list du, list Nu, list umin, double alpha, int W,
              list hflag):
    """Just a wrapper for the old grid_1d function with a unified interface"""

    return grid_2d(u[0], vis,
                   du[0], Nu[0], umin[0],
                   alpha, W, hflag[0])


def degridder1d(list u, np.ndarray[CTYPE_t, ndim=1] regVis, \
                list du, list Nu, list umin, double alpha, int W):
    """
    Just a wrapper around the old degrid_1d function with a unified interface
    """

    return degrid_1d(u[0], regVis,
                     du[0], Nu[0], umin[0],
                     alpha, W)


def gridcorr1d(list dx, list Nx, list xmin, list du, int W, double alpha):
    """
    A thin wrapper around the old get_grid_corr_1d function with a unified
    interface.
    """
    return get_grid_corr_1d(dx[0], Nx[0], xmin[0],
                            du[0], W, alpha)


###############################################################################
# 3D functions
###############################################################################
def grid_3d(np.ndarray[DTYPE_t, ndim=1] u, np.ndarray[DTYPE_t, ndim=1] v,
            np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[CTYPE_t, ndim=1] vis,
            double du, int Nu, double umin, double dv, int Nv, double vmin,
            double dw, int Nw, double wmin, double alpha, int W,
            bool hflag_u, bool hflag_v, bool hflag_w):
    """Gridding for 3D data arrays."""

    cdef int W3 = W ** 3
    cdef int nvis = u.shape[0]

    cdef np.ndarray[CTYPE_t, ndim=3, mode='c'] gv = \
        np.zeros((Nu, Nv, Nw), dtype=CTYPE)  #output array

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
        np.zeros(nvis * W3, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vg = \
        np.zeros(nvis * W3, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] wg = \
        np.zeros(nvis * W3, dtype=DTYPE)
    cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] visg = \
        np.zeros(nvis * W3, dtype=CTYPE)

    # holds the W values after u gridding
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu1 = \
        np.zeros(W, dtype=DTYPE)
    cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tvis1 = \
        np.zeros(W, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv1 = \
        np.zeros(W, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tw1 = \
        np.zeros(W, dtype=DTYPE)


    # holds the W**2 values after subsequent v gridding
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu2 =\
        np.zeros(W ** 2, dtype=DTYPE)
    cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tvis2 = \
        np.zeros(W ** 2, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv2 = \
        np.zeros(W ** 2, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tw2 = \
        np.zeros(W ** 2, dtype=DTYPE)


    # holds the W**3 values after subsequent w gridding
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu3 = \
        np.zeros(W3, dtype=DTYPE)
    cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tvis3 = \
        np.zeros(W3, dtype=CTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv3 = \
        np.zeros(W3, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tw3 = \
        np.zeros(W3, dtype=DTYPE)


    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] su = \
        np.zeros(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] sv = \
        np.zeros(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] sw = \
        np.zeros(1, dtype=DTYPE)
    cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] svis = \
        np.zeros(1, dtype=CTYPE)

    cdef Py_ssize_t i, undx, vndx, wndx

    cdef double beta = get_beta(W, alpha)

    for i in range(nvis):

        # For each visibility point, grid in 3D, one dimension at a time
        # so each visibility becomes W**3 values located on the grid

        # Grid in u
        su[0] = u[i]
        sv[0] = v[i]
        sw[0] = w[i]
        svis[0] = vis[i]
        grid_1d_from_3d(su, svis, du, W, beta, sv, sw, tu1, tvis1,
                        tv1, tw1)

        # Grid in v
        grid_1d_from_3d(tv1, tvis1, dv, W, beta, tu1, tw1,
                        tv2, tvis2, tu2, tw2) # output arrays

        # Grid in l2
        grid_1d_from_3d(tw2, tvis2, dw, W, beta, tu2, tv2,
                        tw3, tvis3, tu3, tv3) # output arrays

        ug[i * W3:(i + 1) * W3] = tu3
        vg[i * W3:(i + 1) * W3] = tv3
        wg[i * W3:(i + 1) * W3] = tw3
        visg[i * W3:(i + 1) * W3] = tvis3


    cdef int N = ug.shape[0]
    cdef double temp = 0

    for i in range(N):
        # compute the location for the visibility in the visibility cube
        temp = (ug[i] - umin)/du + 0.5
        undx = int(temp)
        temp = (vg[i] - vmin)/dv + 0.5
        vndx = int(temp)
        temp = (wg[i] - wmin)/dw + 0.5
        wndx = int(temp)

        if (undx >= 0 and undx < Nu) and (vndx >= 0 and vndx < Nv)\
            and (wndx >= 0 and wndx < Nw):
                gv[undx, vndx, wndx] = gv[undx, vndx, wndx] + visg[i]


        # now compute the location for the -u,-v,-l2 visibility, which is
        # equal to the complex conj of the u,v,l2 visibility if we
        # assume that each Stokes image in Faraday space is real

        if hflag_u or hflag_v or hflag_w:
            if hflag_u:
                temp = (-1. * ug[i] - umin)/du + 0.5
                undx = int(temp)
            if hflag_v:
                temp = (-1. * vg[i] - vmin)/dv + 0.5
                vndx = int(temp)
            if hflag_w:
                temp = (-1. * wg[i] - wmin)/dw + 0.5
                wndx = int(temp)


            if (undx >= 0 and undx < Nu) and (vndx >= 0 and vndx < Nv)\
                and (wndx >= 0 and wndx < Nw):
                    gv[undx, vndx, wndx] = gv[undx, vndx, wndx] +\
                        visg[i].conjugate()

    return gv


def degrid_3d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v, \
    np.ndarray[DTYPE_t,ndim=1] w, np.ndarray[CTYPE_t, ndim=3] regVis, \
    double du, double Nu, double umin, double dv, double Nv, double vmin, \
    double dw, double Nw, double wmin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
            np.arange(0.,Nv,1.)*dv + vmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] wgrid = \
            np.arange(0.,Nw,1.)*dw + wmin

        cdef int nvis = u.shape[0]

        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
            np.zeros(nvis, dtype=CTYPE)

        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du
        cdef double Dv = W*dv
        cdef double Dw = W*dw

        cdef Py_ssize_t i, j, vrang, urang, wrang, k, Wu, Wv, l

        cdef double gcf_val


        for k in range(nvis):

            urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))
            vrang = int(np.ceil((v[k] - 0.5*Dv - vmin)/dv))
            wrang = int(np.ceil((w[k] - 0.5*Dw - wmin)/dw))

            for i in range(urang, urang+W):
                for j in range(vrang, vrang+W):
                    for l in range(wrang, wrang+W):
                        if (i<Nu and i>=0) and (j<Nv and j>=0) and \
                            (l<Nw and l>=0):
                                gcf_val = gcf_kaiser(u[k]-ugrid[i], Du, beta)*\
                                    gcf_kaiser(v[k]-vgrid[j], Dv, beta) \
                                    *gcf_kaiser(w[k]-wgrid[l], Dw, beta)

                                Vis[k] = Vis[k] + regVis[i,j,l]*gcf_val

        return Vis



# Cython version is 180x faster than pure python
def get_grid_corr_3d(double dx, int Nx, double xmin, \
    double dy, int Ny, double ymin, double dz, int Nz, double zmin, \
    double du, double dv, double dw, int W, double alpha):

#    cdef int Nz = z.shape[0]
#    cdef int Ny = y.shape[0]
#    cdef int Nx = x.shape[0]

        cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] gridcorr = \
            np.zeros([Nx, Ny, Nz], dtype=DTYPE)

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] x = \
            np.arange(Nx, dtype=DTYPE)*dx + xmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] y = \
            np.arange(Ny,dtype=DTYPE)*dy + ymin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] z = \
            np.arange(Nz,dtype=DTYPE)*dz + zmin

        # see Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)

        cdef Py_ssize_t i, j, k

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    gridcorr[i,j,k] = inv_gcf_kaiser(x[i], du, W, beta)*\
                        inv_gcf_kaiser(y[j], dv, W, beta)*\
                        inv_gcf_kaiser(z[k], dw, W, beta)

        return gridcorr


cdef inline void grid_1d_from_3d(np.ndarray[DTYPE_t,ndim=1] x, \
    np.ndarray[CTYPE_t,ndim=1] vis, double dx, int W, double beta, \
    np.ndarray[DTYPE_t,ndim=1] y, np.ndarray[DTYPE_t,ndim=1] z, \
    np.ndarray[DTYPE_t,ndim=1] x2, np.ndarray[CTYPE_t,ndim=1] vis2,\
    np.ndarray[DTYPE_t,ndim=1] y2, np.ndarray[DTYPE_t,ndim=1] z2):


        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = x.shape[0]

        cdef double Dx = W*dx

        cdef Py_ssize_t indx, xndx, kndx

        cdef double xval, yval, zval, xref, xg, gcf_val

        cdef CTYPE_t visval

        for indx in range(N):

            visval = vis[indx]

            xval = x[indx]
            yval = y[indx]
            zval = z[indx]

            xref = ceil((xval - 0.5*W*dx)/dx)*dx

            for xndx in range(W):

                xg = xref + xndx*dx

                kndx = indx*W + xndx

                gcf_val = gcf_kaiser(xg-xval, Dx, beta)

                vis2[kndx] = visval*gcf_val

                x2[kndx] = xg
                y2[kndx] = yval
                z2[kndx] = zval



################################################################################
# 2D functions
################################################################################

def grid_2d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v,\
    np.ndarray[CTYPE_t, ndim=1] vis, double du, int Nu, double umin, \
    double dv, int Nv, double vmin, double alpha, int W, bool hflag_u, \
    bool hflag_v):

        cdef int W2 = W**2
        cdef int nvis = u.shape[0]

        cdef np.ndarray[CTYPE_t, ndim=2, mode='c'] gv = \
            np.zeros((Nu, Nv), dtype=CTYPE) #output array

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
            np.zeros(nvis*W2, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vg = \
            np.zeros(nvis*W2, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] visg = \
            np.zeros(nvis*W2, dtype=CTYPE)

        # holds the W values after u gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu1 = \
            np.zeros(W, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tvis1 = \
            np.zeros(W, dtype=CTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv1 = \
            np.zeros(W, dtype=DTYPE)


        # holds the W**2 values after subsequent v gridding
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tu2 =\
            np.zeros(W2, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] tvis2 = \
            np.zeros(W2, dtype=CTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] tv2 = \
            np.zeros(W2, dtype=DTYPE)


        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] su = \
            np.zeros(1, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] sv = \
            np.zeros(1, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] svis = \
            np.zeros(1, dtype=CTYPE)

        cdef Py_ssize_t i, undx, vndx

        cdef double beta = get_beta(W, alpha)

        for i in range(nvis):

            # For each visibility point, grid in 3D, one dimension at a time
            # so each visibility becomes W**3 values located on the grid

            # Grid in u
            su[0] = u[i]
            sv[0] = v[i]
            svis[0] = vis[i]
            grid_1d_from_2d(su, svis, du, W, beta, sv, tu1, tvis1, tv1)

            # Grid in v
            grid_1d_from_2d(tv1, tvis1, dv, W, beta, tu1, \
                tv2, tvis2, tu2) # output arrays

            ug[i*W2:(i+1)*W2] = tu2
            vg[i*W2:(i+1)*W2] = tv2
            visg[i*W2:(i+1)*W2] = tvis2


        cdef int N = ug.shape[0]
        cdef double temp = 0

        for i in range(N):
            # compute the location for the visibility in the visibility cube
            temp = (ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (vg[i] - vmin)/dv + 0.5
            vndx = int(temp)

            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                gv[undx, vndx] = gv[undx, vndx] + visg[i]


            # now compute the location for the -u,-v,-l2 visibility, which is
            # equal to the complex conj of the u,v,l2 visibility if we
            # assume that the individual Stokes images in Faraday space are real

            if hflag_u or hflag_v:
                if hflag_u:
                    temp = (-1.*ug[i] - umin)/du + 0.5
                    undx = int(temp)
                if hflag_v:
                    temp = (-1.*vg[i] - vmin)/dv + 0.5
                    vndx = int(temp)

                if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                    gv[undx, vndx] = gv[undx, vndx] + visg[i].conjugate()

        return gv


def degrid_2d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v, \
    np.ndarray[CTYPE_t, ndim=2] regVis, double du, int Nu, double umin, \
    double dv, int Nv, double vmin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
            np.arange(0.,Nv,1.)*dv + vmin

        cdef int nvis = u.shape[0]

        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
            np.zeros(nvis, dtype=CTYPE)

        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du
        cdef double Dv = W*dv

        cdef Py_ssize_t i, j, urang, vrang, k, Wu, Wv

        cdef double gcf_val, gcf_val_u, gcf_val_v


        for k in range(nvis):

            urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))
            vrang = int(np.ceil((v[k] - 0.5*Dv - vmin)/dv))

            for i in range(urang, urang+W):
                if (i>=Nu or i<0): continue
                gcf_val_u = gcf_kaiser(u[k]-ugrid[i], Du, beta)
                for j in range(vrang, vrang+W):
                    if (j>=Nv or j<0): continue

                    gcf_val_v = gcf_kaiser(v[k]-vgrid[j], Dv, beta)

                    # convolution kernel for position i,j
                    gcf_val = gcf_val_v*gcf_val_u
                    #sampling back to visibility point k
                    Vis[k] = Vis[k] + regVis[i,j]*gcf_val

        return Vis


def get_grid_corr_2d(double dx, int Nx, double xmin, \
    double dy, int Ny, double ymin, double du, double dv, int W, double alpha):

#    cdef int Nz = z.shape[0]
#    cdef int Ny = y.shape[0]
#    cdef int Nx = x.shape[0]

        cdef np.ndarray[DTYPE_t,ndim=2, mode='c'] gridcorr = np.zeros([Ny, Nx],\
            dtype=DTYPE)

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] x = \
            np.arange(Nx, dtype=DTYPE)*dx + xmin
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] y = \
            np.arange(Ny, dtype=DTYPE)*dy + ymin

        # see Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)

        cdef Py_ssize_t i, j

        for i in range(Nx):
            for j in range(Ny):
                gridcorr[i,j] = inv_gcf_kaiser(x[i], du, W, beta)*\
                    inv_gcf_kaiser(y[j], dv, W, beta)

        return gridcorr


cdef inline void grid_1d_from_2d(np.ndarray[DTYPE_t,ndim=1] x, \
    np.ndarray[CTYPE_t,ndim=1] vis, double dx, int W, double beta, \
    np.ndarray[DTYPE_t,ndim=1] y, \
    np.ndarray[DTYPE_t,ndim=1] x2, np.ndarray[CTYPE_t,ndim=1] vis2,\
    np.ndarray[DTYPE_t,ndim=1] y2):


        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = x.shape[0]

        cdef double Dx = W*dx

        cdef Py_ssize_t indx, xndx, kndx

        cdef double xval, yval, xref, xg, gcf_val

        cdef CTYPE_t visval

        for indx in range(N):

            visval = vis[indx]

            xval = x[indx]
            yval = y[indx]

            xref = ceil((xval - 0.5*W*dx)/dx)*dx

            for xndx in range(W):

                xg = xref + xndx*dx

                kndx = indx*W + xndx

                gcf_val = gcf_kaiser(xg-xval, Dx, beta)

                vis2[kndx] = visval*gcf_val

                x2[kndx] = xg
                y2[kndx] = yval



################################################################################
# 1D functions
################################################################################

def grid_1d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[CTYPE_t,ndim=1] vis, \
    double du, int Nu, double umin, int W, double alpha, bool hermitianize):
        """
        Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
        """
        cdef int N = u.shape[0]
        cdef double Du = W*du

        cdef Py_ssize_t indx, undx, kndx
        cdef double uval, uref, tu, gcf_val
        cdef CTYPE_t visval
        cdef double temp = 0.

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
            np.zeros(N*W, dtype=DTYPE)
        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] visg = \
            np.zeros(N*W, dtype=CTYPE)

        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] gv = \
            np.zeros(Nu, dtype=CTYPE) # output array

        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)

        # do convolution
        for indx in range(N):

            visval = vis[indx]
            uval = u[indx]

            uref = ceil((uval - 0.5*W*du - umin)/du)*du + umin

            for undx in range(W):

                tu = uref + undx*du
                kndx = indx*W + undx

                gcf_val = gcf_kaiser(tu-uval, Du, beta)

                visg[kndx] = visval*gcf_val
                ug[kndx] = tu

        # sample onto grid
        for indx in range(N*W):
            # compute the location for the visibility in the visibility cube
            temp = (ug[indx] - umin)/du + 0.5
            undx = int(temp)

            if (undx>=0 and undx<Nu):
                    gv[undx] = gv[undx] + visg[indx]

            if hermitianize:
                temp = (-1.*ug[indx] - umin)/du + 0.5
                undx = int(temp)

                if (undx>=0 and undx<Nu):
                    gv[undx] = gv[undx] + visg[indx].conjugate()

        return gv

def degrid_1d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[CTYPE_t, ndim=1] regVis,\
    double du, int Nu, double umin, double alpha, int W):

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
            np.arange(0.,Nu,1.)*du + umin

        cdef int nvis = u.shape[0]

        cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
            np.zeros(nvis, dtype=CTYPE)

        # From Beatty et al. (2005)
        cdef double beta = get_beta(W, alpha)
        # Grid in u and v
        cdef double Du = W*du

        cdef Py_ssize_t i, j, urang, k

        cdef double gcf_val

        for k in range(nvis):

            urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))

            for i in range(urang, urang+W):
                if (i<Nu and i>=0):
                    #convolution kernel for position i
                    gcf_val = gcf_kaiser(u[k]-ugrid[i], Du, beta)
                    #sampling back to visibility point k
                    Vis[k] = Vis[k] + regVis[i]*gcf_val

        return Vis


def get_grid_corr_1d(double dx, int Nx, double xmin, double du, int W, \
    double alpha):

        cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] gridcorr = np.zeros(Nx,\
            dtype=DTYPE)
        cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] x = np.arange(Nx,\
            dtype=DTYPE)*dx + xmin

        cdef double beta = get_beta(W, alpha)

        cdef Py_ssize_t i

        for i in range(Nx):
            gridcorr[i] = inv_gcf_kaiser(x[i], du, W, beta)

        return gridcorr


def test_gcf_kaiser(double k, double dk, int W, double alpha):

    cdef double beta = get_beta(W, alpha)
    return gcf_kaiser(k, dk*W, beta)

################################################################################
# Common functions
################################################################################

cdef inline double get_beta(int W, double alpha):
    cdef double pi = 3.141592653589793
    # see Beatty et al. (2005)
    cdef double beta = pi*sqrt((W*W/alpha/alpha)*(alpha - 0.5)*(alpha - 0.5) \
        - 0.8)

    return beta

cdef inline double gcf_kaiser(double k, double Dk, double beta):

    cdef double temp3 = 2.*k/Dk

    if (1 - temp3)*(1 + temp3) < -1e-12:
#        print "There is an issue with the gridding code!"
        raise Exception("There is an issue with the gridding code!")

    temp3 = sqrt(abs((1 - temp3)*(1 + temp3)))

    temp3 = beta*temp3

#    cdef double C = (1./Dk)*gsl_sf_bessel_I0(temp3)/gsl_sf_bessel_I0(beta)
    cdef double C = gsl_sf_bessel_I0(temp3)/gsl_sf_bessel_I0(beta)

    return C


cdef inline double inv_gcf_kaiser(double x, double dk, int W, double beta):

    cdef double pi = 3.141592653589793
    cdef double temp1 = pi*pi*W*W*dk*dk*x*x
    cdef double temp2 = beta*beta
    cdef double temp, c

    temp = sqrt(temp2 - temp1)
    c0 = (exp(beta)-exp(-1.*beta))/2./beta
    c = (exp(temp) - exp(-1.*temp))/2./temp

    if temp1>temp2:
        temp = sqrt(temp1 - temp2)
        c = -0.5*(exp(-1.*temp) - exp(temp))/temp
        #print "WARNING: There may be trouble..."


    return c/c0

