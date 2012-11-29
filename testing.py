#!/usr/bin/env python
"""
A test script for GFFT.
"""

import numpy as np
import pylab as pl
import gfft

a = np.zeros(128, dtype=float)
a[32:34] = np.ones(2, dtype=float)

rr = gfft.RRTransform(1)
irr = rr.get_inverse_transform()

rr2 = gfft.RRTransform(1, in_ref=10., out_axes=(0.1, 128))
irr2 = rr2.get_inverse_transform()

A = rr(a)
A2 = rr2(a)

pl.figure()
pl.plot(A.real)
pl.plot(A2.real)
pl.show()

aa = irr(A).real
aa2 = irr2(A2).real

pl.figure()
pl.plot(a)
pl.plot(aa)
pl.plot(aa2)
pl.show()
