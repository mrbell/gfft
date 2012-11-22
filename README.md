GFFT - Generalized Fast Fourier Transformations for Python

** Overview ********************************************************************

GFFT is a package for performing Fast Fourier Transformations even in the
case where the data is not defined on a regularly spaced grid. This is commonly
the case in applications such as radio interferometry and medical imaging.

One can naively just "bin" their data before using a standard FFT algorithm, but
this leads to severe aliasing in the result. Instead, GFFT uses a method known 
as "gridding" to regularize the data. Gridding makes use of the fact that the 
space onto which we wish to transform our data is finite in extent. One can 
think of the finite space as being an infinitely extended space that is 
multiplied by a window function that has finite support.

Gridding involves convolving the irregularly spaced data with a gridding 
convolution kernel (GCK) that is the Fourier transform of the window function
described above. The convolved data is then sampled at regular intervals and 
a normal FFT function can be used to transform the data.

The naive choice for a GCK would be a sinc function, corresponding to a box-
shaped window function. But the sinc function has infinite support, and thus
convolution would be prohibitively expensive. Instead one chooses a GCK with
finite extent, usually just a handful of pixels. The associated window function
should contain as little power outside of the area of interest as possible. The
best choice of GCK is a prolate spheroidal wave function, but this isn't easy
to compute. Instead, GFFT uses the spherical bessel function, which is nearly as
good while being much easier to compute quickly.

The gridding operation requires O(W*N) operations, where W is the GCK width in 
pixels and N is the length of the data array to be transformed. This is ontop of 
the FFT, which requires O(N_x*log N_x) operations, where N_x is the size of the
grid onto which the data is placed in numbers of pixels. Gridding is performed 
in C via functions written in Cython, so the entire transform is still quite 
fast.

Numpy is licensed under the GPLv3. Please visit

http://www.gnu.org/licenses/gpl.html

for more information. If you have any questions or comments, please contact
Michael Bell at bellmr at gmail.com

References: 

For more information about gridding, see e.g.
- Beatty, Philip J., Dwight G. Nishimura, and John M. Pauly. "Rapid gridding 
reconstruction with a minimal oversampling ratio." Medical Imaging, IEEE 
Transactions on 24.6 (2005): 799-808.

- Schwab, F. R. "Optimal gridding of visibility data in radio interferometry." 
Indirect Imaging. Measurement and Processing for Indirect Imaging. Vol. 1. 1984.

- Briggs, Daniel S., Frederic R. Schwab, and Richard A. Sramek. "Imaging." 
Synthesis Imaging in Radio Astronomy II. Vol. 180. 1999.

** Dependencies ****************************************************************

GFFT is a Python function, so Python will obviously be required. The software
has been tested using Python 2.7, but will probably work fine with any 
Python 2.X version. At the moment we do not support Python 3, although a cross-
compatible version is planned. Other dependent packages include: 

- Numpy - http://numpy.scipy.org/
- Cython - http://cython.org/
- GSL - http://www.gnu.org/software/gsl/

** Installation ****************************************************************

To install GFFT: 

1) Install all dependencies

2) Enter the GFFT directory that you have unpacked or downloaded from a 
repository and type: 

python setup.py install

You may need to change the setup.py file to point to your GSL library location.
Just change the include_gsl_dir and lib_gsl_dir variables at the top of the 
file. This command will require super user priveledges (and therefore should be 
run using e.g. sudo). If you do not have such preveledges, instead you can run

python setup.py install --user

and the package will be installed in your local python package directory (e.g.
on Linux this is ~/.local/lib/python2.X/site-packages)

3) That's it! You're done. Leave the directory containing the source code, then
enter your favorite python environment and type "import gfft" to test. If error
messages pop up then something has gone wrong. Otherwise, you should be OK.
