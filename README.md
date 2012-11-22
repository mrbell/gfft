GFFT - Generalized Fast Fourier Transformations for Python
==========================================================

GFFT is a package for performing Fast Fourier Transformations even in the
case where the data is not defined on a regularly spaced grid. This is commonly
the case in applications such as radio interferometry and medical imaging. 

Features include:
  - Fast transformations between regular to regular, regular to irregular, 
    irregular to regular, and irregular to irregular spaces.
  - Efficiently use with Hermitian symmetric data. One can store only half of 
    the data, and the symmetric data are implied.
  - Handles mixtures of FFT and IFFT along different axes for multi-dimensional
    data.
  - Handles phase shifting of data for arbitrary axis definitions whether 
    zero-centered or centered at any other location along the input/output axes.

For more information, please refer to the [GFFT Wiki](https://bitbucket.org/mrbell/gfft/wiki).

GFFT is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).
