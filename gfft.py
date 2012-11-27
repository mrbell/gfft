"""
GFFT

This package mainly consists of a single function, gfft, which is a generalized 
Fourier transformation function that can transform between regularly- or 
irregularly-spaced, N-D fields. Gridding and degridding is performed when 
irregularly spaced fields are requested. Gridding is only supported for 1-, 2-,
or 3-D fields. The function also handles arbitrary phase shifts in the input and
output arrays.
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
import warnings

import gridding

VERSION = '0.5.0'

FTT_FFT = 'f'
FTT_IFFT = 'i'
FTT_NONE = 'n'
FFT = -1
IFFT = 1
NONE = 0
FT_TYPES = {FTT_FFT: FFT, FTT_IFFT: IFFT, FTT_NONE: NONE}
INV_FT_TYPES = {v:k for k, v in FT_TYPES.items()}
ii = complex(0,1)

class transform():
    """
    class transform
    
    Generic GFFT transformation class. This does nothing... it's just an abstract
    class for others to inherit from. Common attributes and methods that ALL 
    children will have are listed here. Method definitions listed below indicate 
    which args are required or not.
    
    Common attributes: 
        ndim - number of dimensions
        in_center - indicates whether the input array is centered on the reference 
            value (True) or the reference value is the zero index (False). It is a 
            list of booleans of length ndim. A scalar can also be given and it 
            applies to all dimensions. [False]
        out_center - same as the incenter attribute, but applies to the output of
            the transform.
        in_ref - an ndim length array or list of floats indicating the reference 
            value for each axis. This is used for applying the appropriate phase 
            shifts. If a scalar, the same reference applies to all axes. [0.]
        out_ref - same as inref but it applies to the output axis. [0]
        ft_type - An ndim length list of characters indicating which type of  
            Fourier transform should be performed along each axis. Valid entries 
            are 'f', 'i', or 'n' for fourier, inverse fourier, or none, 
            respectively. If none, no transformation is performed along the axis.
    
    Common methods:
        run - The function that actually applies the transformation. Takes a data 
            array as input. **Must be implemented by all children.**
        get_inverse_transform - Returns an instance of the class that will do the
            inverse transformation that is defined here. **Must be implemented by
            all children.**
        __init__ - Initialize the transformation, defining the attributes. **Must 
            be implemented by all children.**
        _check_data - Checks whether the input data superficially adheres to the 
            transform as it has been initialized. Takes a data array as input. 
            **Must be implemented by all children.**
        _fourier - the part common to all transforms that performs the Fourier 
            transformations and any shifts that are necessary. A default 
            implementation is provided in this class that can be called by its 
            children, so this does not need to be (and shouldn't be) overwritten.
    """    
    # TODO: Update docs
    
    # Common attributes #########################   
    ndim = 0
    
    in_ref = None
    out_ref = None
    
    do_fft = False
    do_ifft = False
    fftaxes = None
    ifftaxes = None
    ft_type = None

    do_pre_fftshift = False
    pre_fftshift_axes = None
    do_pre_ifftshift = False
    pre_ifftshift_axes = None
    
    do_post_fftshift = False
    post_fftshift_axes = None
    do_post_ifftshift = False
    post_ifftshift_axes = None
    
    in_axes = None
    out_axes = None
        
    # Common methods ############################
    def __init__(self):
        """
        Generic transformation class init. **Must be implemented by all 
        children.**
        
        Args:
            **May vary between child classes.**
        Returns:
            Nothing
        """
        print "Must overwrite the transform class init function!"
        pass
    
    def run(self,data):
        """
        Generic transformation class run method. **Must be implemented by all 
        children.**
        
        Args: 
            data - an ndarray of data to be transformed. If defined on an
                irregular space, this is a 1D data array regardless of the 
                problem dimension. If defined on a regular space, this is an
                ndim numpy array. 
        
        Returns: 
            result - an ndim array that is the transform of the input data.
        """
        print "Must overwrite the transform class run function!"
        pass
    
    def get_inverse_transform(self):
        """
        Generic transform class get_inverse_transform method. **Must be 
        implemented by all children.**
        
        Args:
            None
        
        Returns:
            An instance of a derivative of the transform class that performs the
                inverse transformation of the caller class.
        """
        print "Must overwrite the transform class get_inverse_transform function!"
        pass
    
    def set_ft_type(self, ft_type):
        """
        """
        [self.do_fft, self.fftaxes, self.do_ifft, self.ifftaxes, \
                self.ft_type] = self._parse_ft_type_arg(ft_type)
    
    def get_ft_type(self):
        """
        """
        temp_ft_type = []
        for i in range(self.ndim):
            temp_ft_type.append(INV_FT_TYPES[self.ft_type[i]])
            
            
        return temp_ft_type
    
    def _check_data(self, data):
        """
        Generic transform class _check_input method. **Must be implemented by all 
        children.**
        
        Args: 
            data - The data array to check against the attributes of the class 
                instance
        Returns:
            Boolean value indicating whether the data is OK.
        """
        print "Must overwrite the transform class _check_data function!"
        pass
    
    def _fourier(self, data):
        """
        The global fourier method that handles all regularly gridded fourier (and
        inverse fourier) transformations and any required phase shifting.
        
        Args:
            data - The *regularly gridded* data array to transform and shift.
        
        Returns: 
            A regularly gridded transform of the data array.
        """
        
        # Get the reference pixel in the zeroth entry of the data array
        if self.do_pre_fftshift:
            data2 = np.fft.fftshift(data, axes=self.pre_fftshift_axes)
        else: 
            data2 = data.copy()
        
        if self.do_pre_ifftshift:
            data2 = np.fft.ifftshift(data2, axes=self.pre_ifftshift_axes)
        
        for i in range(self.ndim):
            if self.out_ref[i] != 0. and self.ft_type[i] != NONE:
                shift = np.exp(self.ft_type[i]*np.pi*ii\
                    *self.out_ref[i]*self.in_axes[i])
                # I think this works...
                idx_obj = [None]*self.ndim
                idx_obj[i] = slice(None) 
                data2 = data2*shift[idx_obj]          
        
        if self.do_fft:
            out = np.fft.fftn(data2, axes=self.fftaxes)
        else:
            out = data2.copy()
        
        if self.do_ifft:
            out = np.fft.ifftn(out, axes=self.ifftaxes)
            
        for i in range(self.ndim):
            if self.in_ref[i] != 0. and self.ft_type[i] != NONE:
                shift = np.exp(self.ft_type[i]*np.pi*ii\
                    *self.in_ref[i]*(self.out_axes[i]+self.out_ref[i]))
                # I think this works...
                idx_obj = [None]*self.ndim
                idx_obj[i] = slice(None) 
                out = out*shift[idx_obj]
         
        if self.do_post_fftshift:
            out = np.fft.fftshift(out, axes=self.post_fftshift_axes)
        
        if self.do_post_ifftshift:
            out = np.fft.ifftshift(out, axes=self.post_ifftshift_axes)
        
        return out
    
    
    def _parse_ft_type_arg(self, ft_type):
        """
        An init check routine for the ft_type argument. Initialize ndim first!
        
        Args:
            ft_type - a valid ft_type argument.
        
        Returns: 
            A properly formatted ft_type attribute.
        """
        
        # flags to determine whether I need to use fftn and/or ifftn
        do_fft = False
        do_ifft = False
        # if you give an empty list to fftn in the axes position, nothing happens
        fftaxes = []
        ifftaxes = []
        
        new_ft_type = []
        
        if type(ft_type) == str:
            if ft_type.lower() == FTT_FFT:
                do_fft = True
                fftaxes = None # Does FFT on all axes
            elif ft_type.lower() == FTT_IFFT:
                do_ifft = True
                ifftaxes = None # Does IFFT on all axes
            
            for i in range(self.ndim):
                new_ft_type.append(FT_TYPES[ft_type.lower()])
                
        elif type(ft_type) == list:
            if len(ft_type) != self.ndim:
                raise Exception('ft_type is a list with invalid length')
            
            for i in range(len(ft_type)):
                if ft_type[i].lower() == FTT_FFT:
                    do_fft = True
                    fftaxes += [i]
                elif ft_type[i].lower() == FTT_IFFT:
                    do_ifft = True
                    ifftaxes += [i]
                
                new_ft_type.append(FT_TYPES[ft_type[i].lower()])
                    
        return do_fft, fftaxes, do_ifft, ifftaxes, new_ft_type
        
    def _validate_iterrable_types(self, l, t):
        """
        Used to check whether the types of the items within the list or tuple l 
        are of the type t. Returns True if all items are of type t, False 
        otherwise.
        """
        
        is_valid = True
        
        for i in range(len(l)):
            if type(l[i]) != t:
                is_valid = False
                
        return is_valid
    
    def _parse_center_arg(self, center_arg):
        """
        """
        
        do_fftshift = False
        do_ifftshift = False
        
        fftshift_axes = []
        ifftshift_axes = []
        
        if type(center_arg) == bool:
            
            for i in range(self.ndim):
                if center_arg and self.ft_type[i] == FFT:
                    do_fftshift = True
                    fftshift_axes.append(i)
                elif self.ft_type[i] == IFFT and center_arg:
                    ifftshift_axes.append(i)
                    do_ifftshift = True
                
        elif type(center_arg) == list:
            if len(center_arg) != self.ndim:
                raise Exception('center_arg is a list with invalid length')
                
            for i in range(self.ndim):
                if center_arg[i] and self.ft_type[i] == FFT:
                    do_fftshift = True
                    fftshift_axes.append(i)
                elif center_arg[i] and self.ft_type[i] == IFFT:
                    do_ifftshift = True
                    ifftshift_axes.append(i)
        
        return do_fftshift, fftshift_axes, do_ifftshift, ifftshift_axes
    
    def _parse_ref_arg(self, ref):
        """
        """
        new_ref = list()
        
        if np.isscalar(ref):
            # all axes use the same ft type
            for indx in range(self.ndim):
                new_ref.append(ref)
                        
        else:
            if len(ref) != self.ndim:
                raise TypeError("Invalid axis reference argument!")
            new_ref = ref
        
        return new_ref
        
    def _parse_regular_axes(self, axes):
        """
        Pass a list of tuples that define the gridded coordinate axes.
        The list contains one axis definition per dimsion of the data. If a tuple, 
        it contains (dx, nx) where dx is the pixel distance, and nx is the 
        number of pixels. Alternatively, a single tuple can be provided and it
        will apply to all coordinate axes. Can be 'None' if the reference pixel
        is zero and therefore no special phase shifting is required.
        """
        newaxes = list()
        if axes is None:
            for i in range(self.ndim):
                newaxes.append(None)
        elif type(axes) == tuple:
            for i in range(self.ndim):
                newaxes.append(np.arange(axes[1])*axes[0])
        elif type(axes) == list:
            if len(axes) != self.ndim:
                raise Exception("Incorrect number of axes has been provided.")
            else:
                for i in range(self.ndim):
                    if axes[i] is None:
                        newaxes.append(None)
                    elif type(axes[i]) == tuple:
                        newaxes.append(np.arange(axes[i][1])\
                            *axes[i][0])
                    elif type(axes[i]) == np.ndarray:
                        newaxes.append(axes[i])
                    else:
                        raise Exception("Invalid axis definition.")
        else:
            raise Exception("Invalid axis definition")
        
        return newaxes
    
    def __call__(self, data):
        
        return self.run(data)


class rrtransform(transform):
    """
    A GFFT regular-to-regular transform. This effectively just runs the usual 
    numpy fftn routine, but it will handle mixtures of FFTs and IFFTs along
    different axes for you, as well as any phase shifts (arbitrary or trivial
    fftshift type shifts).
    """
    
    def __init__(self, ndim, ft_type='f', in_center = False, out_center = False, \
        in_ref = 0., out_ref = 0., in_axes=None, out_axes=None):
            """
            initializes the rrtransform class.
            
            Args: 
                ndim - number of dimensions of the array to be transformed.
                ft_type - An ndim length list of characters indicating which type
                    of Fourier transform should be performed along each axis. 
                    Valid entries are 'f', 'i', or 'n' for fourier, inverse 
                    fourier, or none respectively. If none, no transformation is 
                    performed along the axis.
                in_center - indicates whether the input array is centered on 
                    the reference value (True) or the reference value is the zero 
                    index (False). It is a list of booleans of length ndim. 
                    A scalar can also be given and it applies to all dimensions. 
                    [False]
                out_center - Same as in_zero_center but for the output array.
                in_ref - an ndim length array or list of floats indicating the 
                    reference value for each axis. This is used for applying the 
                    appropriate phase shifts. If a scalar, the same reference 
                    applies to all axes. [0.]
                out_ref - same as inref but it applies to the output axis. [0]
                    
            Returns: 
                Nothing
            """
            
            self.ndim = ndim
            self.set_ft_type(ft_type)
            
            self.in_center = in_center
            self.out_center = out_center

            [self.do_pre_fftshift, self.pre_fftshift_axes, self.do_pre_ifftshift,\
                self.pre_ifftshift_axes] = self._parse_center_arg(in_center)
            [self.do_post_fftshift, self.post_fftshift_axes, \
                self.do_post_ifftshift, self.post_ifftshift_axes] \
                = self._parse_center_arg(out_center)
            
            self.in_ref = self._parse_ref_arg(in_ref)
            self.out_ref = self._parse_ref_arg(out_ref)
            
            self.in_axes = self._parse_regular_axes(in_axes)
            self.out_axes = self._parse_regular_axes(out_axes)
            
            for i in range(self.ndim):
                if self.in_ref[i] != 0. and self.out_axes[i] is None:
                    raise Exception("Must define out_axes if the in_ref "+\
                        "value is non-zero!")
                    
                if self.out_ref[i] != 0. and self.in_axes[i] is None:
                    raise Exception("Must define in_axes if the out_ref "+\
                        "value is non-zero!")
            
            print self.in_axes
            print self.out_axes
    
    def run(self, data):
        """
        Performs the regular to regular transformation.
        
        Args: 
            data - an ndim numpy array to be transformed. 
                
        Returns: 
            The requested transform of the input data.
        """
        
        if not self._check_data(data):
            raise Exception("Invalid data for this transformation!")
        
        return self._fourier(data)
    
    def get_inverse_transform(self):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        ft_type = []
        for i in range(self.ndim):
            if self.ft_type[i] == NONE:
                ft_type.append(FTT_NONE)
            elif self.ft_type[i] == FFT:
                ft_type.append(FTT_IFFT)
            else:
                ft_type.append(FTT_FFT)
                
        it = rrtransform(self.ndim, ft_type, self.out_center, self.in_center, \
            self.out_ref, self.in_ref, self.out_axes, self.in_axes)
        
        return it
    
    def _check_data(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        if type(data) != np.ndarray:
            raise Exception("Data must be a numpy array.")
            
        if data.ndim != self.ndim:
            raise Exception("Data has incorrect number of dimensions!")
        
        return True





class irtransform(transform):
    """
    Desc.
    
    Attributes:
    
    Methods:
    
    """

    def __init__(self, in_ax, out_ax, in_center = False, out_center = False,\
        in_ref = 0., out_ref = 0.):
            """
            Desc.
            Args:
                
            Returns:
                
            """
            pass
    
    def run(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def get_inverse_transform(self):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def _check_data(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    
    

    
class ritransform(transform):
    """
    Desc.
    
    Attributes:
    
    Methods:
    
    """

    def __init__(self, in_ax, out_ax, in_center = False, out_center = False,\
        in_ref = 0., out_ref = 0.):
            """
            Desc.
            Args:
                
            Returns:
                
            """
            pass
    
    def run(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def get_inverse_transform(self):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def _check_data(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass




class iitransform(transform):
    """
    Desc.
    
    Attributes:
    
    Methods:
    
    """

    def __init__(self, in_ax, out_ax, in_center = False, out_center = False,\
        in_ref = 0., out_ref = 0.):
            """
            Desc.
            Args:
                
            Returns:
                
            """
            pass
    
    def run(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def get_inverse_transform(self):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    def _check_data(self, data):
        """
        Desc.
        Args:
            
        Returns:
            
        """
        pass
    
    
    
    


def gfft(inp, in_ax=[], out_ax=[], ftmachine='fft', in_zero_center=True, \
    out_zero_center=True, enforce_hermitian_symmetry=False, W=6, alpha=1.5,\
    verbose=True):

    """
    gfft (Generalized FFT)
    
    
    def gfft(inp, in_ax=[], out_ax=[], ftmachine='fft', in_zero_center=True, \
        out_zero_center=True, out_is_real=False, W=6, alpha=1.5)
    
    This is a generalized Fourier transformation function that can transform
    between regularly- or irregularly-spaced, 1- 2- or 3-D fields. Gridding and
    degridding is performed when irregularly spaced fields are requested.
    
    input
    ------------------
    inp: The input data to be transformed. This can be a 1-, 2- or 3-D 
        (henceforth N-D) numpy array.
    
    in_ax, out_ax: The axes on which the input/output arrays are defined. There 
        are a few options here depending on the types of fields that are to be 
        transformed:
    
        To go from regularly spaced input to regularly spaced output: in can be 
            an N-D array, leave in_ax and out_ax blank. No gridding is
            performed, it just does an fft or ifft directly.
    
        To go from irregularly spaced input to regularly spaced output: in must
            be a list of 1-D arrays, in_ax = [N*array([...])] and 
            out_ax = [N*(dx, nx)]. So in_ax is a length N list of numpy arrays
            (each of length len(in)) that contain the coordinates for which the
            input data are defined. out_ax is a length N list of tuples
            containing the number of pixels and size of the pixels in the
            regularly spaced N-D out array. Gridding is performed on the input
            data before performing the fft or ifft.
        
        To go from regularly spaced input to irregularly spaced output: same as 
            above except in_ax and out_ax are reversed. out will always be a 1D 
            array. De-gridding is performed.
    
        To go from irregularly spaced input to irregularly spaced output: This
            gets a bit tricky. In this case either in_ax or out_ax = 
            ([N x array([...])], [N x (dx, nx)]) **this is a tuple** and the
            other is just [N x array([...])] as before. In this mode, the code
            grids in, Fourier transforms, then degrids onto the coordinates
            given in out_ax. The N tuples of (nx,dx) are necessary because a
            grid must be defined in the middle even though neither the input or
            output arrays live on a grid. The grid can be defined either for the
            input or output space (which is why either in_ax or out_ax can be
            given as a tuple).
    
    ftmachine: a length N list of strings, with each entry containing either
        'fft' or 'ifft'. This defines whether an FFT or and IFFT should be
        performed for each axis. So, if you have a 3D dataset and you want to do
        an FFT on the first two axes, but an IFFT on the last, you would pass 
        ftmachine=['fft', 'fft', 'ifft']. In principle, we could also make DFTs 
        an option here, and the code would just do a DFT rather than gridding.
        For an N-D input array, one could also just use ftmachine='fft' and it
        would do an fft for all axes. 
        
        For now, options include: 'fft', 'ifft', and 'none'.
    
    in_zero_center/out_zero_center: a length N list of booleans. True indicates 
        that the zero frequency is in (or should be in) the central pixel, false 
        indicates that it is in pixel 0. Basically this indicates whether 
        fftshifts should be performed before and after Fourier transforming. For
        an N-D array, in_zero_center=T would indicate that all axes should have 
        the zero channel in the central pixel.
    
    W, alpha: These are gridding parameters.
    
    enforce_hermitian_symmetry: A length N list of booleans. If the in array is
        to be gridded, setting this to 'True' indicates that the Hermitian
        conjugate of the input array needs to be generated during gridding.
        This can be set for each axis independently. This is ignored when going 
        from a regular grid to another regular grid.
        
        
    output
    ------------------
    out: A numpy array that contains the FT or IFT of inp. 
    
    """
    
    VERSION = "0.2.1"
    
    if verbose:
        print "gfft v. "+VERSION
    
    ############################################################################
    # Set some global variables
    
    # different modes of operation
    MODE_RR = 0 # regular grid to regular grid
    MODE_IR = 1 # irregular grid to regular grid
    MODE_RI = 2 # regular grid to irregular grid
    MODE_II = 3 # irregular grid to irregular grid
    
    mode_types = {MODE_RR:"regular to regular (no gridding)", \
        MODE_IR:"irregular to regular (gridding)", \
        MODE_RI:"regular to irregular (de-gridding)", \
        MODE_II:"irregular to irregular (gridding and degridding)"}
    
    # Different ftmachine options
    FTM_FFT = 'fft'
    FTM_IFFT = 'ifft'
    FTM_NONE = 'none'

    ############################################################################
    # Validate the inputs...
    
    if type(inp) != np.ndarray:
        raise TypeError('inp must be a numpy array.')
        
    if type(in_ax) != list and type(in_ax) != tuple:
        raise TypeError('in_ax must be either a list or a tuple.')        
    if type(out_ax) != list and type(out_ax) != tuple:
        raise TypeError('out_ax must be either a list or a tuple.')
    if type(out_ax) == tuple and type(in_ax) == tuple:
        raise TypeError('out_ax and in_ax cannot both be tuples')
        
    if type(in_ax) == tuple and (not validate_iterrable_types(in_ax, list)\
        or len(in_ax) != 2):
            raise TypeError('If in_ax is a tuple, it must contain two lists.')
    if type(out_ax) == tuple and (not validate_iterrable_types(out_ax, list)\
        or len(out_ax) != 2):
            raise TypeError('If out_ax is a tuple, it must contain two lists.')

    if type(in_ax) == tuple and \
        not validate_iterrable_types(in_ax[0], np.ndarray):
            raise TypeError('If in_ax is a tuple, it must contain two lists,' +\
                ' the first of which is a list of arrays.')
    if type(in_ax) == tuple and \
        not validate_iterrable_types(in_ax[1], tuple):
            raise TypeError('If in_ax is a tuple, it must contain two lists,' +\
                ' the second of which is a list of tuples.')
    
    if type(out_ax) == tuple and \
        not validate_iterrable_types(out_ax[0], np.ndarray):
            raise TypeError('If out_ax is a tuple, it must contain two lists,'+\
                ' the first of which is a list of arrays.')
    if type(out_ax) == tuple and \
        not validate_iterrable_types(out_ax[1], tuple):
            raise TypeError('If out_ax is a tuple, it must contain two lists,'+\
                ' the second of which is a list of tuples.')
    
    if type(W) != int:
        raise TypeError('W must be an integer.')
    if type(alpha) != float and type(alpha) != int:
        raise TypeError('alpha must be a float or int.')
        
    if (type(ftmachine) != str and type(ftmachine) != list) or \
        (type(ftmachine) == list and \
        not validate_iterrable_types(ftmachine, str)):
            raise TypeError('ftmachine must be a string or a list of strings.')
            
    if (type(in_zero_center) != bool and type(in_zero_center) != list) or \
        (type(in_zero_center) == list and \
        not validate_iterrable_types(in_zero_center, bool)):
            raise TypeError('in_zero_center must be a Bool or list of Bools.')
            
    if (type(out_zero_center) != bool and type(out_zero_center) != list) or \
        (type(out_zero_center) == list and \
        not validate_iterrable_types(out_zero_center, bool)):
            raise TypeError('out_zero_center must be a Bool or list of Bools.')
            
    if (type(enforce_hermitian_symmetry) != bool and \
        type(enforce_hermitian_symmetry) != list) or \
        (type(enforce_hermitian_symmetry) == list and \
        not validate_iterrable_types(enforce_hermitian_symmetry, bool)):
            raise TypeError('enforce_hermitian_symmetry must be a Bool '\
                +'or list of Bools.')

    
    ############################################################################
    # figure out how many dimensions we are talking about, and what mode we
    # want to use
    
    N = 0 # number of dimensions
    mode = -1    
    
    if len(in_ax) == 0:
        # regular to regular transformation
        mode = MODE_RR
        N = inp.ndim
        if len(out_ax) != 0:
            warnings.warn('in_ax is empty, indicating regular to regular '\
                +'transformation is requested, but out_ax is not empty. '+\
                'Ignoring out_ax and proceeding with regular to regular mode.')             
    elif type(in_ax)==tuple or type(out_ax)==tuple:
        # irregular to irregular transformation
        mode = MODE_II
        if type(out_ax)==tuple:
            if len(out_ax) != 2:
                raise TypeError('Invalid out_ax for '+\
                    'irregular to irregular mode.')
            N = len(in_ax)
        else:
            if len(in_ax) != 2:
                raise TypeError('Invalid in_ax for '+\
                    'irregular to irregular mode.')
            N = len(out_ax)    
    else:
        if type(in_ax[0])==tuple:
            # regular to irregular transformation
            mode = MODE_RI
        else:
            # irregular to regular transformation
            mode = MODE_IR
        N = len(in_ax)
        if len(out_ax) != len(in_ax):
            raise TypeError('For regular to irregular mode, len(in_ax) must '+\
                'equal len(out_ax).') 
        
    if N==0 or mode == -1:
        raise Exception('Something went wrong in setting the mode and ' \
            + 'dimensionality.')
            
    if N > 3 and mode != MODE_RR:
        raise Exception('Gridding has been requested for an unsupported '+\
            'number of dimensions!')
        
    if verbose:
        print 'Requested mode = ' + mode_types[mode]
        print "Number of dimensions = " + str(N)
        
    ############################################################################
    # Figure out which axes should have which transforms applied to them    
    
    # flags to determine whether I need to use fftn and/or ifftn
    do_fft = False
    do_ifft = False
    # if you give an empty list to fftn in the axes position, nothing happens
    fftaxes = []
    ifftaxes = []
    
    if type(ftmachine) == str:
        if ftmachine.lower() == FTM_FFT:
            do_fft = True
            fftaxes = None
        elif ftmachine.lower() == FTM_IFFT:
            do_ifft = True
            ifftaxes = None
    elif type(ftmachine) == list:
        if len(ftmachine) != N:
            raise Exception('ftmachine is a list with invalid length')
        
        for i in range(len(ftmachine)):
            if ftmachine[i].lower() == FTM_FFT:
                do_fft = True
                fftaxes += [i]
            elif ftmachine[i].lower() == FTM_IFFT:
                do_ifft = True
                ifftaxes += [i]

# As requested by Marco, if no FFT is requested, the function will still 
# perform a shift.
    if (do_fft == False and do_ifft == False) or \
        (fftaxes == [] and ifftaxes == []):
            warnings.warn('No Fourier transformation requested, only '+\
                'shifting will be performed!')
#            return
            mode = MODE_RR #Since gridding will not be needed, just use RR mode
            
    ############################################################################
    # figure out which axes need to be shifted (before and after FT)

    do_preshift = False
    do_postshift = False
    
    preshift_axes = []
    postshift_axes = []    
    
    if type(in_zero_center) == bool:
        if in_zero_center:
            do_preshift = True
            preshift_axes = None
    elif type(in_zero_center) == list:
        if len(in_zero_center) != N:
            raise Exception('in_zero_center is a list with invalid length')
            
        for i in range(len(in_zero_center)):
            if in_zero_center[i]:
                do_preshift = True
                preshift_axes += [i]
    
    if type(out_zero_center) == bool:
        if out_zero_center:
            do_postshift = True
            postshift_axes = None
    elif type(out_zero_center) == list:
        if len(out_zero_center) != N:
            raise Exception('out_zero_center is a list with invalid length')
            
        for i in range(len(out_zero_center)):
            if out_zero_center[i]:
                do_postshift = True
                postshift_axes += [i]
            
    ############################################################################
    # figure out which axes need to be hermitianized
    
    hermitianized_axes = []
    
    if type(enforce_hermitian_symmetry) == bool:
        if enforce_hermitian_symmetry:
            for i in range(N):
                hermitianized_axes += [True]
        else:
            for i in range(N):
                hermitianized_axes += [False]
                
    elif type(enforce_hermitian_symmetry) == list:
        if len(enforce_hermitian_symmetry) != N:
            raise Exception('enforce_hermitian_symmetry is a list with '+\
                'invalid length')
            
        for i in range(len(enforce_hermitian_symmetry)):
            if enforce_hermitian_symmetry[i]:
                hermitianized_axes += [True]
            else:
                hermitianized_axes += [False]
    
    if len(hermitianized_axes) != N:    
        raise Exception('Something went wrong when setting up the '+\
            'hermitianized_axes list!')
            
    ############################################################################
    # Print operation summary
    
    if verbose:
        print ""
        print "Axis#, FFT, IFFT, ZCIN, ZCOUT, HERM"
    
        for i in range(N):
            pstr = str(N)+', '
            
            if fftaxes == None or fftaxes.count(i)>0:
                pstr = pstr + 'True, '
            else:
                pstr = pstr + 'False, '
            
            if ifftaxes == None or ifftaxes.count(i)>0:
                pstr = pstr + 'True, '
            else:
                pstr = pstr + 'False, '
            
            if preshift_axes == None or preshift_axes.count(i)>0:
                pstr = pstr + 'True, '
            else:
                pstr = pstr + 'False, '
            
            if postshift_axes == None or postshift_axes.count(i)>0:
                pstr = pstr + 'True, '
            else:
                pstr = pstr + 'False, '
                
            if hermitianized_axes[i]:
                pstr = pstr + 'True'
            else:
                pstr = pstr + 'False'
            
        print pstr
    
    ############################################################################
    # Do MODE_RR transform
    
    if mode == MODE_RR:
        if do_preshift:
            inp = np.fft.fftshift(inp, axes=preshift_axes)
        
        if do_fft:
            out = np.fft.fftn(inp, axes=fftaxes)
        else:
            out = inp.copy()
        
        if do_ifft:
            out = np.fft.ifftn(out, axes=ifftaxes)
            
        if do_postshift:
            out = np.fft.fftshift(out, axes=postshift_axes)
            
        if verbose:
            print "Done!"
            print ""
            
        return out
    
    ############################################################################
    # Do MODE_IR transform
    
    elif mode == MODE_IR:
        
        # all gridding code assumes the data array is complex
        inp = np.array(inp, dtype=complex)

        # grid
        if N == 1:
            dx = out_ax[0][0]
            Nx = out_ax[0][1]
            xmin = 0.
            if do_postshift:
                xmin = -0.5*Nx*dx
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            if do_preshift:
                umin = -0.5*Nu*du
                
            inp_grid = gridding.grid_1d(in_ax[0], inp, du, Nu, umin, W, alpha, \
                hermitianized_axes[0])

        elif N == 2:
            dx = out_ax[0][0]
            Nx = out_ax[0][1]
            xmin = 0.
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            
            dy = out_ax[1][0]
            Ny = out_ax[1][1]
            ymin = 0.
            dv = 1./dy/Ny/alpha
            Nv = alpha*Ny
            vmin = 0.
            
            if do_preshift:
                if preshift_axes == None: #shift all axes
                    vmin = -0.5*Nv*dv
                    umin = -0.5*Nu*du
                else: #only shift some axes...
                    if preshift_axes.count(1) > 0:
                        vmin = -0.5*Nv*dv
                    if preshift_axes.count(0) > 0:
                        umin = -0.5*Nu*du
            if do_postshift:
                if postshift_axes == None: #shift all axes
                    xmin = -0.5*Nx*dx
                    ymin = -0.5*Ny*dy
                else: #only shift some axes...
                    if postshift_axes.count(1) > 0:
                        ymin = -0.5*Ny*dy
                    if postshift_axes.count(0) > 0:
                        xmin = -0.5*Nx*dx
                    
            inp_grid = gridding.grid_2d(in_ax[0], in_ax[1], inp, du, Nu, umin, \
                dv, Nv, vmin, alpha, W, \
                hermitianized_axes[0], hermitianized_axes[1])
            
            
        elif N == 3:
            dx = out_ax[0][0]
            Nx = out_ax[0][1]
            xmin = 0.
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            
            dy = out_ax[1][0]
            Ny = out_ax[1][1]
            ymin = 0.
            dv = 1./dy/Ny/alpha
            Nv = alpha*Ny
            vmin = 0.
            
            dz = out_ax[2][0]
            Nz = out_ax[2][1]
            zmin = 0.
            dw = 1./dz/Nz/alpha
            Nw = alpha*Nz
            wmin = 0.

                
            if do_preshift:
                if preshift_axes == None:
                    vmin = -0.5*Nv*dv
                    umin = -0.5*Nu*du
                    wmin = -0.5*Nw*dw
                else:
                    if preshift_axes.count(1) > 0:
                        vmin = -0.5*Nv*dv
                    if preshift_axes.count(0) > 0:
                        umin = -0.5*Nu*du
                    if preshift_axes.count(2) > 0:
                        wmin = -0.5*Nw*dw
            if do_postshift:
                if postshift_axes == None:
                    xmin = -0.5*Nx*dx
                    ymin = -0.5*Ny*dy
                    zmin = -0.5*Nz*dz
                else:
                    if postshift_axes.count(1) > 0:
                        ymin = -0.5*Ny*dy
                    if postshift_axes.count(0) > 0:
                        xmin = -0.5*Nx*dx
                    if postshift_axes.count(2) > 0:
                        zmin = -0.5*Nz*dz
                
            inp_grid = gridding.grid_3d(in_ax[0], in_ax[1], in_ax[2], inp, \
                du, Nu, umin, dv, Nv, vmin, dw, Nw, wmin, W, alpha, \
                hermitianized_axes[0], hermitianized_axes[1], \
                hermitianized_axes[2])
        
        if do_preshift:
            inp_grid = np.fft.fftshift(inp_grid, axes=preshift_axes)
        
        if do_fft:
            out = np.fft.fftn(inp_grid, axes=fftaxes)
        else:
            out = inp_grid.copy()
        
        if do_ifft:
            out = np.fft.ifftn(out, axes=ifftaxes)
            
        # shift    
        if do_postshift:
            out = np.fft.fftshift(out, axes=postshift_axes)
                
        # crop & grid correct
        if N == 1:
            tndxx = int(0.5*Nx*(alpha-1))
            if do_postshift:
                out = out[tndxx:tndxx+Nx]
            else:
                out = out[0:Nx]
            gc = gridding.get_grid_corr_1d(dx, Nx, xmin, du, W, alpha)
            
        elif N == 2:
            tndxx = int(0.5*Nx*(alpha-1))
            tndxy = int(0.5*Ny*(alpha-1))
            xl = 0
            yl = 0
            
            if do_postshift:
                if postshift_axes == None:
                    xl = tndxx
                    yl = tndxy
                else:
                    if postshift_axes.count(0)>0:
                        xl = tndxx
                    if postshift_axes.count(1)>0:
                        yl = tndxy
                        
            out = out[xl:xl+Nx, yl:yl+Ny]
            gc = gridding.get_grid_corr_2d(dx, Nx, xmin, dy, Ny, ymin, \
                du, dv, W, alpha)
            
        elif N == 3:
            tndxx = int(0.5*Nx*(alpha-1))
            tndxy = int(0.5*Ny*(alpha-1))
            tndxz = int(0.5*Nz*(alpha-1))
            xl = 0
            yl = 0
            zl = 0
            
            if do_postshift:
                if postshift_axes == None:
                    xl = tndxx
                    yl = tndxy
                    zl = tndxz
                else:
                    if postshift_axes.count(0)>0:
                        xl = tndxx
                    if postshift_axes.count(1)>0:
                        yl = tndxy
                    if postshift_axes.count(2)>0:
                        zl = tndxz
                        
            out = out[xl:xl+Nx, yl:yl+Ny, zl:zl+Nz]
            gc = gridding.get_grid_corr_3d(dx, Nx, xmin, dy, Ny, ymin, \
                dz, Nz, zmin, du, dv, dw, W, alpha)
                
        if verbose:
            print "Done!"
            print ""
        
        return out/gc

    ############################################################################
    # Do MODE_RI transform
    
    elif mode == MODE_RI:

        # all gridding code assumes the data array is complex
        inp = np.array(inp, dtype=complex)

        # grid basics 
        if N == 1:
            dx = in_ax[0][0]
            Nx = in_ax[0][1]
            xmin = 0.
            if do_preshift:
                xmin = -0.5*Nx*dx
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            if do_postshift:
                umin = -0.5*Nu*du

        elif N == 2:
            dx = in_ax[0][0]
            Nx = in_ax[0][1]
            xmin = 0.
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            
            dy = in_ax[1][0]
            Ny = in_ax[1][1]
            ymin = 0.
            dv = 1./dy/Ny/alpha
            Nv = alpha*Ny
            vmin = 0.
                
            if do_postshift:
                if postshift_axes == None:
                    vmin = -0.5*Nv*dv
                    umin = -0.5*Nu*du
                else:
                    if postshift_axes.count(1) > 0:
                        vmin = -0.5*Nv*dv
                    if postshift_axes.count(0) > 0:
                        umin = -0.5*Nu*du
            if do_preshift:
                if preshift_axes == None:
                    xmin = -0.5*Nx*dx
                    ymin = -0.5*Ny*dy
                else:
                    if preshift_axes.count(1) > 0:
                        ymin = -0.5*Ny*dy
                    if preshift_axes.count(0) > 0:
                        xmin = -0.5*Nx*dx

        elif N == 3:
            dx = in_ax[0][0]
            Nx = in_ax[0][1]
            xmin = 0.
            du = 1./dx/Nx/alpha
            Nu = alpha*Nx
            umin = 0.
            
            dy = in_ax[1][0]
            Ny = in_ax[1][1]
            ymin = 0.
            dv = 1./dy/Ny/alpha
            Nv = alpha*Ny
            vmin = 0.
            
            dz = in_ax[2][0]
            Nz = in_ax[2][1]
            zmin = 0.
            dw = 1./dz/Nz/alpha
            Nw = alpha*Nz
            wmin = 0.
            
            if do_postshift:
                if postshift_axes == None:
                    vmin = -0.5*Nv*dv
                    umin = -0.5*Nu*du
                    wmin = -0.5*Nw*dw
                else:
                    if postshift_axes.count(1) > 0:
                        vmin = -0.5*Nv*dv
                    if postshift_axes.count(0) > 0:
                        umin = -0.5*Nu*du
                    if postshift_axes.count(2) > 0:
                        wmin = -0.5*Nw*dw
            if do_preshift:
                if preshift_axes == None:
                    xmin = -0.5*Nx*dx
                    ymin = -0.5*Ny*dy
                    zmin = -0.5*Nz*dz
                else:
                    if preshift_axes.count(1) > 0:
                        ymin = -0.5*Ny*dy
                    if preshift_axes.count(0) > 0:
                        xmin = -0.5*Nx*dx
                    if preshift_axes.count(2) > 0:
                        zmin = -0.5*Nz*dz
        

        # degrid correct & enlargement
        if N == 1:
            tndxx = int(0.5*Nx*(alpha-1))
            inp = inp/gridding.get_grid_corr_1d(dx, Nx, xmin, du, W, alpha)
            
            inp_oversam = np.zeros(Nu, dtype=complex)
            
            xl = 0
            
            if do_preshift:
                xl = tndxx
            
            inp_oversam[xl:xl+Nx] = inp

        elif N == 2:
            tndxx = int(0.5*Nx*(alpha-1))
            tndxy = int(0.5*Ny*(alpha-1))
            inp = inp/gridding.get_grid_corr_2d(dx, Nx, xmin, dy, Ny, ymin, \
                du, dv, W, alpha)
            inp_oversam = np.zeros((Nu,Nv), dtype=complex)
            
            xl = 0
            yl = 0
            
            if do_preshift:
                if preshift_axes == None:
                    xl = tndxx
                    yl = tndxy
                else:
                    if preshift_axes.count(0) > 0:
                        xl = tndxx
                    if preshift_axes.count(1) > 0:
                        yl = tndxy
            
            inp_oversam[xl:xl+Nx, yl:yl+Ny] = inp


        elif N == 3:
            tndxx = int(0.5*Nx*(alpha-1))
            tndxy = int(0.5*Ny*(alpha-1))
            tndxz = int(0.5*Nz*(alpha-1))
            inp = inp/gridding.get_grid_corr_3d(dx, Nx, xmin, dy, Ny, ymin, \
                dz, Nz, zmin, du, dv, dz, W, alpha)
            inp_oversam = np.zeros((Nu,Nv,Nw))
            
            xl = 0
            yl = 0
            zl = 0
            
            if do_preshift:
                if preshift_axes == None:
                    xl = tndxx
                    yl = tndxy
                    zl = tndxz
                else:
                    if preshift_axes.count(0) > 0:
                        xl = tndxx
                    if preshift_axes.count(1) > 0:
                        yl = tndxy
                    if preshift_axes.count(2) > 0:
                        zl = tndxz            
            
            inp_oversam[xl:xl+Nx, yl:yl+Ny, zl:zl+Nz] = inp
        
        # shift
        if do_preshift:
            inp_oversam = np.fft.fftshift(inp_oversam, axes=preshift_axes)
        
        # fft
        if do_fft:
            out = np.fft.fftn(inp_oversam, axes=fftaxes)
        else:
            out = inp_oversam.copy()
        
        if do_ifft:
            out = np.fft.ifftn(out, axes=ifftaxes)
            
        # shift    
        if do_postshift:
            out = np.fft.fftshift(out, axes=postshift_axes)

        # degrid
        if N == 1:
            out_degrid = gridding.degrid_1d(out_ax[0], out, du, Nu, umin,\
                alpha, W)

        elif N == 2:
            out_degrid = gridding.degrid_2d(out_ax[0], out_ax[1], out, du, Nu, \
                umin, dv, Nv, vmin,  alpha, W)

        elif N == 3:
            out_degrid = gridding.degrid_3d(out_ax[0], out_ax[1], out_ax[2], \
                out, du, Nu, umin, dv, Nv, vmin, dw, Nw, wmin, alpha, W)
        
        if verbose:
            print "Done!"
            print ""

        return out_degrid
        
    ############################################################################
    # Do MODE_II transform
    
    elif mode == MODE_II:

        #defining the grids
        if type(in_ax) == tuple:
            raise Exception("Defining grid on in_ax in MODE_II not yet "\
                +"supported...")
            # everything in here is out of date... needs total overhaul!
            if N == 1:
                du = in_ax[1][0][0]
                Nu = in_ax[1][0][1]*alpha 
                umin = 0.
                if do_preshift:
                    umin = -0.5*Nu*du
                dx = 1./du/Nu
                Nx = Nu
                umin = 0.
                if do_postshift:
                    umin = -0.5*Nu*du

                in_ax = in_ax[0]

            elif N == 2:
                dx = in_ax[1][0][0]
                Nx = in_ax[1][0][1]
                xmin = 0.
                if type(in_zero_center)==bool and in_zero_center==True:
                    xmin = -0.5*Nx*dx
                elif in_zero_center[0] == True:
                    xmin = -0.5*Nx*dx
                du = 1./dx/Nx/alpha
                Nu = alpha*Nx
                umin = 0.
                if do_postshift:
                    umin = -0.5*Nu*du
            
                dy = in_ax[1][1][0]
                Ny = in_ax[1][1][1]
                ymin = 0.
                if type(in_zero_center)==bool and in_zero_center==True:
                    ymin = -0.5*Ny*dy
                elif in_zero_center[1] == True:
                    ymin = -0.5*Ny*dy
                dv = 1./dy/Ny/alpha
                Nv = alpha*Ny
                vmin = 0.
                if do_postshift:
                    vmin = -0.5*Nv*dv

                in_ax = in_ax[0]

            elif N == 3:
                dx = in_ax[1][0][0]
                Nx = in_ax[1][0][1]
                xmin = 0.
                if type(in_zero_center)==bool and in_zero_center==True:
                    xmin = -0.5*Nx*dx
                elif in_zero_center[0] == True:
                    xmin = -0.5*Nx*dx
                du = 1./dx/Nx/alpha
                Nu = alpha*Nx
                umin = 0.
                if do_postshift:
                    umin = -0.5*Nu*du
            
                dy = in_ax[1][1][0]
                Ny = in_ax[1][1][1]
                ymin = 0.
                if type(in_zero_center)==bool and in_zero_center==True:
                    ymin = -0.5*Ny*dy
                elif in_zero_center[1] == True:
                    ymin = -0.5*Ny*dy
                dv = 1./dy/Ny/alpha
                Nv = alpha*Ny
                vmin = 0.
                if do_postshift:
                    vmin = -0.5*Nv*dv
            
                dz = in_ax[1][2][0]
                Nz = in_ax[1][2][1]
                zmin = 0.
                if type(in_zero_center)==bool and in_zero_center==True:
                    zmin = -0.5*Nz*dz
                elif in_zero_center[2] == True:
                    zmin = -0.5*Nz*dz
                dw = 1./dz/Nz/alpha
                Nw = alpha*Nz
                wmin = 0.
                if do_postshift:
                    wmin = -0.5*Nw*dw
                    
                in_ax = in_ax[0]

        else:

            if N == 1:
                dx = out_ax[1][0][0]
                Nx = out_ax[1][0][1]
                xmin = 0.
                if do_postshift:
                    xmin = -0.5*Nx*dx
                du = 1./dx/Nx/alpha
                Nu = Nx*alpha
                umin = 0.
                if do_preshift:
                    umin = -0.5*Nu*du
                
                out_ax = out_ax[0]

            elif N == 2:
                dx = out_ax[1][0][0]
                Nx = out_ax[1][0][1]
                xmin = 0.
                du = 1./dx/Nx/alpha
                Nu = alpha*Nx
                umin = 0.
            
                dy = out_ax[1][0]
                Ny = out_ax[1][1]
                ymin = 0.
                dv = 1./dy/Ny/alpha
                Nv = alpha*Ny
                vmin = 0.
                    
                if do_preshift:
                    if preshift_axes == None:
                        vmin = -0.5*Nv*dv
                        umin = -0.5*Nu*du
                    else:
                        if preshift_axes.count(1) > 0:
                            vmin = -0.5*Nv*dv
                        if preshift_axes.count(0) > 0:
                            umin = -0.5*Nu*du
                if do_postshift:
                    if postshift_axes == None:
                        xmin = -0.5*Nx*dx
                        ymin = -0.5*Ny*dy
                    else:
                        if postshift_axes.count(1) > 0:
                            ymin = -0.5*Ny*dy
                        if postshift_axes.count(0) > 0:
                            xmin = -0.5*Nx*dx

                out_ax = out_ax[0]
                
            
            elif N == 3:
                dx = out_ax[1][0][0]
                Nx = out_ax[1][0][1]
                xmin = 0.
                du = 1./dx/Nx/alpha
                Nu = alpha*Nx
                umin = 0.
            
                dy = out_ax[1][1][0]
                Ny = out_ax[1][1][1]
                ymin = 0.
                dv = 1./dy/Ny/alpha
                Nv = alpha*Ny
                vmin = 0.
            
                dz = out_ax[1][2][0]
                Nz = out_ax[1][2][1]
                zmin = 0.
                dw = 1./dz/Nz/alpha
                Nw = alpha*Nz
                wmin = 0.
                    
                if do_preshift:
                    if preshift_axes == None:
                        vmin = -0.5*Nv*dv
                        umin = -0.5*Nu*du
                        wmin = -0.5*Nw*dw
                    else:
                        if preshift_axes.count(1) > 0:
                            vmin = -0.5*Nv*dv
                        if preshift_axes.count(0) > 0:
                            umin = -0.5*Nu*du
                        if preshift_axes.count(2) > 0:
                            wmin = -0.5*Nw*dw
                if do_postshift:
                    if postshift_axes == None:
                        xmin = -0.5*Nx*dx
                        ymin = -0.5*Ny*dy
                        zmin = -0.5*Nz*dz
                    else:
                        if postshift_axes.count(1) > 0:
                            ymin = -0.5*Ny*dy
                        if postshift_axes.count(0) > 0:
                            xmin = -0.5*Nx*dx
                        if postshift_axes.count(2) > 0:
                            zmin = -0.5*Nz*dz

                out_ax = out_ax[0]

        # all gridding code assumes the data array is complex
        inp = np.array(inp, dtype=complex)

        # grid
        if N == 1:
            inp_grid = gridding.grid_1d(in_ax[0], inp, du, Nu, umin, W, alpha, \
                hermitianized_axes[0])
        elif N == 2:    
            inp_grid = gridding.grid_2d(in_ax[0], in_ax[1], inp, du, Nu, umin, \
                dv, Nv, vmin, W, alpha, \
                hermitianized_axes[0], hermitianized_axes[1])
        elif N == 3:    
            inp_grid = gridding.grid_3d(in_ax[0], in_ax[1], in_ax[2], inp, \
                du, Nu, umin, dv, Nv, vmin, dw, Nw, wmin, W, alpha, \
                hermitianized_axes[0], hermitianized_axes[1], \
                hermitianized_axes[2])
        
        # degrid correct
        if N == 1:
            inp_grid = inp_grid/gridding.get_grid_corr_1d(du, Nu, umin, \
                dx/alpha, W, alpha)
                
            inp_grid_os = np.zeros(Nu*alpha, dtype=complex)

            ul = 0
            if do_preshift:
                ul = 0.5*Nu*(alpha-1.)
                    
            inp_grid_os[ul:ul+Nu] = inp_grid
            inp_grid = inp_grid_os
            del inp_grid_os
            
        elif N == 2:
            inp_grid = inp_grid/gridding.get_grid_corr_2d(du, Nu, umin, \
                dv, Nv, vmin, dx/alpha, dy/alpha, W, alpha)
                
            inp_grid_os = np.zeros((Nu*alpha, Nv*alpha), dtype=complex)

            ul = 0
            vl = 0
            
            if do_preshift:
                if preshift_axes == None:
                    ul = 0.5*Nu*(alpha-1.)
                    vl = 0.5*Nv*(alpha-1.)
                else:
                    if preshift_axes.count(0) > 0:
                        ul = 0.5*Nu*(alpha-1.)
                    if preshift_axes.count(1) > 0:
                        vl = 0.5*Nv*(alpha-1.)
                    
            inp_grid_os[ul:ul+Nu, vl:vl+Nv] = inp_grid
            inp_grid = inp_grid_os
            del inp_grid_os
            
        elif N == 3:
            inp_grid = inp_grid/gridding.get_grid_corr_3d(du, Nu, umin, \
                dv, Nv, vmin, dw, Nw, wmin, dx/alpha, dy/alpha, dz/alpha, \
                W, alpha)
                
            inp_grid_os = np.zeros((Nu*alpha, Nv*alpha, Nw*alpha), \
                dtype=complex)

            ul = 0
            vl = 0
            wl = 0
            
            if do_preshift:
                if preshift_axes == None:
                    ul = 0.5*Nu*(alpha-1.)
                    vl = 0.5*Nv*(alpha-1.)
                    wl = 0.5*Nw*(alpha-1.)
                else:
                    if preshift_axes.count(0) > 0:
                        ul = 0.5*Nu*(alpha-1.)
                    if preshift_axes.count(1) > 0:
                        vl = 0.5*Nv*(alpha-1.)
                    if preshift_axes.count(2) > 0:
                        wl = 0.5*Nw*(alpha-1.)
                    
            inp_grid_os[ul:ul+Nu, vl:vl+Nv, wl:wl+Nw] = inp_grid
            inp_grid = inp_grid_os
            del inp_grid_os

        # shift
        if do_preshift:
            inp_grid = np.fft.fftshift(inp_grid, axes=preshift_axes)
        
        # fft
        if do_fft:
            out = np.fft.fftn(inp_grid, axes=fftaxes)
        else:
            out = inp_grid.copy()
        
        if do_ifft:
            out = np.fft.ifftn(out, axes=ifftaxes)
            
        # shift    
        if do_postshift:
            out = np.fft.fftshift(out, axes=postshift_axes)
                
        # grid correct
        if N == 1:
#            print "Now here..."
#            print len(out)
#            print alpha**2*Nx
            out = out/gridding.get_grid_corr_1d(dx/alpha, alpha**2*Nx, \
                alpha*xmin, du, W, alpha)
            
        elif N == 2:
            out = out/gridding.get_grid_corr_2d(dx/alpha, alpha**2*Nx, \
                alpha*xmin, dy/alpha, alpha**2*Ny, alpha*ymin, \
                du, dv, W, alpha)
            
        elif N == 3:
            out = out/gridding.get_grid_corr_3d(dx/alpha, alpha**2*Nx, \
                alpha*xmin, dy/alpha, alpha**2*Ny, alpha*ymin, \
                dz/alpha, alpha**2*Nz, alpha*zmin, du, dv, dw, W, alpha)

        # degrid
        if N == 1:
            out_degrid = gridding.degrid_1d(out_ax[0], out, dx/alpha, \
                alpha**2*Nx, alpha*xmin, alpha, W)

        elif N == 2:
            out_degrid = gridding.degrid_2d(out_ax[0], out_ax[1], out, \
                dx/alpha, alpha**2*Nx, alpha*xmin, dy/alpha, alpha**2*Ny, \
                alpha*ymin, alpha, W)

        elif N == 3:
            out_degrid = gridding.degrid_3d(out_ax[0], out_ax[1], out_ax[2], \
                out, dx/alpha, alpha**2*Nx, alpha*xmin, dy/alpha, alpha**2*Ny, \
                alpha*ymin, dz/alpha, alpha**2*Nz, alpha*zmin, alpha, W)
        
        if verbose:
            print "Done!"
            print ""

        return out_degrid
        
    
def validate_iterrable_types(l, t):
    """
    Used to check whether the types of the items within the list or tuple l are 
    of the type t. Returns True if all items are of type t, False otherwise.
    """
    
    is_valid = True
    
    for i in range(len(l)):
        if type(l[i]) != t:
            is_valid = False
            
    return is_valid
    

def dft(in_vals, in_ax, out_ax):
    """ 
    A function that transforms a list of values using a discrete Fourier 
    transformation. Works for arbitrary number of dimensions.
    
    in_ax/out_ax must be a list of numpy arrays, one array for each axis.
    """
    
    nax = len(in_ax)
    if len(out_ax) != len(in_ax):
        raise Exception('dft: number of input and output dimensions not equal!')
    
    nin = len(in_vals)
    nout = len(out_ax[0])
    
    for i in range(nax):
        if len(in_ax[i]) != nin:
            raise Exception('dft: input axis length invalid')
        if len(out_ax[i]) != nout:
            raise Exception('dft: output axis length invalid')

    out_vals = np.zeros(nout, dtype=complex)
    
    for i in range(nout):
        val = complex(0,0)
        for j in range(nin):
            psum = 0
            for k in range(nax):
                psum += in_ax[k][j]*out_ax[k][i]
            cphs = -2.*np.pi*psum
            val += in_vals[j]*complex(np.cos(cphs),np.sin(cphs))
        out_vals[i] = val
#        progress(20, i+1., nk)
    
    return out_vals/len(in_vals)

def idft(in_vals, in_ax, out_ax):
    """ 
    A function that transforms a list of values using a discrete Fourier 
    transformation. Works for arbitrary number of dimensions.
    
    in_ax/out_ax must be a list of numpy arrays, one array for each axis.
    """
    
    nax = len(in_ax)
    if len(out_ax) != len(in_ax):
        raise Exception('dft: number of input and output dimensions not equal!')
    
    nin = len(in_vals)
    nout = len(out_ax[0])
    
    for i in range(nax):
        if len(in_ax[i]) != nin:
            raise Exception('dft: input axis length invalid')
        if len(out_ax[i]) != nout:
            raise Exception('dft: output axis length invalid')

    out_vals = np.zeros(nout, dtype=complex)
    
    for i in range(nout):
        val = complex(0,0)
        for j in range(nin):
            psum = 0
            for k in range(nax):
                psum += in_ax[k][j]*out_ax[k][i]
            cphs = 2.*np.pi*psum
            val += in_vals[j]*complex(np.cos(cphs),np.sin(cphs))
        out_vals[i] = val
#        progress(20, i+1., nk)
    
    return out_vals/len(in_vals)