import numpy as np
import scipy.io.wavfile as wav
from numpy.lib.stride_tricks import as_strided
import tifffile as tiff
from PIL import Image
import pyora as ora


NORM_CONST = 8.0 # Normalization constant for symlog
DEFAULT_SYMLOG_SHIFT = 8.0 # Default shift parameter for symlog

INPUT_RANGE = (-NORM_CONST, NORM_CONST)

def norm (x): return x / np.max(np.abs(x))



def symlog_scale(x, shift = 0, range = INPUT_RANGE):
    """
    Symlog function

    x: input signal
    shift: shift parameter
    range: input range
    """



    log = np.log

    shift =np.exp(-shift)
    lg_s = log(shift)

    lg_mag = np.log(np.max(np.abs(range)))

    s_log = lambda x: np.sign(x) * (1 + (lg_mag - log(np.abs(x) + shift)) / (lg_s - lg_mag))


    clamp_factor = s_log(NORM_CONST)

    return s_log(x) / clamp_factor

def inverse_symlog_scale(y, shift = 0, range = INPUT_RANGE):
    """
    Inverse symlog function
    
    y: input signal
    shift: shift parameter
    range: input range
    """

    exp = np.exp
    log = np.log


    lg_mag = log(np.max(np.abs(range)))

    shift = exp(-shift)
    lg_s = log(shift)

    is_log = lambda y: np.sign(y) * (exp((1 - abs(y)) * (lg_s - lg_mag) + lg_mag) - shift)

    s_log = lambda x: 1 + (lg_mag - log(np.abs(x) + shift)) / (lg_s - lg_mag)
    clamp_factor = s_log(NORM_CONST)

    return is_log(y * clamp_factor)




def dctiv(x):
    """
    DCT-IV implementation for a vector of audio samples.

    x: input signal
    symlog_shift: shift parameter for symlog
    """



    # Check if input length is a multiple of 4
    N = x.shape[0]
    if N % 4 != 0:
        raise ValueError("MDCT4 only defined for vectors of length multiple of four.")

    # Define parameters
    M = N // 2
    N4 = N // 4

    # Apply rotation and sign change  
    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]

    # Generate twiddle factors
    t = np.arange(0, N4)
    w = np.exp(-1j * 2 * np.pi * (t + 1./8.) / N)

    # Pre-twiddle stage
    c = np.take(rot, 2 * t) - np.take(rot, N - 2 * t - 1) - 1j * (np.take(rot, M + 2 * t) - np.take(rot, M - 2 * t - 1))

    # Apply FFT and scaling
    c = (2. / np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)

    # Post-twiddle stage
    y = np.zeros(M)
    y[2 * t] = np.real(c[t])
    y[M - 2 * t - 1] = np.imag(c[t])
    
    # Return normalized output


    return -y


def idctiv(x):
    """
    Inverse DCT-IV implementation for a vector of audio samples.

    x: input frequency coefficients
    symlog_shift: shift parameter for symlog
    """


    # Check if input length is even
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError("iMDCT4 only defined for even-length vectors.")
    
    # Define parameters
    M = N // 2
    N2 = N * 2

    # Generate twiddle factors
    t = np.arange(0, M)
    w = np.exp(-1j * 2 * np.pi * (t + 1./8.) / N2)

    # Pre-twiddle stage
    c = np.take(x, 2 * t) + 1j * -np.take(x, N - 2 * t - 1)
    c = 0.5 * w * c

    # Apply FFT and scaling
    c = np.fft.fft(c, M)
    c = ((8 / np.sqrt(N2)) * w) * c

    # Construct the output vector with appropriate rotations
    rot = np.zeros(N2)
    rot[2 * t] = np.real(c[t])
    rot[N + 2 * t] = np.imag(c[t])

    # Invert rotation
    t = np.arange(1, N2, 2)
    rot[t] = -rot[N2 - t - 1]

    # Construct final output
    x = np.zeros(N2)
    t = np.arange(0, 3 * M)
    x[t] = rot[t + M]
    t = np.arange(3 * M, N2)
    x[t] = -rot[t - 3 * M]
 

    return x




def signal_to_blocks(signal2d, window_size=1024, overlap=512):


    if signal2d.ndim == 1:
        # Convert to stereo if mono
        signal2d = signal2d[:, None]
        signal2d = np.concatenate((signal2d, signal2d), axis=1)

    elif signal2d.ndim != 2:
        raise ValueError("Input signal must be a 2D array with shape (num_samples, num_channels)")

    stride = window_size - overlap  # Stride between window starts
    num_windows = 1 + (signal2d.shape[0] - window_size) // stride 
    
    shape = (num_windows, window_size, signal2d.shape[1])  # Shape for each window
    strides = (signal2d.strides[0] * stride, signal2d.strides[0], signal2d.strides[1])
    
    blocks = as_strided(signal2d, shape=shape, strides=strides)
    return blocks


def blocks_to_signal(blocks, window_size=1024, overlap=512):
    if blocks.ndim != 3:
        raise ValueError("Input blocks must be a 3D array with shape (num_windows, window_size, num_channels)")

    stride = window_size - overlap  # Stride between window starts
    num_windows, _, num_channels = blocks.shape

    # Calculate the length of the reconstructed signal
    signal_length = (num_windows - 1) * stride + window_size

    # Initialize an array to hold the reconstructed signal
    signal = np.zeros((signal_length, num_channels))

    # Generate the indices for where each block will be added
    window_positions = stride * np.arange(num_windows)[:, np.newaxis] + np.arange(window_size)

    # Reconstruct each channel using advanced indexing
    for channel in range(num_channels):
        np.add.at(signal[:, channel], window_positions.ravel(), blocks[:, :, channel].ravel())

    return signal


def sin_window_vector(N):
    n  = np.arange(0, N)
    return np.sin((np.pi * n) / N)




# ============================================ Equal Loudness Curve ============================================


# Equal loudness contour parameters ISO 226

# Hz: Frequency in Hertz
# Af: Equal loudness contour adjustment factor
# Lu: Magnitude of the linear transfer function normalised at 1 kHz
# Tf: Exponent of loudness perception
countour_params = {
  'Hz': np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]),
  'Af': np.array([0.532, 0.506, 0.48, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.33, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.25, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301]),
  'Lu': np.array([-31.6, -27.2, -23, -19.1, -15.9, -13, -10.3, -8.1, -6.2, -4.5, -3.1, -2, -1.1, -0.4, 0, 0.3, 0.5, 0, -2.7, -4.1, -1, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1]),
  'Tf': np.array([78.5, 68.7, 59.5, 51.1, 44, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6, -5.4, -1.5, 6, 12.6, 13.9, 12.3])
}



def frequency_bins(rate, n_freq_bins):
    return np.fft.fftfreq(n_freq_bins, 1 / rate)

def equal_loudness_contour(rate, block_size, LN=40, aggression=1.0):
    """
    Calculate the equal loudness contour for a given range of frequencies.
    
    Parameters
    ----------
    frequencies : array
        Array of frequencies in Hz.
    LN : float
        Reference loudness level in phons.

    Returns
    -------
    Lp : array
        Sound pressure level in dB SPL for each frequency.
    """

    frequencies = frequency_bins(rate, block_size)

    # Constants for the calculation
    af = 2e-5  # Threshold of hearing in Pascals

    # Define interpolation functions for the data 
    Af = np.interp(frequencies, countour_params['Hz'], countour_params['Af'])
    Lu = np.interp(frequencies, countour_params['Hz'], countour_params['Lu'])
    T_f = np.interp(frequencies, countour_params['Hz'], countour_params['Tf'])
  


    # Calculate the adjustment factor Af for each frequency
    Af = 4.47e-3 * (10**(0.025 * LN) - 1.15) + \
        (0.4 * 10**((T_f + Lu)/10 - 9))**(af)

    # Calculate the SPL Lp for each frequency
    Lp = 10 / af * np.log10(Af) - Lu + 94  # The formula to compute SPL Lp


    # Apply the aggression factor
    Lp = Lp ** aggression

    # For our purposes, we normalise the Lp to maximum of 1.0, ignoring a 0Hz component
    Lp = 1 / Lp
    Lp = Lp / np.max(Lp)

    return Lp


# ============================================ MDCT ============================================





def mdct(blocks, rate, symlog_shift=DEFAULT_SYMLOG_SHIFT):
    num_blocks, window_size, num_channels = blocks.shape

    # Create the windowing function and reshape for broadcasting
    window = sin_window_vector(window_size)[:, np.newaxis]  # Reshape to (window_size, 1)

    # Apply the window to each block for each channel
    windowed_blocks = blocks * window

    # Apply MDCT to each block
    raw_spectrum = np.apply_along_axis(dctiv, 1, windowed_blocks)

    # Normalise the spectrum
    spectrum = symlog_scale(raw_spectrum, shift=symlog_shift)

    # Apply equal loudness curve over the log-scaled spectrum along the frequency axis
    eq_loudness_vector = equal_loudness_contour(rate, window_size //2)[:, np.newaxis]
    spectrum = spectrum * eq_loudness_vector

    return spectrum


def i_mdct(spectrum, rate, symlog_shift=DEFAULT_SYMLOG_SHIFT):
    num_blocks, n_freq_bins, num_channels = spectrum.shape


    # Apply inverse equal loudness curve along the frequency axis
    eq_loudness_vector = equal_loudness_contour(rate, n_freq_bins)[:, np.newaxis]
    spectrum = spectrum / eq_loudness_vector

    # Apply inverse normalisation
    raw_spectrum = inverse_symlog_scale(spectrum, shift=symlog_shift)
    
    # Apply iMDCT to each block
    blocks = np.apply_along_axis(idctiv, 1, raw_spectrum)

    # Create the windowing function and reshape for broadcasting
    window = sin_window_vector(n_freq_bins * 2)[:, np.newaxis] 
    blocks = blocks * window

    return blocks


# ============================================ Exporting ============================================


def orient_spectrum(spectrum):
    """
    Orient the spectrum for export to image formats

    spectrum: numpy array of shape (n_samples, n_channels, n_bins)
    """



    transposed = np.transpose(spectrum, (1, 0, 2))
    flipped = np.flip(transposed, axis=0)

    left, right = flipped[:, :, 0], flipped[:, :, 1]

    return left, right

def split_signed_values(array):
    where_positives = array > 0
    positive_values = np.zeros_like(array)
    negative_values = np.zeros_like(array)

    positive_values[where_positives] = array[where_positives]
    negative_values[~where_positives] = -array[~where_positives]

    return positive_values, negative_values

def save_image(array, filename, dtype=(np.float32, 'float32')):
    nptype, tifftype = dtype
    tiff.imwrite(filename, array.astype(nptype), dtype=tifftype)

def export_spectrum(spectrum, filename, process_func):
    left, right = orient_spectrum(spectrum)
    left_processed, right_processed = process_func(left), process_func(right)
    save_image(left_processed, f'{filename}_l.tiff')
    save_image(right_processed, f'{filename}_r.tiff')

def export_absolute(spectrum, filename):
    export_spectrum(spectrum, filename, np.abs)

def export_split_signs(spectrum, filename):
    left, right = orient_spectrum(spectrum)
    export_absolute(spectrum, filename)
    l_pos, _ = split_signed_values(left)
    r_pos, _ = split_signed_values(right)

    ## threshold to either 0 or 255
    l_pos = np.where(l_pos > 0, 1, 0)
    r_pos = np.where(r_pos > 0, 1, 0)

    ## experiment : no signs!

    # l_pos = np.ones_like(l_pos)
    # r_pos = np.ones_like(r_pos)

    save_image(l_pos, f'{filename}_signs_l.tiff')
    save_image(r_pos, f'{filename}_signs_r.tiff')



# ============================================ Reimporting ============================================
    

def reorient_spectrum(left, right):
    
    left = left.T
    right = right.T

    spectrum = np.stack([left, right], axis=2)

    unflipped = np.flip(spectrum, axis=1)

    return unflipped


def sanitize_import(img):
    ## detect if image is int or float
    if img.dtype == np.uint8:
        print ("Reimport, image is uint8! Fidelity may be impacted.")

        ## Convert to float32
        img = img / 255

    if img.ndim == 3:
        ## Discard additional channels
        print ("Reimport, image dimensions:" + str(img.shape))
        img = img[:, :, 0]
    
    return np.clip(img, 0, 1)

def reimport_split_signs(filename):
    left_img = tiff.imread(f'{filename}_l.tiff', dtype=np.float32)
    right_img = tiff.imread(f'{filename}_r.tiff', dtype=np.float32)

    sign_l = tiff.imread(f'{filename}_signs_l.tiff', dtype=np.float32)
    sign_r = tiff.imread(f'{filename}_signs_r.tiff', dtype=np.float32)

    left = sanitize_import(left_img)
    right = sanitize_import(right_img)

    sign_l = sanitize_import(sign_l)
    sign_r = sanitize_import(sign_r)

    sign_l = np.where(sign_l > 0.5, 1, -1)
    sign_r = np.where(sign_r < 0.5, 1, -1)

    left = left * sign_l
    right = right * sign_r

    return reorient_spectrum(left, right)

## ================================= ORA OpenRaster Project Format =======================================


## ============================================ Exporting ORA ============================================


def f32_to_u16(array):
    UINT_MAX = np.iinfo(np.uint16).max

    return (array * UINT_MAX).astype(np.uint16)

def export_ora(spectrum, filename):
    left, right = orient_spectrum(spectrum)
    W, H, _ = spectrum.shape

    project = ora.Project.new(W, H)

    def add_channel_layers(group, channel_data, group_name, abbrev):
        neg, pos = split_signed_values(channel_data)

        neg = f32_to_u16(neg)
        pos = f32_to_u16(pos)

        print (neg.min(), neg.max())
        print (pos.min(), pos.max())

        layer = project.add_layer(neg, group_name + f'/{abbrev}-negative')
        layer.composite_op = 'svg:src-over'

        ### set image as 16bit grayscale png
        n_img = Image.fromarray(neg, )
        layer.set_image_data(n_img)


        layer = project.add_layer(pos, group_name + f'/{abbrev}-positive')
        layer.composite_op = 'svg:plus'
        p_img = Image.fromarray(pos, )
        layer.set_image_data(p_img) 



    l_group = project.add_group('left-stereo')
    add_channel_layers(l_group, left, 'left-stereo', 'l')
    
    r_group = project.add_group('right-stereo')
    add_channel_layers(r_group, right, 'right-stereo', 'r')
    
    project.save(filename + '.ora')



## ============================================ Reimporting ORA ============================================
    
def u16_to_f32(array):
    UINT_MAX = np.iinfo(np.uint16).max

    return array.astype(np.float32) / UINT_MAX

def reimport_ora(filename):
    project = ora.Project.load(filename + '.ora')

    l_neg = project.get_by_path('left-stereo/l-negative').get_image_data()
    l_pos = project.get_by_path('left-stereo/l-positive').get_image_data()


    r_neg = project.get_by_path('right-stereo/r-negative').get_image_data()
    r_pos = project.get_by_path('right-stereo/r-positive').get_image_data()


    l_neg = np.array(l_neg)
    l_pos = np.array(l_pos)

    r_neg = np.array(r_neg)
    r_pos = np.array(r_pos)

    left = u16_to_f32(l_pos) - u16_to_f32(l_neg)
    right = u16_to_f32(r_pos) - u16_to_f32(r_neg)

    return reorient_spectrum(left, right)