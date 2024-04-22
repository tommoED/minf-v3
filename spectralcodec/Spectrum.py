from spectralcodec import codec
from scipy.io import wavfile as wav
from numpy import float32

class Spectrum:
  def __init__(self, spectrum_data, rate,  blocksize = 1024, overlap = 512):
    self.blocksize = 1024
    self.overlap = 512
    self.rate = rate
    self.data = spectrum_data



  def from_numpy(data, rate, blocksize = 1024, overlap = 512):
    """
      Create a Spectrum object from a numpy signal array

      data: numpy array of signal (n_samples, n_channels)

      rate: sample rate
    """

    return Spectrum(data, rate, blocksize, overlap)
  
  def from_wav(wavfile, blocksize = 1024, overlap = 512, show = False):
    rate, sample = wav.read(wavfile)

    sample = codec.norm(sample)


    blocks = codec.signal_to_blocks(sample, blocksize, overlap)

    print (blocks.shape)

    spectra = codec.mdct(blocks, rate)

    print (spectra.shape)

    return Spectrum ( spectra, rate, blocksize, overlap )
  
  def get_left_right(self):
    return self.data[:, :, 0], self.data[:, :, 1]
  

  def from_tiffs(sourcename, rate, func = codec.reimport_split_signs):
    ## Load tiffs

    return Spectrum(func(sourcename), rate)
  


  def from_ora(sourcename, rate, func = codec.reimport_ora):
    ## Load tiffs

    if sourcename.endswith('.ora'):
      sourcename = sourcename.replace('.ora', '')

    return Spectrum(func(sourcename), rate)


  def export(self, filename):
    """"
      Reconstruct and export signal as Wav file
    """

    if not filename.endswith('.wav'):
      filename = filename + '.wav'

    blocks = codec.i_mdct(self.data, self.rate)
    signal = codec.blocks_to_signal(blocks)

    print (signal.min(), signal.max())

    wav.write(filename, self.rate, signal.astype(float32))

  def save(self, filename, exportfunc = codec.export_split_signs):

    if filename.endswith('.tiff'):
      self.save(filename.replace('.tiff', ''))
    elif filename.endswith('.ora'):
      self.save(filename.replace('.ora', ''), codec.export_ora)
    else:
      exportfunc(self.data, filename)