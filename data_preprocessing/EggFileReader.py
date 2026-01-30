"""
Authors: F. Thomas, Mingyu (Charles) Li
Date: February 22, 2021
"""

__all__ = ['LocustP3File']

import numpy as np
import h5py
from scipy.fft import fft, fftshift, fftfreq

def _apply_DFT(data, dft_window):

    """
    Helper function that takes FFT of data on the last axis
    """

    mean = np.mean(data, axis=-1, keepdims=True)
    normalization = np.sqrt(1 / dft_window)
    # By default uses the last axis
    data_freq = fft(data - mean)
    data_freq = fftshift(data_freq, axes=-1)
    data_freq = data_freq * normalization
    return data_freq

def _to_complex(data):
    
    """
    Convert to complex
    """
    
    return data[::2] + 1j * data[1::2]

class EggFileReader:
    _int_max = {8: 255, 16: 65535}
    def __init__(self, file_name):
        self._input_file = h5py.File(file_name, 'r')
        self._get_attributes()
    def _get_attributes(self):
        ### Attrs see https://monarch.readthedocs.io/en/latest/EggStandard.v3.2.0.html#file-structure

        # A dict for file attrs
        self._file_attrs = self._input_file.attrs

        # A dict of dict for stream attrs (includes all streams)
        self._streams_attrs = self._get_streams_attrs()

        # A dict of dict for channel attrs (includes all channels)
        self._channels_attrs = self._get_channels_attrs()

        # Some useful attributes obtained from default file/stream0
        self._n_channels = self._file_attrs['n_channels']
        self._n_streams = self._file_attrs['n_streams']

        # Channel -> Stream mapping
        self._channel_streams = self._file_attrs['channel_streams']

    def _get_streams_attrs(self):
    
        """
        Get all streams attrs
        """

        h5file = self._input_file
        streams_attr = {}

        for (key, item) in h5file['streams'].items():
            att = item.attrs
            streams_attr[key] = att

        return streams_attr

    def _get_channels_attrs(self):

        """
        Get all channels attrs
        """

        h5file = self._input_file
        channels_attr = {}

        for (key, item) in h5file['channels'].items():
            att = item.attrs
            channels_attr[key] = att

        return channels_attr

    def _convert_to_voltage(self, data: np.ndarray, channel: int):

        """
        Convert complex data from digitizer unit to volts, specific by channel
        """

        if not np.issubdtype(data.dtype, np.complexfloating):
            raise ValueError("Data is not complex.")

        attr = self.get_channel_attrs(channel)
        bit_depth = attr['bit_depth']
        voltage_range = attr['voltage_range']
        voltage_offset = attr['voltage_offset']
        result = data / self._int_max[
            bit_depth] * voltage_range + voltage_offset + 1j * voltage_offset
        return result

    def _read_ts(self, s, attr):

        """
        Helper method that returns a channel list and raw time series in np.array of shape:
        (n_acquisitions, n_records, n_channels, record_size)
        """

        n_acq = attr['n_acquisitions']
        channels = attr['channels']
        ch_format = attr['channel_format']
        data = []

        for i in range(n_acq):
            acq = s['acquisitions']['%s' % i]
            n_records = acq.attrs['n_records']
            temp = []
            for j in range(n_records):
                temp_data = _to_complex(acq[j])
                if ch_format == 1:
                    temp.append(temp_data.reshape((len(channels), -1)))
                elif ch_format == 0:
                    n_ch = len(channels)
                    for k in range(n_ch):
                        temp_data_copy = []
                        temp_data_copy.append(temp_data[k::n_ch])
                    temp.append(temp_data_copy)
                else:
                    raise RuntimeError("Invalid channel format")
            data.append(temp)

        data = np.asarray_chkfinite(data)

        return channels, data

    def load_ts_stream(self, stream: int = 0):

        """
        Load the time series in a stream by stream basis. Only a single stream is allowed.
        The dataset is structured as (dict of arrays):
        {Channel #: [Acquisition0: [Record0, Record1, ...], Acquisition1: [...], ...], Channel #: [...], ...}
        """

        try:
            s = self._input_file['streams']["stream{}".format(stream)]
            attr = self.get_stream_attrs(stream)
        except KeyError as kerr:
            print(kerr)
        except Exception as e:
            print(e)
            raise

        channels, data = self._read_ts(s, attr)
        result = {}

        for ch in channels:
            data_ts = self._convert_to_voltage(data[:, :, ch, :], ch)
            result[ch] = np.ascontiguousarray(data_ts)

        return result

    def quick_load_ts_stream(self, stream: int = 0):

        """
        Load the time series in default format (numpy array) with shape:
        (n_acquisitions, n_channels, n_records * record_size)
        """

        try:
            s = self._input_file['streams']["stream{}".format(stream)]
            attr = self.get_stream_attrs(stream)
        except KeyError as kerr:
            print(kerr)
        except Exception as e:
            print(e)
            raise

        channels, data = self._read_ts(s, attr)

        # transpose axis 1 and 2 and concatenate all records
        data = np.swapaxes(data, 1, 2)
        data = np.reshape(data, data.shape[:2] + (-1,))

        for ch in channels:
            data[:, ch, :] = self._convert_to_voltage(data[:, ch, :], ch)

        data = np.ascontiguousarray(data)
        return data

    def quick_load_IQ_stream(self, stream: int = 0):

        """
        Load the I and Q time series for a given stream (default 0).
        Returns a tuple (I_data, Q_data), each a numpy array of shape 
        (n_acquisitions, n_channels, n_records * record_size), containing 
        the I (real) and Q (imaginary) components of the voltage time series.
        """

        try:
            s = self._input_file['streams'][f"stream{stream}"]
            attr = self.get_stream_attrs(stream)
        except KeyError as kerr:
            print(kerr)
            return None, None
        except Exception as e:
            print(e)
            raise

        # Read raw time series data (complex) and channel list
        channels, data = self._read_ts(s, attr)
        
        # Reshape data to shape: (n_acquisitions, n_channels, n_records * record_size)
        data = np.swapaxes(data, 1, 2)                        # swap axes to (acq, channel, record, size)
        data = np.reshape(data, data.shape[:2] + (-1,))       # merge record and size into one axis

        # Convert raw ADC units to voltage for each channel
        for ch in channels:
            data[:, ch, :] = self._convert_to_voltage(data[:, ch, :], ch)

        # Separate the real and imaginary parts into I and Q components
        I_data = np.ascontiguousarray(data.real)
        Q_data = np.ascontiguousarray(data.imag)
        return I_data, Q_data

    def load_fft_stream(self, dft_window: int = 4096, stream: int = 0):

        """
        Load the FFT of the timeseries.
        dft_window: sample size for each DFT slice, defaults to 4096.
        Structure is the same as that of the timeseries.
        """

        ts = self.load_ts_stream(stream=stream)
        result = {}
        attr = self.get_stream_attrs(stream)

        # Rate from MHz -> Hz
        acq_rate = attr['acquisition_rate'] * 1e6
        channels = attr['channels']

        for ch in channels:
            ts_ch = ts[ch]
            n_slices = int(ts_ch.shape[-1] / dft_window)
            ts_sliced = ts_ch[:, :, :n_slices * dft_window].reshape(
                (ts_ch.shape[0], ts_ch.shape[1], n_slices, -1))

            # Apply DFT to sliced ts and return DFT in the original shape
            # freq is a single array since acq_rate is the same for all data in the stream
            data_freq = _apply_DFT(ts_sliced, dft_window)
            result[ch] = np.ascontiguousarray(data_freq)

        frequency = fftshift(fftfreq(dft_window, d=1 / acq_rate))
        return frequency, result

    def quick_load_fft_stream(self, dft_window: int = 4096, stream: int = 0):

        """
        Load FFT in default format:
        Note n_slices = int(n_records * record_size / dft_window)
        (n_acquisitions, n_channels, n_slices, FFT for each slice)
        """

        ts = self.quick_load_ts_stream(stream)
        attr = self.get_stream_attrs(stream)

        # Rate from MHz -> Hz
        acq_rate = attr['acquisition_rate'] * 1e6
        n_slices = int(ts.shape[-1] / dft_window)

        # Discard extra data points
        ts_sliced = ts[:, :, :n_slices * dft_window].reshape(
            (ts.shape[0], ts.shape[1], n_slices, -1))
        data_freq = _apply_DFT(ts_sliced, dft_window)
        data_freq = np.ascontiguousarray(data_freq)
        frequency = fftfreq(dft_window, d=1 / acq_rate)
        return frequency, data_freq

    @property
    def n_channels(self):
        return self._n_channels

    @property

    def n_streams(self):
        return self._n_streams

    @property
    def channel_streams(self):
        return self._channel_streams

    def print_file_attrs(self):

        """
        Print egg file attributes
        """

        for (k, it) in self._file_attrs.items():
            print("key: " + k)
            print("value: " + str(it))

    def get_stream_attrs(self, stream: int = 0):

        """
        Get the attributes of a specific stream. Only a single stream number is allowed. Defaults to stream 0.
        """

        try:
            attrs = self._streams_attrs["stream{}".format(stream)]
        except KeyError as kerr:
            print(kerr)
        except Exception as e:
            print(e)
            raise
        return attrs

    def print_stream_attrs(self, streams=[0]):

        """
        Print the attributes of streams. Defaults to only stream 0.
        """

        for s in streams:
            print("Getting attributes for stream {}".format(s))
            attrs = self.get_stream_attrs(s)
            
            for (k, it) in attrs.items():
                print("key: " + k)
                print("value: " + str(it))

    def get_channel_attrs(self, channel: int = 0):

        """
        Get the attributes of a specific channel. Only a single channel number is allowed. Defaults to channel 0.
        """

        try:
            attrs = self._channels_attrs["channel{}".format(channel)]
        except KeyError as kerr:
            print(kerr)
        except Exception as e:
            print(e)
            raise
        return attrs

    def print_channel_attrs(self, channels=[0]):

        """
        Print the attributes of channels. Defaults to only channel 0.
        """

        for c in channels:
            print("Getting attributes for channel {}".format(c))
            attrs = self.get_channel_attrs(c)
            for (k, it) in attrs.items():
                print("key: " + k)
                print("value: " + str(it))
