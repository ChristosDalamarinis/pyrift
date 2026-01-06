"""
EEG Data Import Module
======================
This module provides functions for importing EEG data from BDF and EDF file formats.

Supported formats:
- EDF (European Data Format)
- EDF+ (European Data Format Plus)
- BDF (BioSemi Data Format - 24-bit version of EDF)
- BDF+ (BioSemi Data Format Plus)
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
import warnings

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not installed. Install with: pip install mne")

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False
    warnings.warn("pyedflib not installed. Install with: pip install pyedflib")


class EEGData:
    """
    Container class for EEG data and metadata.
    
    Attributes
    ----------
    data : np.ndarray
        EEG signal data with shape (n_channels, n_samples)
    channel_names : list
        Names of EEG channels
    sampling_rate : float
        Sampling frequency in Hz
    n_channels : int
        Number of channels
    n_samples : int
        Number of samples per channel
    metadata : dict
        Additional file metadata
    """
    
    def __init__(self, data: np.ndarray, channel_names: List[str], 
                 sampling_rate: float, metadata: Dict):
        self.data = data
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.n_channels = data.shape[0]
        self.n_samples = data.shape[1] if data.ndim > 1 else data.shape[0]
        self.metadata = metadata
    
    def __repr__(self):
        return (f"EEGData(n_channels={self.n_channels}, "
                f"n_samples={self.n_samples}, "
                f"sampling_rate={self.sampling_rate} Hz)")


def read_eeg(filepath: Union[str, Path], 
             backend: str = 'mne',
             preload: bool = True,
             **kwargs) -> EEGData:
    """
    Read EEG data from BDF or EDF files.
    
    This is the main function users should call to import EEG data.
    Automatically detects file format and uses appropriate reader.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .bdf or .edf file
    backend : str, optional
        Backend library to use ('mne' or 'pyedflib'). Default is 'mne'.
    preload : bool, optional
        If True, load data into memory immediately. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to the backend reader
    
    Returns
    -------
    EEGData
        Object containing the EEG data and metadata
    
    Examples
    --------
    >>> # Read an EDF file
    >>> eeg_data = read_eeg('path/to/file.edf')
    >>> print(eeg_data)
    >>> print(f"Shape: {eeg_data.data.shape}")
    >>> print(f"Channels: {eeg_data.channel_names}")
    
    >>> # Read a BDF file with pyedflib backend
    >>> eeg_data = read_eeg('path/to/file.bdf', backend='pyedflib')
    
    Notes
    -----
    - EDF files are 16-bit format
    - BDF files are 24-bit format (higher resolution)
    - Both EDF+ and BDF+ (with annotations) are supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check file extension
    extension = filepath.suffix.lower()
    if extension not in ['.edf', '.bdf']:
        raise ValueError(f"Unsupported file format: {extension}. "
                        f"Only .edf and .bdf files are supported.")
    
    # Route to appropriate backend
    if backend == 'mne':
        return _read_with_mne(filepath, extension, preload, **kwargs)
    elif backend == 'pyedflib':
        return _read_with_pyedflib(filepath, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. "
                        f"Choose 'mne' or 'pyedflib'.")


def _read_with_mne(filepath: Path, extension: str, 
                   preload: bool, **kwargs) -> EEGData:
    """Read EEG data using MNE-Python backend."""
    if not MNE_AVAILABLE:
        raise ImportError("MNE-Python is required for this backend. "
                         "Install with: pip install mne")
    
    # Read file based on extension
    if extension == '.edf':
        raw = mne.io.read_raw_edf(filepath, preload=preload, verbose=False, **kwargs)
    elif extension == '.bdf':
        raw = mne.io.read_raw_bdf(filepath, preload=preload, verbose=False, **kwargs)
    
    # Extract data and metadata
    data = raw.get_data()  # shape: (n_channels, n_samples)
    channel_names = raw.ch_names
    sampling_rate = raw.info['sfreq']
    
    # Collect metadata
    metadata = {
        'file_path': str(filepath),
        'file_format': extension[1:].upper(),
        'recording_date': raw.info.get('meas_date', None),
        'n_times': raw.n_times,
        'duration_seconds': raw.times[-1] if len(raw.times) > 0 else 0,
        'highpass': raw.info.get('highpass', None),
        'lowpass': raw.info.get('lowpass', None),
    }
    
    return EEGData(data, channel_names, sampling_rate, metadata)


def _read_with_pyedflib(filepath: Path, **kwargs) -> EEGData:
    """Read EEG data using pyedflib backend."""
    if not PYEDFLIB_AVAILABLE:
        raise ImportError("pyedflib is required for this backend. "
                         "Install with: pip install pyedflib")
    
    # Open file
    f = pyedflib.EdfReader(str(filepath))
    
    try:
        # Get basic information
        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        
        # Read all signals
        data = np.zeros((n_channels, f.getNSamples()[0]))
        sampling_rates = []
        
        for i in range(n_channels):
            data[i, :] = f.readSignal(i)
            sampling_rates.append(f.getSampleFrequency(i))
        
        # Check if all channels have same sampling rate
        if len(set(sampling_rates)) > 1:
            warnings.warn("Channels have different sampling rates. "
                         f"Using first channel's rate: {sampling_rates[0]} Hz")
        sampling_rate = sampling_rates[0]
        
        # Collect metadata
        metadata = {
            'file_path': str(filepath),
            'file_format': filepath.suffix[1:].upper(),
            'recording_date': f.getStartdatetime(),
            'duration_seconds': f.getFileDuration(),
            'patient_name': f.getPatientName(),
            'patient_code': f.getPatientCode(),
            'technician': f.getTechnician(),
            'equipment': f.getEquipment(),
            'physical_dimensions': [f.getPhysicalDimension(i) 
                                   for i in range(n_channels)],
        }
        
    finally:
        f.close()
    
    return EEGData(data, channel_names, sampling_rate, metadata)


def get_channel_info(filepath: Union[str, Path], 
                     backend: str = 'mne') -> Dict:
    """
    Get channel information without loading the full data.
    
    Useful for inspecting large files before loading.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .bdf or .edf file
    backend : str, optional
        Backend library to use ('mne' or 'pyedflib'). Default is 'mne'.
    
    Returns
    -------
    dict
        Dictionary containing channel names, sampling rates, and units
    
    Examples
    --------
    >>> info = get_channel_info('path/to/file.edf')
    >>> print(f"Channels: {info['channel_names']}")
    >>> print(f"Sampling rate: {info['sampling_rate']} Hz")
    """
    filepath = Path(filepath)
    
    if backend == 'pyedflib':
        if not PYEDFLIB_AVAILABLE:
            raise ImportError("pyedflib is required. Install with: pip install pyedflib")
        
        f = pyedflib.EdfReader(str(filepath))
        try:
            info = {
                'channel_names': f.getSignalLabels(),
                'n_channels': f.signals_in_file,
                'sampling_rates': [f.getSampleFrequency(i) 
                                  for i in range(f.signals_in_file)],
                'physical_dimensions': [f.getPhysicalDimension(i) 
                                       for i in range(f.signals_in_file)],
                'duration_seconds': f.getFileDuration(),
            }
        finally:
            f.close()
        
        return info
    
    elif backend == 'mne':
        if not MNE_AVAILABLE:
            raise ImportError("MNE-Python is required. Install with: pip install mne")
        
        extension = filepath.suffix.lower()
        if extension == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
        elif extension == '.bdf':
            raw = mne.io.read_raw_bdf(filepath, preload=False, verbose=False)
        
        return {
            'channel_names': raw.ch_names,
            'n_channels': len(raw.ch_names),
            'sampling_rate': raw.info['sfreq'],
            'duration_seconds': raw.times[-1] if len(raw.times) > 0 else 0,
        }
