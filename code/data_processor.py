"""
Advanced data processing and feature extraction for exoplanet light curves
"""

import numpy as np
from scipy import signal, stats
import lightkurve as lk

class AdvancedDataProcessor:
    """Advanced processing pipeline for TESS light curves"""
    
    def __init__(self):
        self.optimized_params = {}
    
    def download_tess_data(self, tic_id, sector=None):
        """Download TESS light curve data"""
        try:
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            if len(search_result) > 0:
                lc = search_result[0].download()
                return lc
            else:
                print(f"No TESS data found for TIC {tic_id}")
                return None
        except Exception as e:
            print(f"Error downloading data for TIC {tic_id}: {e}")
            return None
    
    def advanced_detrending(self, flux, method='savitzky_golay'):
        """Advanced detrending algorithms"""
        if method == 'savitzky_golay':
            # Savitzky-Golay filter for smooth detrending
            window_size = min(101, len(flux) // 10 * 2 + 1)  # Ensure odd
            if window_size < 5:
                window_size = 5
            return signal.savgol_filter(flux, window_size, 2)
        
        elif method == 'median_filter':
            # Median filter for robust detrending
            return signal.medfilt(flux, kernel_size=51)
        
        else:
            return flux
    
    def create_phase_folded_image(self, time, flux, period, img_size=128):
        """Create 2D image from phase-folded light curve"""
        if period <= 0:
            period = 10.0  # Default period
        
        # Phase folding
        phase = (time % period) / period
        sort_idx = np.argsort(phase)
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            phase[sort_idx], flux[sort_idx], 
            bins=[img_size, img_size]
        )
        
        # Normalize and create RGB image
        H_norm = H / np.max(H)
        image = np.stack([H_norm.T] * 3, axis=-1)
        
        return image
    
    def extract_advanced_features(self, time, flux):
        """Extract comprehensive feature set"""
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(flux)
        features['std'] = np.std(flux)
        features['skew'] = stats.skew(flux)
        features['kurtosis'] = stats.kurtosis(flux)
        
        # Signal quality metrics
        features['snr'] = np.mean(flux) / np.std(flux)
        
        return features

# Example usage
if __name__ == "__main__":
    processor = AdvancedDataProcessor()
    print("Data processor initialized successfully!")
