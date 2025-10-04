class ExoplanetAnalyzer:
    def __init__(self, model_pipeline):
        self.pipeline = model_pipeline
    
    def analyze_tic_target(self, tic_id):
        """Analyze a real TESS target"""
        print(f"Analyzing TIC {tic_id}...")
        
        try:
            # Search for TESS data
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            
            if len(search_result) > 0:
                # Download and process light curve
                lc = search_result[0].download()
                lc_clean = lc.remove_nans().remove_outliers()
                
                time = lc_clean.time.value - lc_clean.time.value[0]  # Normalize time
                flux = lc_clean.flux.value
                flux_normalized = flux / np.median(flux)  # Normalize flux
                
                # Detrend
                flux_detrended = self.pipeline.data_processor.advanced_detrending(flux_normalized)
                
                # Extract features to get period estimate
                features = self.pipeline.data_processor.extract_features(time, flux_detrended)
                period_guess = features.get('dominant_period', 10.0)
                
                # Create model inputs
                phase_image = self.pipeline.data_processor.create_phase_folded_image(
                    time, flux_detrended, period_guess
                )
                temporal_input = flux_detrended[:1000].reshape(1, 1000, 1)
                temporal_input = (temporal_input - np.mean(temporal_input)) / (np.std(temporal_input) + 1e-8)
                
                vision_input = phase_image.reshape(1, 128, 128, 3)
                
                # Make prediction
                predictions = self.pipeline.model_wrapper.model.predict(
                    {'image_input': vision_input, 'time_input': temporal_input},
                    verbose=0
                )
                
                detection_prob = float(predictions[0][0][0])
                pred_period = float(predictions[1][0][0])
                pred_depth = float(predictions[2][0][0])
                
                result = {
                    'tic_id': tic_id,
                    'detection_probability': detection_prob,
                    'predicted_period': max(0.1, pred_period),  # Ensure positive
                    'predicted_depth': max(0.001, pred_depth),   # Ensure positive
                    'features': features,
                    'has_exoplanet': detection_prob > 0.7
                }
                
                return result
            else:
                print(f"No data found for TIC {tic_id}")
                return None
                
        except Exception as e:
            print(f"Error analyzing TIC {tic_id}: {str(e)}")
            return None
    
    def plot_light_curve(self, time, flux, title="Light Curve"):
        """Plot light curve"""
        plt.figure(figsize=(12, 4))
        plt.plot(time, flux, 'k-', alpha=0.7, linewidth=0.8)
        plt.title(title)
        plt.xlabel('Time (days)')
        plt.ylabel('Normalized Flux')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_phase_folded(self, time, flux, period, title="Phase-Folded Light Curve"):
        """Plot phase-folded light curve"""
        phase = (time % period) / period
        sort_idx = np.argsort(phase)
        
        plt.figure(figsize=(10, 5))
        plt.plot(phase[sort_idx], flux[sort_idx], 'bo', alpha=0.5, markersize=2)
        plt.title(f"{title} (Period: {period:.2f} days)")
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        plt.grid(True, alpha=0.3)
        plt.show()
