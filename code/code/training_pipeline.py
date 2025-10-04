class ExoplanetTrainingPipeline:
    def __init__(self):
        self.model_wrapper = ExoHunterVision()
        self.data_processor = AdvancedDataProcessor()
        self.model_wrapper.compile_model()
    
    def generate_synthetic_light_curve(self, has_planet=True, duration=80, n_points=2000):
        """Generate realistic synthetic light curve"""
        time = np.linspace(0, duration, n_points)
        
        # Stellar variability (realistic patterns)
        stellar_var = (0.01 * np.sin(2*np.pi*time/12.5) + 
                      0.005 * np.sin(2*np.pi*time/3.2) +
                      0.003 * np.cos(2*np.pi*time/25.0))
        
        flux = 1.0 + stellar_var
        
        if has_planet:
            # Add planetary transit
            period = np.random.uniform(1.5, 30.0)
            depth = np.random.uniform(0.002, 0.015)
            transit_duration = 0.1 * period  # 10% of period
            
            # Add multiple transits
            transit_times = np.arange(0, duration, period)
            for transit_center in transit_times:
                if transit_center < duration:
                    in_transit = (time > transit_center - transit_duration/2) & (time < transit_center + transit_duration/2)
                    # Realistic transit shape
                    if np.sum(in_transit) > 0:
                        transit_shape = depth * np.ones(np.sum(in_transit))
                        flux[in_transit] -= transit_shape
        
        # Add realistic noise
        flux += np.random.normal(0, 0.003, size=len(time))
        
        return time, flux, period if has_planet else 0.0, depth if has_planet else 0.0
    
    def generate_dataset(self, n_samples=2000):
        """Generate training dataset"""
        print(f"Generating {n_samples} synthetic light curves...")
        
        vision_data = []
        temporal_data = []
        detection_labels = []
        period_labels = []
        depth_labels = []
        
        for i in range(n_samples):
            has_planet = np.random.random() > 0.3  # 70% have planets
            
            time, flux, period, depth = self.generate_synthetic_light_curve(has_planet)
            
            # Detrend the light curve
            flux_detrended = self.data_processor.advanced_detrending(flux)
            
            # Create phase-folded image
            phase_image = self.data_processor.create_phase_folded_image(time, flux_detrended, period)
            
            # Prepare temporal data (first 1000 points, normalized)
            temporal_seq = flux_detrended[:1000].reshape(1000, 1)
            temporal_seq = (temporal_seq - np.mean(temporal_seq)) / (np.std(temporal_seq) + 1e-8)
            
            vision_data.append(phase_image)
            temporal_data.append(temporal_seq)
            detection_labels.append(1.0 if has_planet else 0.0)
            period_labels.append(period)
            depth_labels.append(depth)
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        # Convert to numpy arrays
        vision_data = np.array(vision_data)
        temporal_data = np.array(temporal_data)
        detection_labels = np.array(detection_labels)
        period_labels = np.array(period_labels)
        depth_labels = np.array(depth_labels)
        
        return vision_data, temporal_data, detection_labels, period_labels, depth_labels
    
    def train_model(self, n_samples=2000, epochs=30, batch_size=32):
        """Train the complete model"""
        print("Starting model training...")
        
        # Generate dataset
        vision_data, temporal_data, det_labels, period_labels, depth_labels = self.generate_dataset(n_samples)
        
        # Split into train/validation
        split_idx = int(0.8 * n_samples)
        
        train_data = {
            'image_input': vision_data[:split_idx],
            'time_input': temporal_data[:split_idx]
        }
        
        train_labels = {
            'detection': det_labels[:split_idx],
            'period': period_labels[:split_idx], 
            'depth': depth_labels[:split_idx]
        }
        
        val_data = {
            'image_input': vision_data[split_idx:],
            'time_input': temporal_data[split_idx:]
        }
        
        val_labels = {
            'detection': det_labels[split_idx:],
            'period': period_labels[split_idx:],
            'depth': depth_labels[split_idx:]
        }
        
        # Train the model
        history = self.model_wrapper.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        return history
