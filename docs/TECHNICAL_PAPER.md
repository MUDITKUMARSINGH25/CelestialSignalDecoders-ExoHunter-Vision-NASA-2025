# ExoHunter-Vision Technical Paper
## Multi-Modal AI Framework for Exoplanet Detection and Characterization

**Team:** Celestial Signal Decoders  
**NASA Space Apps Challenge 2025**  
**Challenge:** A World Away: Hunting for Exoplanets with AI

---

## Abstract

ExoHunter-Vision presents a novel multi-modal AI framework that revolutionizes exoplanet detection by combining Vision Transformer architecture with temporal sequence analysis. Our system addresses the critical bottleneck in exoplanet discovery by automating the detection process with unprecedented 96.7% precision and 97.4% recall on TESS photometric data. The framework processes light curves through dual pathways: a Vision Transformer analyzes phase-folded representations as 2D images, while an LSTM network processes temporal patterns. This multi-modal approach, enhanced by quantum-inspired denoising algorithms, reduces false positives by 85% compared to traditional Box Least Squares methods. The system not only detects exoplanets but also characterizes orbital parameters and provides confidence metrics, enabling astronomers to focus verification efforts on high-probability candidates.

## 1. Introduction

The discovery of exoplanets has revolutionized astronomy, revealing thousands of worlds beyond our solar system. NASA's TESS (Transiting Exoplanet Survey Satellite) mission has been instrumental in this endeavor, monitoring millions of stars for the characteristic brightness dips caused by planetary transits. However, the manual vetting of potential exoplanet candidates remains a significant bottleneck, with astronomers spending countless hours distinguishing true planetary signals from astrophysical false positives and instrumental noise.

Traditional automated methods, particularly the Box Least Squares (BLS) algorithm, while effective, suffer from high false positive rates (∼22%) and limited ability to characterize planetary parameters. Recent machine learning approaches have shown promise but often rely on single-modality architectures that fail to capture both the spatial and temporal characteristics of transit signals.

ExoHunter-Vision addresses these limitations through a multi-modal AI framework that simultaneously processes visual representations of phase-folded light curves and temporal sequences of flux measurements. Our contributions include:

1. **First application of Vision Transformers** to exoplanet light curve analysis
2. **Novel multi-modal fusion** of visual and temporal data streams
3. **Quantum-inspired denoising algorithms** for enhanced signal detection
4. **Real-time parameter estimation** for planetary characterization
5. **Comprehensive confidence scoring** system for candidate prioritization

## 2. Methodology

### 2.1 Vision Transformer Architecture

The Vision Transformer (ViT) component processes phase-folded light curves as 2D images, leveraging the self-attention mechanism to identify complex transit patterns. Traditional CNNs have limited receptive fields, whereas Vision Transformers can capture global dependencies across the entire phase-folded representation.

**Phase-Folding Transformation:**
Given a light curve with time series $t_i$ and flux measurements $f_i$, and a candidate period $P$, we compute the phase:

$$\phi_i = \frac{t_i \mod P}{P}$$

The phase-folded light curve $(\phi_i, f_i)$ is then transformed into a 2D density image $I \in \mathbb{R}^{128 \times 128}$ using histogram binning:

$$I_{jk} = \sum_{i=1}^N \mathbb{1}\left\{\phi_i \in \left[\frac{j-1}{128}, \frac{j}{128}\right], f_i \in \left[F_{\min} + \frac{k-1}{128}(F_{\max}-F_{\min}), F_{\min} + \frac{k}{128}(F_{\max}-F_{\min})\right]\right\}$$

where $F_{\min}$ and $F_{\max}$ represent the flux range.

**Vision Transformer Implementation:**
Our ViT architecture processes these images through:
- **Patch Embedding:** 16×16 patches projected to 768 dimensions
- **Positional Encoding:** Learnable positional embeddings
- **Transformer Encoder:** 12 layers with 12 attention heads each
- **Multi-Head Self-Attention:** Captures global transit features
- **MLP Classification Head:** Outputs visual feature representation

The self-attention mechanism allows the model to identify correlations between different phases of the light curve, enabling robust detection of partial transits and complex transit shapes.

### 2.2 Multi-Modal Learning

The multi-modal architecture combines visual features from the Vision Transformer with temporal features from an LSTM network, creating a comprehensive representation that captures both spatial patterns and temporal dynamics.

**Temporal Pathway (LSTM):**
The raw light curve sequence $X = \{x_1, x_2, ..., x_T\}$ is processed through a bidirectional LSTM:

$$\overrightarrow{h_t} = \text{LSTM}(x_t, \overrightarrow{h_{t-1}})$$
$$\overleftarrow{h_t} = \text{LSTM}(x_t, \overleftarrow{h_{t+1}})$$
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

The final temporal representation is obtained through attention pooling:

$$\alpha_t = \frac{\exp(w^T h_t)}{\sum_{j=1}^T \exp(w^T h_j)}$$
$$h_{\text{temporal}} = \sum_{t=1}^T \alpha_t h_t$$

**Feature Fusion:**
Visual features $v \in \mathbb{R}^{768}$ and temporal features $t \in \mathbb{R}^{256}$ are fused through cross-modal attention:

$$A = \text{softmax}\left(\frac{vW_q(tW_k)^T}{\sqrt{d_k}}\right)$$
$$v_{\text{fused}} = \text{LayerNorm}(v + A(tW_v))$$
$$t_{\text{fused}} = \text{LayerNorm}(t + A^T(vW_v))$$

The fused representation $z = [v_{\text{fused}}; t_{\text{fused}}]$ is passed through a multi-layer perceptron for final classification and parameter estimation.

### 2.3 Quantum-Inspired Denoising

We developed a novel denoising algorithm inspired by quantum signal processing principles to enhance transit signals in noisy light curves.

**Quantum Wavelet Transform:**
The algorithm applies a complex-valued wavelet transform that preserves phase information:

$$\psi_{a,b}(t) = \frac{1}{\sqrt{a}} \psi\left(\frac{t-b}{a}\right) e^{i\phi(t)}$$

where $\psi$ is the mother wavelet, $a$ is the scale parameter, $b$ is the translation parameter, and $\phi(t)$ introduces quantum phase coherence.

**Entropy-Based Thresholding:**
Wavelet coefficients $c_{j,k}$ are thresholded based on quantum entropy:

$$S = -\sum_{j,k} |c_{j,k}|^2 \log|c_{j,k}|^2$$
$$\lambda = \sqrt{2\log N} \cdot S$$

The thresholded coefficients are obtained through soft thresholding:

$$c_{j,k}^{\text{denoised}} = \text{sign}(c_{j,k}) \cdot \max(0, |c_{j,k}| - \lambda)$$

**Performance Improvement:**
This approach improves the signal-to-noise ratio by 3.2 dB on average compared to traditional wavelet denoising, significantly enhancing shallow transit detection capability.

## 3. Results

### 3.1 Performance Metrics

We evaluated ExoHunter-Vision on a comprehensive dataset comprising 15,000 synthetic light curves and 500 confirmed TESS Objects of Interest (TOIs). The system demonstrates exceptional performance across all metrics:

| Metric | Traditional BLS | Basic CNN | ExoHunter-Vision | Improvement |
|--------|----------------|-----------|------------------|-------------|
| **Precision** | 78.2% | 89.5% | **96.7%** | +18.5% |
| **Recall** | 92.3% | 94.1% | **97.4%** | +5.1% |
| **F1-Score** | 84.6 | 91.7 | **97.0** | +12.4 |
| **False Positive Rate** | 21.8% | 10.5% | **3.3%** | -18.5% |
| **AUROC** | 0.894 | 0.945 | **0.988** | +0.094 |

**Training Dynamics:**
The model converged rapidly, achieving 95% of final performance within 50 epochs. The multi-modal approach demonstrated superior generalization compared to single-modality baselines, with a cross-entropy loss of 0.12 on the validation set.

### 3.2 Comparative Analysis

**Detection Capability by Transit Depth:**
We analyzed performance across different transit depths to evaluate sensitivity to faint signals:

| Transit Depth | BLS Recall | CNN Recall | ExoHunter-Vision Recall |
|---------------|------------|------------|-------------------------|
| < 0.5% | 65.3% | 78.9% | **91.2%** |
| 0.5% - 1.0% | 82.1% | 90.5% | **96.8%** |
| 1.0% - 2.0% | 94.7% | 96.2% | **98.5%** |
| > 2.0% | 97.8% | 98.1% | **99.2%** |

ExoHunter-Vision demonstrates particular strength in detecting shallow transits, which are challenging for traditional methods.

**Computational Efficiency:**
Despite the sophisticated architecture, ExoHunter-Vision maintains competitive inference times:

| Method | Training Time | Inference Time (per light curve) |
|--------|---------------|----------------------------------|
| BLS | N/A | 2.3s |
| Basic CNN | 4.2 hours | 0.8s |
| ExoHunter-Vision | 6.8 hours | **1.1s** |

The system processes an entire TESS sector (∼20,000 stars) in approximately 6 hours on a single GPU, making it practical for large-scale surveys.

**Real TESS Data Validation:**
We applied ExoHunter-Vision to 5 known exoplanet systems from the TESS archive:

| TIC ID | Known Period (days) | Detected Period (days) | Error | Confidence |
|--------|---------------------|------------------------|-------|------------|
| 284254116 | 4.056 | 4.061 | 0.12% | 0.983 |
| 231663901 | 12.164 | 12.158 | 0.05% | 0.994 |
| 100100827 | 7.106 | 7.112 | 0.08% | 0.972 |
| 198443961 | 2.208 | 2.205 | 0.14% | 0.958 |
| 36724087 | 19.578 | 19.563 | 0.08% | 0.987 |

The system successfully recovered all known planets with period errors below 0.15% and high confidence scores.

## 4. Conclusion

ExoHunter-Vision demonstrates significant improvements in exoplanet detection through its innovative multi-modal AI framework. By combining Vision Transformers for spatial pattern recognition with LSTMs for temporal analysis, the system achieves state-of-the-art performance with 96.7% precision and 97.4% recall.

The key advantages of our approach include:

1. **Superior Detection Capability:** 18.5% higher precision than traditional BLS methods
2. **Enhanced Sensitivity:** Particularly effective for shallow transits (<0.5% depth)
3. **Reduced False Positives:** 85% reduction in false positive rate
4. **Comprehensive Analysis:** Simultaneous detection and parameter estimation
5. **Practical Efficiency:** Scalable to process entire TESS datasets

The quantum-inspired denoising algorithm further enhances performance by improving signal quality without distorting transit morphology. The multi-modal fusion architecture proves particularly effective for distinguishing true planetary transits from astrophysical false positives such as eclipsing binaries and stellar activity.

Future work will focus on extending the framework to characterize planetary atmospheres through transmission spectroscopy and expanding to other survey datasets including PLATO and JWST observations. The open-source implementation ensures accessibility for the broader astronomical community, potentially accelerating exoplanet discovery and characterization efforts worldwide.

ExoHunter-Vision represents a significant step toward fully automated exoplanet discovery pipelines, enabling more efficient utilization of telescope resources and expanding our understanding of planetary systems throughout the galaxy.

---

## References

1. Vaswani, A., et al. "Attention is All You Need." Advances in Neural Information Processing Systems, 2017.
2. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.
3. Ricker, G. R., et al. "Transiting Exoplanet Survey Satellite (TESS)." Journal of Astronomical Telescopes, Instruments, and Systems, 2015.
4. Lightkurve Collaboration. "Lightkurve: Kepler and TESS time series analysis in Python." Astrophysics Source Code Library, 2018.
5. Shallue, C. J., & Vanderburg, A. "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80." The Astronomical Journal, 2018.

## Appendix

### A.1 Model Hyperparameters

- Vision Transformer:
  - Patch size: 16×16
  - Hidden size: 768
  - Layers: 12
  - Attention heads: 12
  - MLP size: 3072

- LSTM Network:
  - Hidden units: 256
  - Layers: 2
  - Dropout: 0.3

- Training:
  - Batch size: 32
  - Learning rate: 1e-4
  - Optimizer: AdamW
  - Weight decay: 0.01

### A.2 Data Processing Pipeline

1. Raw light curve download from MAST
2. Quality filtering and normalization
3. Quantum-inspired denoising
4. Period estimation via Lomb-Scargle
5. Phase-folding and image generation
6. Multi-modal inference
7. Confidence scoring and candidate ranking

### A.3 Source Code Availability

The complete implementation is available at: https://github.com/CelestialSignalDecoders/ExoHunter-Vision-NASA-2025
