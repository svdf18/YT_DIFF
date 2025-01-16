"""
This file implements sigma sampling strategies for diffusion model training.

Key components:

1. Sigma Sampler Configuration:
   - Controls sampling parameters like min/max sigma values
   - Configures distribution type and parameters
   - Handles PDF-based and stratified sampling options

2. Sampling Distributions:
   - Log-normal distribution
   - Log-sech and log-sech^2 distributions  
   - Log-linear distribution
   - Scale-invariant distribution
   - Custom PDF-based distribution

3. Sampling Methods:
   - Stratified sampling for better coverage
   - Static sampling for reproducibility
   - PDF-based importance sampling
   - Various distribution-specific sampling functions

4. Helper Functions:
   - PDF generation and normalization
   - CDF computation and inversion
   - Distribution parameter calculations
"""
