"""
This file implements the sampling pipeline for the YT_DIFF model.

Key functionality:
- Provides command line interface for sampling with required arguments:
  - sample_cfg_file: Path to JSON configuration file with sampling parameters

- Initializes environment for GPU sampling (mac only)

- Loads sampling configuration from JSON into SampleParams object
  which contains all hyperparameters and settings for sampling
"""