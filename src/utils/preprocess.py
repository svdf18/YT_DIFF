# 2. Create dataset preprocessing (src/utils/preprocess.py)
class AudioPreprocessor:
    def __init__(self, config):
        self.input_dir = config.input_dir
        self.output_dir = config.output_dir
        self.format = SpectrogramFormat(config)
        
    def process_file(self, audio_path):
        # Load audio
        # Convert to spectrogram
        # Save processed data
        pass
