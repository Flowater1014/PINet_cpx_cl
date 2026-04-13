# PINet Complex Reconstruction (PINet_cpx_cl)

A PyTorch implementation of PINet (Phase Imaging Network) for complex-valued data reconstruction. This repository contains model definitions, dataset processing, loss functions, and training/testing scripts for deep learning-based phase retrieval and complex signal recovery.

## Features

- Complex-valued neural network layers and functions
- Custom dataset loaders for complex data
- Training and evaluation pipelines
- Jupyter notebooks for experimentation and visualization

## Project Structure

```
PINet_cpx_cl/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── complexFunctions.py # Complex number operations
│   ├── complexLayers.py    # Complex-valued neural layers
│   ├── model.py           # PINet model definition
│   ├── mydataset.py       # Custom dataset classes
│   ├── myloss.py          # Loss functions for complex data
│   └── utilities.py       # Utility functions
├── notebooks/             # Jupyter notebooks for experiments
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

**Note**: The following directories contain large datasets, model weights, and experimental results which are excluded from version control (see `.gitignore`):

- `datasets/` - Raw and processed datasets
- `real_data/` - Real-world measurement data  
- `save_dir/` - Trained model checkpoints
- `result_data_experiment/` - Experiment results
- `model_saved_pinet_compared*/` - Model comparison results
- `pinet_dataset_compared*/` - Dataset comparison results
- `pinet_1600/` - Specific dataset variant

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Flowater1014/PINet_cpx_cl.git
   cd PINet_cpx_cl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The `requirements.txt` currently specifies Python 3.8.10 and PyTorch 1.10.0+cu113. For a more complete dependency list, you may need to install:
   ```bash
   pip install torch torchvision numpy matplotlib scipy jupyter
   ```

## Usage

### Training

1. Prepare your data in the appropriate format (see `src/mydataset.py` for expected structure)
2. Modify training parameters in the relevant scripts or notebooks
3. Run training:
   ```python
   # Example training command (adjust based on your setup)
   python -m src.train  # If a training script exists
   ```
   
   Alternatively, use the provided Jupyter notebooks in `notebooks/` for experimentation.

### Evaluation

Load a trained model and run inference:
```python
import torch
from src.model import PINet

model = PINet()
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
# Perform inference...
```

### Notebooks

The `notebooks/` directory contains Jupyter notebooks for:
- Data exploration and visualization
- Model training and validation
- Result analysis and plotting

## Data Management

Large files are excluded via `.gitignore`. To share data or models:

1. Use cloud storage (Google Drive, Dropbox, etc.) with shareable links
2. Include download instructions in this README
3. Consider providing a small example dataset for demonstration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:
```bibtex
[Add citation information]
```

## Acknowledgments

[Add any acknowledgments here]