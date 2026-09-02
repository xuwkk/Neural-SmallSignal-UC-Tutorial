# NN-based Small-Signal Stability-Constrained Unit Commitment Tutorial

## Key packages

| Package | Purpose |
| --- | --- |
| [ANDES](https://docs.andes.app/) | Run AC power flow, dynamic initialization, and small-signal eigenvalue analysis. |
| [PyTorch](https://pytorch.org/) | Train the ReLU neural network to predict the critical eigenvalue real part. |
| [CVXPY](https://www.cvxpy.org/) | Formulate and solve the baseline and stability-constrained unit commitment problems. |
| [NCET](https://github.com/xuwkk/ncet) | Encode the trained PyTorch network as mixed-integer linear constraints in CVXPY. |

Run the notebooks in this order:

1. `small_signal_concept_with_andes.ipynb`: [![Open ANDES basics in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuwkk/Neural-SmallSignal-UC-Tutorial/blob/main/small_signal_concept_with_andes.ipynb)
2. `stability_constrained_optimization_with_ncet.ipynb`: [![Open stability-constrained UC in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuwkk/Neural-SmallSignal-UC-Tutorial/blob/main/stability_constrained_optimization_with_ncet.ipynb)

## Local installation

```bash
git clone https://github.com/xuwkk/Neural-SmallSignal-UC-Tutorial.git
cd Neural-SmallSignal-UC-Tutorial
mamba env create -f environment.yml
conda activate smallsignal_sco_native
python -m ipykernel install --user --name smallsignal_sco_native --display-name "Python (smallsignal_sco_native)"
```

Open the notebooks in Jupyter or VS Code and select the `Python (smallsignal_sco_native)` kernel.

Each notebook contains a cell explicitly marked **GOOGLE COLAB ONLY**. In Colab, that cell clones this complete repository and installs `requirements-colab.txt`. In a local Jupyter environment, it performs no action.

The Colab virtual machine is temporary. Download files from `outputs/` before ending the session if they need to be retained.

For the complete model design and implementation notes, see [tutorial_design_and_instructions.md](tutorial_design_and_instructions.md).
