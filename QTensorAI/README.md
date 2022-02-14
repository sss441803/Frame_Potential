# QTensorAI
A hybrid quantum-classical neural network simulation platform. Quantum simulation uses QTensor, a state-of-the-art tensor network-based simulator that usually has linear complexity in the number of qubits for shallow circuits, instead of exponential complexity. This opens up the possibility to simulate large hybrid models with many qubits. The hybrid model is a PyTorch model, batch-parallelized, GPU compatible and fully differentiable.
# Installation
First, you need to install QTensor from [source code](https://github.com/danlkv/qtensor).

```bash
# --recurse-submodules is important since qtensor has a submodule qtree  
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
cd QTensor

# using git branch -v -a to check all branchs
# and switch to dev branch
git switch dev
# checkout the specific commit 
git checkout dc53509

cd qtree
pip install .

cd .. # back to QTensor folder
pip install .
```

You can check if the `qtensor` and `qtree` are installed or not using `pip3 list`

### QTensorAI Installation ###

```bash
cd QTensorAI
python3 setup.py install
```

You can check if the `qtensor-ai` is installed or not using `pip3 list`
