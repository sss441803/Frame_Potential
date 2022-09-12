# Frame_Potential
Evaluating the frame potential of unitary ensembles using QTensor trace evaluation

This is the implementation of the paper "Estimating frame potential of large-scale quantum circuit sampling using tensor networks up to 50 qubits" by Minzhao Liu, Junyu Liu, Yuri Alexeev and Liang Jiang

Frame potential is a quantity of quantum circuit ensembles that describes how well it represents the Haar distribution (uniform sampling the unitary group). This is of important theoretical consideration due to its relationship with k-designs, complexity, expressibility and the barren plateau problem. We evaluate the frame potential numerically to verify the scaling laws associated with the parallel and local random unitary ansatze and the hardware efficient ansatze. This leads to a numerical verification of the Brown-Susskind conjecture, as well as an understanding in expressibility.

The numerical scheme uses the QTensorAI library, based on the quantum simulator QTensor which utilizes the graphical tensor network formalism. This allows shallow circuits to be simulated with complexity linear to the number of qubits (instead of exponential).

The code base requires the QTensorAI library, available at https://github.com/sss441803/QTensorAI.git. We recommend a separate environment for data analysis to avoid potential conflicts with scipy and other libraries. The data analysis environment requires scipy, numpy, matplotlib and uncertainties.

To run simulations for the parallel random unitary or the hardware efficient ansatze, first run the following lines to pre-compute the contraction orders on a CPU system:
```bash
cd scripts
python run.py --mode peo_finder
```
After that, run the following lines to execute the actual simulation:
```bash
python run.py --mode simulation
```
To calculate the frame potential from the simulation trace results, run:
```bash
python run.py --mode bootstrapped_frame_potential
```
The implementation of the local random unitary ansatze is CPU based. To run it, you need to run the `run_local.py` file similarly like before. No need to run the `peo_finder` mode because we will NOT optimize the contraction order (they will be all different for each random circuit).


Results are in the `/results` folder, and respective ansatze directories with a `bootstrapped_frame_potential.npy` file.
