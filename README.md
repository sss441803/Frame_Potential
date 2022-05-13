# Frame_Potential
Evaluating the frame potential of unitary ensembles using QTensor trace evaluation

This is the implementation of the paper "Estimating frame potential of large-scale quantum circuit sampling using tensor networks up to 50 qubits" by Minzhao Liu, Junyu Liu, Yuri Alexeev and Liang Jiang

Frame potential is a quantity of quantum circuit ensembles that describes how well it represents the Haar distribution (uniform sampling the unitary group). This is of important theoretical consideration due to its relationship with k-designs, complexity, expressibility and the barren plateau problem. We evaluate the frame potential numerically to verify the scaling laws associated with the parallel and local random unitary ansatze and the hardware efficient ansatze. This leads to a numerical verification of the Brown-Susskind conjecture, as well as an understanding in expressibility.

The numerical scheme uses the QTensorAI library, based on the quantum simulator QTensor which utilizes the graphical tensor network formalism. This allows shallow circuits to be simulated with complexity linear to the number of qubits (instead of exponential).

The code base requires the QTensorAI library, available at https://github.com/sss441803/QTensorAI.git. We recommend a separate environment for data analysis to avoid potential conflicts with scipy and other libraries. The data analysis environment requires scipy, numpy, matplotlib and uncertainties.

Each circuit ansatze has its own folder of code for simulation. Results are in the `/results` folder, and respective ansatze directories with a `frame_potential.npy` file.
