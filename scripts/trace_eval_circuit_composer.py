from qtensor_ai import ParallelComposer

# compose the circuit to evaluate the trace of the target circuit
class TraceEvaluationComposer(ParallelComposer):
    
    def __init__(self, n_qubits, com):
        self.n_target_qubits = n_qubits
        self.n_qubits = n_qubits*2
        self.com = com
        super().__init__(n_qubits*2)

    # add circuit (ancilla qubit circuit) needed for calculating the trace
    def added_circuit(self):
        for target_qubit in range(self.n_target_qubits):
            control_qubit = target_qubit + self.n_target_qubits
            self.apply_gate(self.operators.H, control_qubit)
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)

    # building circuit whose first amplitude is the expectation value of the measured circuit wrt to the cost_operator
    def updated_full_circuit(self, **parameters):
        self.com.n_batch = self.n_batch
        circuit = self.com.updated_full_circuit(**parameters)
        self.builder.reset()
        self.added_circuit()
        first_part = self.builder.circuit
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        result_circuit = first_part + circuit + second_part
        return result_circuit

    def name(self):
        return 'TraceEvaluation'