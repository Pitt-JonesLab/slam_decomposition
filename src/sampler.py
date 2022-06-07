"""
Define functions which act as distributions for a template to train against
"""

#XXX if provide a gate object should override this
#TODO empiric distribution ,pass in a circuit and build a generator object using stopiteration

if name == "Haar":
    return lambda: random_unitary(dims=2 ** self.template.n_qubits).data
if name == "Haar-2":
    logging.warning(f"Only works for \sqrt[2]iSwap")
    return lambda: Operator(self._haar_ground_truth(2)).data
if name == "Haar-3":
    logging.warning(f"Only works for \sqrt[2]iSwap")
    return lambda: Operator(self._haar_ground_truth(3)).data
if name == "Clifford":
    return lambda: Operator(
        random_clifford(num_qubits=self.template.n_qubits)
    ).data
else:
    raise ValueError(f"No sample function named {name}")

def _haar_ground_truth(self, haar_exact=2):
    """When using sqrt[2] iswap, we might want to do a haar sample where we know ahead of time if it will take 2 or 3 uses
    this is used for establishing the effectiveness of our optimizer, but won't work for any other basis gate"""
    from qiskit.transpiler.passmanager import PassManager
    from qiskit.transpiler.passes import CountOps
    from deprecate.weyl_exact import RootiSwapWeylDecomposition
    pm0 = PassManager()
    pm0.append(RootiSwapWeylDecomposition(basis_gate=RiSwapGate(0.5)))
    pm0.append(CountOps())
    logger.setLevel(logging.CRITICAL)
    while True:
        qc = QuantumCircuit(2)
        qc.append(random_unitary(dims=4), [0,1])
        pm0.run(qc)
        if haar_exact == pm0.property_set['count_ops']['riswap']:
            logger.setLevel(logging.INFO)
            return qc