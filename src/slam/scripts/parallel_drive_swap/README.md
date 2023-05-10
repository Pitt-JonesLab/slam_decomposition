## Searching for a faster SWAP decomposition

We are considering this Hamiltonian design space, given 2Q conversion and gain drive strengths, and smaller duration 1Q drive strengths performed simulatenously. We have shown that combining drives in this manner expands the accessible region of decomposition targets in a way that reduces the overall execution time for 2Q basis gates.

$\hat{H} = g_c (e^{i \phi_c} a^\dagger b + e^{-i \phi_c}a b^\dagger) + g_g (e^{i \phi_g}ab + e^{-i \phi_c}a^\dagger b^\dagger) + \epsilon_1(t)(a + a^\dagger) + \epsilon_2(t)(b + b^\dagger)$



### ParallelDrivenGateWidget
This class let's us build a Cartan trajectory[^1], given a parameterization of the Conversion-Gain Hamiltonian. Rather than using the unitary instance `ConversionGainSmush1QPhaseGate`, which outputs only the final point on the trajectory, `ParallelDrivenGateWidget` may offer a way to improve cost of decomposition by analyzing the trajectory along the trajectory (with the goal to move to the target gate more directly and hence shorter in time).

#### Tool that lets you modify the parallel-driven 1Q drive amplitudes and instantly visualize the updated trajectory
![image](https://user-images.githubusercontent.com/47376937/211901638-8de0c0c1-d1ac-49f5-9a62-564b2c501407.png)


Three main components of the widget
First, constructing the circuit which applies the gate for each time step we want to track.
```python
def construct_basis(self):

        varg_offset = 0
        #ConversionGainSmushGate
        pp2 =lambda *vargs: ConversionGainSmush1QPhaseGate(self.phase_a, self.phase_b, self.phase_c,self.phase_g, self.gc, self.gg, vargs[varg_offset:varg_offset+round(self.t/self.duration_1q)], vargs[varg_offset+round(self.t/self.duration_1q):], t_el=self.t)
        basis = CircuitTemplateV2(n_qubits=2, base_gates = [pp2], edge_params=[[(0,1)]], no_exterior_1q=True, param_vec_expand=[varg_offset,round(self.t/self.duration_1q),round(self.t/self.duration_1q)])
        basis.build(1)
        # basis.circuit.draw(output='mpl');

        # repeat the atomic pd gate multiple times
        qc = QuantumCircuit(2)
        for _ in range(self.N):
            qc = qc.compose(basis.circuit)
        self.qc = qc
```
Second, strip the circuit of real-valued durations and replace with a `Parameter`
```python
 def prepare_parameters_nonuniform(self, g0_vector, g1_vector):
        """same as prepare_parameters but the 1Q amplitudes change in time"""
        assert len(g0_vector) == len(g1_vector) == self.N
        i = 0
        out = self.qc.copy()
        for instr, qargs, cargs in out:
            if instr.params and instr.name =="2QSmushGate":
                instr.params[4:6] = [g0_vector[i], g1_vector[i]]
                instr.params[-1] = Parameter(f't{i}')
                i +=1
            elif instr.params and instr.name == "2QSmushGate1QPhase":
                instr.params[6:8] = [g0_vector[i], g1_vector[i]]
                instr.params[-1] = Parameter(f't{i}')
                i +=1
        self.prep_qc = out
 ```
Third, iterate through the timesteps. This function takes the template circuit and only fills with values up until the current time step before saving as a Weyl coordinate.

```python
def iterate_time(self, R=5):
        # R = 5 # resolution
        endpoints = range(1, self.N+1)
        coordinate_list = []
        end_segment_list = []

        for end in endpoints:
            temp_coords = []
            qc = QuantumCircuit(2)
            for gate in self.prep_qc[0:end]:
                qc.append(gate[0], gate[1])

            qc2 = qc.copy()
            # for all prior 2Q gates, set time parameter to full length
            for i in [el for el in endpoints if el < end]:
                qc2 = qc2.bind_parameters({qc2[i-1][0].params[-1] : self.duration_1q})
            # for current 2Q gate, iterate over time and append coordinate
            for t in np.linspace(0, self.duration_1q, R):
                qc3 = qc2.bind_parameters({qc2[end-1][0].params[-1]: t})
                #eliminating x-axis symmetry
                c = list(c1c2c3(Operator(qc3).data))
                if c[0] > 0.5:
                    c[0] = -1*c[0] + 1
                temp_coords.append(c)
            coordinate_list.append(temp_coords)
            end_segment_list.append(c)
        self.coordinate_list = coordinate_list
        self.end_segment_list = end_segment_list
        self.final_unitary = Operator(qc3).data
```

[^1]:
