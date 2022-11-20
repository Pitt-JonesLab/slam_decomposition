# decomposition_EM
Numerical optimizers for decomposing unitary ops, or Hamiltonian parameter sweeping

Example Usage [slighlty deprecated]:

```python
from basis import CircuitTemplate
basis = CircuitTemplate(maximum_span_guess=4, preseed=False)
```

```python
from cost_function import BasicCost
objective = BasicCost()
```

```python
from optimizer import TemplateOptimizer
optimizer = TemplateOptimizer(basis=basis, objective=objective, use_callback=True)
```

```python
from sampler import HaarSample
sampler = HaarSample(n_samples=1)
ret = optimizer.approximate_from_distribution(sampler=sampler)
```

```
INFO:root:Starting sample iter 0
INFO:root:Begin search: (0.58941013, 0.22184674, 0.11209285)
INFO:root:Starting opt on template size 1
INFO:root:Starting opt on template size 2
INFO:root:Break on cycle 2
INFO:root:Loss=1.0999245958487336e-10
INFO:root:Success: (0.58941013, 0.22184674, 0.11209285)
INFO:root:Saving data back to file
```

```python
from utils.visualize import optimizer_training_plot
optimizer_training_plot(*ret)
```
![image](https://user-images.githubusercontent.com/47376937/172430812-33e6a9ec-0470-4cd0-b6b3-43eb5b3214d1.png)