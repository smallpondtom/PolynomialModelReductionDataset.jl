### [Fitzhugh-Nagumo Equation](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model?oldformat=true)

The data provided follows the exact setting from [MORwiki](http://modelreduction.org/index.php/FitzHugh-Nagumo_System)
```
The MORwiki Community, FitzHugh-Nagumo System. MORwiki - Model Order Reduction Wiki, 2018. 
http://modelreduction.org/index.php/FitzHugh-Nagumo_System
```

To produce the data, [Qian et al. 2020](https://linkinghub.elsevier.com/retrieve/pii/S0167278919307651) was also referenced.

##### Operators
$$
\begin{align}
  \dot{\mathbf x} &= \mathbf A \mathbf x + \mathbf H(\mathbf x \otimes \mathbf x) + (\mathbf N\mathbf x)\mathbf u + \mathbf K \\
  \mathbf y &= \mathbf C \mathbf
\end{align}
$$
- A: Linear Operator
- B: Input Operator
- C: Output Operator
- H: Quadratic Operator
- F: Quadratic Operator (with no redundancy)
- N: Bilinear Operator
- K: Constant Operator
