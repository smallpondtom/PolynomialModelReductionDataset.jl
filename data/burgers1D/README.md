### 1D Burgers' Equation

#### Type 01
- Spatial domain $\Omega \in [0,1.0]$
- Time domain $\mathcal T \in [0,1.0]$
- Dirichlet BC
    - $x(0,t) = 1.0$
    - $x(1,t) = -1.0$
- Zero initial condition
- Quadratic
- Output
    - $\mathbf{C} = \frac{1}{n} [1, 1, 1, \cdots, 1 ]^\top$  where $n$ is the state dimension
- Reference input vector of 1
- Integrated with semi-implicit Euler scheme with time step of $\Delta t = 1e\text{-}4$

#### Type 02
Energy preserving modified Burgers' equation explained in [[Aref and Daripa 1984]](http://epubs.siam.org/doi/10.1137/0905060).

- Spatial domain $\Omega \in [0,1.0]$
- Time domain $\mathcal T \in [0,1.0]$
- Periodic BC
- Sine Wave initial condition: $\sin(2\pi x)$
- Quadratic
- No input
- No output
- Integrated with semi-implicit Euler scheme with timestep $\Delta t=1e\text{-}4$

#### Operators
$$
\begin{align}
    \dot{\mathbf x} &= \mathbf A \mathbf x + \mathbf H(\mathbf x \otimes \mathbf x) + \mathbf B\mathbf u \\
    \mathbf y &= \mathbf C \mathbf x
\end{align}
$$
- A: Linear Operator
- B: Input Operator
- C: Output Operator
- H: Quadratic Operator
- F: Quadratic Operator (without redundancy)
