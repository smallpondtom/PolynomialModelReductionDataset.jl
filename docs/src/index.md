# UniqueKronecker.jl

**UniqueKronecker.jl** is a Julia package for computing non-redundant (unique) Kronecker products of vectors, generalizing to _n_ dimensions and _k_-repeated products. It provides utility functions to work with the associated coefficient matrices, enabling conversions between unique Kronecker products and their standard (possibly redundant) Kronecker counterparts.

### What is the Unique Kronecker Product?

The standard Kronecker product of a vector ``\mathbf{x} \in \mathbb{R}^n`` with itself, ``\text{kron}(\mathbf{x}, \mathbf{x}) = \mathbf{x} \otimes \mathbf{x}``, produces all possible pairwise products of its elements, resulting in redundant terms when ``x_i x_j = x_j x_i``.

The **unique Kronecker product**, denoted here as ``\text{uniquekron}(\mathbf{x},\mathbf{x}) = \mathbf{x} \oslash \mathbf{x}``, eliminates these redundancies by considering only unique combinations of indices. For example:

For ``\mathbf{x} \in \mathbb{R}^2``:

- **Standard Kronecker product**:

```math
  \mathbf{x} \otimes \mathbf{x} = \begin{bmatrix} x_1^2 \\ x_1 x_2 \\ x_2 x_1 \\ x_2^2 \end{bmatrix}
```

- **Unique Kronecker product**:

```math
  \mathbf{x} \oslash \mathbf{x} = \begin{bmatrix} x_1^2 \\ x_1 x_2 \\ x_2^2 \end{bmatrix}
```

Here, ``x_1 x_2`` and ``x_2 x_1`` are considered the same and included only once.

### Coefficient Matrices

The package provides functions to compute the associated coefficient matrices:

- **Unique Kronecker Coefficient Matrix ``\mathbf{A}_{2u} \in \mathbb{R}^{n \times \frac{n(n+1)}{2}}``**: Represents the mapping of the unique Kronecker product back to the original vector ``\mathbf{x}\in\mathbb{R}^2``.
- **Kronecker Coefficient Matrix ``\mathbf{A}_2 \in \mathbb{R}^{n \times n^2}``**: Represents the mapping of the Kronecker product back to the original vector.

These matrices are useful for applications in polynomial regression, symmetric tensor computations, and vectorization of symmetric matrices.

## Features

- Compute the unique Kronecker product for vectors of any dimension ``n`` and any repeated (Kronecker) order ``k``.
- Generate the associated polynomial and Kronecker coefficient matrices ``\mathbf{A}_{2u}`` and ``\mathbf{A}_2``.
- Convert between unique and standard Kronecker products.
- Utility functions for polynomial modeling and symmetric tensor operations.

## Installation

You can install it using the command

```julia
using Pkg
Pkg.add("UniqueKronecker")
using UniqueKronecker
```

or install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/YourUsername/UniqueKronecker.jl")
```

Replace `YourUsername` with the actual GitHub username or organization where the package is hosted.

## Usage

### Importing the Package

```julia
using UniqueKronecker
```

### Computing the Unique Kronecker Product

Compute the ``k``-th order unique Kronecker product of vector `x`:

```julia
x = [2.0, 3.0, 4.0]  # Example vector in ℝ³

x_unique_kron =  x ⊘ x 
println(x_unique_kron)
# Output: [4.0, 6.0, 8.0, 9.0, 12.0, 16.0]
# Corresponding to [x₁², x₁x₂, x₁x₃, x₂², x₂x₃, x₃²]
```

### Computing Coefficient Matrices

#### Polynomial Matrix ``\mathbf{A}_2``

Compute the polynomial coefficient matrix ``\mathbf{A}_2``:

```julia
n = 3
H = zeros(n,n^2)
for i in 1:n
    x = rand(n)
    H[i,:] = kron(x,x)
end

println(H)
# Output: A matrix of size (3, 9) for this example
```

#### Unique/Nonredundant Polynomial Coefficient Matrix ``\mathbf{A}_{2u}``

Convert the polynomial matrix ``\mathbf{A}_2`` into the unique polynomial coefficient matrix ``\mathbf{A}_{2u}``:

```julia
A2u = eliminate(A2, 2)

println(A2u)
# Output: A matrix of size (3, 6) for this example
```

This can be converted back

```julia
A2 = duplicate(A2u, 2)
println(A2)
# Output: the H matrix
```

To make the coefficients symmetric for redundant terms use `duplicate_symmetric`

```julia
A2s = duplicate_symmetric(A2u, 2)
println(A2s)
# Output: the H matrix with symmetric coefficients
```

### Relationship Between Matrices

The following relationship holds:

```math
\mathbf{A}_{2u} \cdot (\mathbf{x} \oslash \mathbf{x}) = \mathbf{A}_2 \cdot (\mathbf{x} \otimes \mathbf{x})
```

This allows mapping between the unique Kronecker product space and the standard Kronecker product space.

### Generalizing to Higher-Order Products

Compute higher-order unique Kronecker products by specifying a higher value of ``k``:

```julia
k = 3  # Third-order product

x_unique_kron_k3 = unique_kronecker(x, k)  # or ⊘(x,k)

println(x_unique_kron_k3)
# Output: Corresponding unique products of order 3
```

## Applications

- **Polynomial Regression**: Efficient computation of polynomial features without redundant terms.
- **Symmetric Tensor Computations**: Simplifies operations involving symmetric tensors.
- **Model Reduction**: Construction of reduced-order models with polynomial structures.
- **Machine Learning**: Feature engineering for higher-order interactions.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue. If you'd like to contribute code, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/smallpondtom/UniqueKronecker.jl/blob/main/LICENSE).