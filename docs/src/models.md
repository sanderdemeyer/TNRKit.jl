# Models

TNRKit.jl includes several well-tested statistical mechanical models that can be used with various tensor network renormalization schemes. These models are implemented as partition function tensors that respect the TNRKit leg convention.

## Overview

The models are organized into two main categories:
- **2D Models**: For square lattice tensor networks  
- **3D Models**: For cubic lattice tensor networks

Each model typically comes in multiple variants:
- **Regular variants**: Standard implementations without explicit symmetry
- **Symmetric variants**: Implementations with explicit global symmetry that can improve computational efficiency

## 2D Models

### Classical Ising Model

The classical Ising model is one of the most studied models in statistical mechanics, describing magnetic systems with spin-1/2 particles on a lattice.

#### Physics
The Hamiltonian for the 2D Ising model on a square lattice is:
```
H = -J ∑_{⟨i,j⟩} σᵢσⱼ - h ∑ᵢ σᵢ
```
where `σᵢ ∈ {-1, +1}` are spin variables, `J` is the coupling constant, `h` is the external magnetic field, and the first sum runs over nearest-neighbor pairs.

#### Available Functions

##### [`classical_ising`](@ref)
Standard implementation for arbitrary temperature and magnetic field.

```julia
# Critical temperature, zero field
T = classical_ising()  # Uses βc = ln(1+√2)/2

# Custom temperature and magnetic field  
T = classical_ising(0.5; h=0.1)
```

!!! note "Free Energy Calculation"
    When using `free_energy()` with `classical_ising`, set `initial_size=2.0` since this tensor represents a 2×2 unit cell.

##### [`classical_ising_symmetric`](@ref) 
Implementation with explicit ℤ₂ symmetry, computationally more efficient.

```julia
# Critical temperature (default)
T = classical_ising_symmetric()

# Custom temperature
T = classical_ising_symmetric(0.8)
```

This version exploits the global ℤ₂ spin-flip symmetry and typically provides better performance and more stable numerics.

#### Example Usage

```julia
using TNRKit, TensorKit

# Compute free energy at critical temperature
T = classical_ising_symmetric(ising_βc)
scheme = BTRG(T)  # Bond-weighted TRG works well for Ising
data = run!(scheme, truncdim(32), maxiter(20))

# Calculate free energy and compare to Onsager's exact result
f_numerical = free_energy(data, ising_βc)
println("Numerical: $f_numerical")
println("Onsager:   $f_onsager") 
println("Error:     $(abs(f_numerical - f_onsager))")
```

### Classical Potts Model

The q-state Potts model generalizes the Ising model to systems with q discrete states per site.

#### Physics
The Hamiltonian is:
```
H = -J ∑_{⟨i,j⟩} δ(σᵢ, σⱼ)
```
where `σᵢ ∈ {1, 2, ..., q}` and `δ(σᵢ, σⱼ)` is the Kronecker delta.

#### Available Functions

##### [`classical_potts`](@ref)
Standard implementation for q states.

```julia
# 3-state Potts at critical temperature
T = classical_potts(3)  # Uses βc = ln(1+√q)

# Custom temperature
T = classical_potts(4, 0.6)
```

##### [`classical_potts_symmetric`](@ref)
Implementation with explicit ℤq symmetry.

```julia
# 3-state Potts with Z3 symmetry
T = classical_potts_symmetric(3)

# 5-state Potts with custom temperature  
T = classical_potts_symmetric(5, 0.4)
```

#### Utility Functions

##### [`potts_βc`](@ref)
Returns the critical inverse temperature for the q-state Potts model:

```julia
βc = potts_βc(3)  # Critical temperature for 3-state Potts
```

#### Example Usage

```julia
# Study 3-state Potts model phase transition
T = classical_potts_symmetric(3, potts_βc(3))
scheme = TRG(T)
data = run!(scheme, truncdim(24), maxiter(15))
f = free_energy(data, potts_βc(3))
```

### Six-Vertex Model

The six-vertex model describes the statistical mechanics of hydrogen bonds in ice or vertex configurations on a square lattice.

#### Physics
The model assigns Boltzmann weights to six allowed vertex configurations with parameters `a`, `b`, and `c`. Each vertex represents the intersection of four bonds, and the model enforces an "ice rule" constraint.

#### Available Functions

##### [`sixvertex`](@ref)
Configurable implementation supporting different symmetry sectors and element types.

```julia
# Default: Trivial symmetry, equal weights
T = sixvertex()

# Custom weights with Trivial symmetry
T = sixvertex(Float64, Trivial; a=1.0, b=2.0, c=1.5)

# With U(1) symmetry 
T = sixvertex(ComplexF64, U1Irrep; a=2.0, b=1.0, c=1.0)

# With compact U(1) symmetry
T = sixvertex(CU1Irrep; a=1.0, b=1.0, c=2.0)
```

#### Symmetry Options
- `Trivial`: No explicit symmetry
- `U1Irrep`: U(1) symmetry (continuous rotation)  
- `CU1Irrep`: Compact U(1) symmetry

#### Example Usage

```julia
# Study six-vertex model at special point
T = sixvertex(Float64, U1Irrep; a=1.0, b=1.0, c=1.0)
scheme = HOTRG(T)  # Higher-order TRG often works well
data = run!(scheme, truncdim(20), maxiter(12))
```

### Classical Clock Model

The clock model interpolates between the Ising model (q=2) and the XY model (q→∞), with q discrete orientations per site.

#### Physics  
The Hamiltonian involves nearest-neighbor interactions between discrete angles:
```
H = -J ∑_{⟨i,j⟩} cos(2π(σᵢ - σⱼ)/q)
```
where `σᵢ ∈ {0, 1, ..., q-1}` represents one of q equally spaced angles.

#### Available Functions

##### [`classical_clock`](@ref)
Implementation for q discrete states and inverse temperature β.

```julia
# 6-state clock model
T = classical_clock(6, 1.0)

# 4-state clock model (equivalent to two decoupled Ising models)
T = classical_clock(4, 0.8)
```

#### Example Usage

```julia
# Study 8-state clock model
T = classical_clock(8, 1.2) 
scheme = LoopTNR(T)  # Loop optimization can help with critical models
data = run!(scheme, truncdim(28), maxiter(18))
```

### Gross-Neveu Model

An experimental implementation of the Gross-Neveu model, relevant for studying fermionic systems and quantum field theory.

#### Physics
The Gross-Neveu model is a quantum field theory with fermionic degrees of freedom, relevant for studying dynamical symmetry breaking and critical phenomena in (1+1) dimensions.

#### Available Functions

##### [`gross_neveu_start`](@ref) 
Constructs the partition function tensor with parameters μ (chemical potential), m (mass), and g (coupling).

```julia
# Gross-Neveu model at symmetric point
T = gross_neveu_start(0.0, 0.0, 1.0)
```

!!! warning "Experimental"
    This model is experimental and may undergo interface changes in future versions.

## 3D Models

### 3D Classical Ising Model

Extensions of the classical Ising model to three-dimensional cubic lattices.

#### Available Functions

##### [`classical_ising_3D`](@ref)
Standard 3D implementation with coupling constant control.

```julia
# Default: critical temperature and unit coupling
T = classical_ising_3D()  # Uses βc_3D ≈ 1/4.51

# Custom parameters
T = classical_ising_3D(0.3; J=1.5)
```

##### [`classical_ising_symmetric_3D`](@ref)
3D implementation with explicit ℤ₂ symmetry.

```julia
# Symmetric version at critical temperature
T = classical_ising_symmetric_3D()

# Custom temperature
T = classical_ising_symmetric_3D(0.25)
```

#### Example Usage

```julia
# 3D Ising model simulation
T = classical_ising_symmetric_3D(ising_βc_3D)
scheme = ATRG_3D(T)  # 3D anisotropic TRG
data = run!(scheme, truncdim(16), maxiter(10))
```

## Implementation Notes

### Leg Convention
All model tensors follow the TNRKit leg convention where the tensor lives in the space `V₁ ⊗ V₂ ← V₃ ⊗ V₄` with legs ordered as:

```
     3
     |
     v
     |
1-<--┼--<-4  
     |
     v
     |
     2
```

For 3D models, the space is `V_D ⊗ V_U' ← V_N ⊗ V_E ⊗ V_S' ⊗ V_W'` where D,U,N,E,S,W represent Down, Up, North, East, South, West directions.

### Symmetry Benefits
Using symmetric variants (when available) provides several advantages:
- **Computational efficiency**: Exploiting block structure reduces memory and computation
- **Numerical stability**: Better conditioning of the renormalization flow  
- **Physical insight**: Makes symmetry-breaking phenomena more transparent

### Temperature Parameters
Most functions accept inverse temperature `β = 1/(kT)` where `k` is Boltzmann constant and `T` is temperature. Critical temperatures are provided as constants:
- `ising_βc`: 2D Ising critical inverse temperature  
- `ising_βc_3D`: 3D Ising critical inverse temperature
- `potts_βc(q)`: q-state Potts critical inverse temperature

### Usage with TNR Schemes
Different models work better with different TNR schemes:
- **Ising models**: BTRG, TRG, LoopTNR all work well
- **Potts models**: TRG, HOTRG are reliable choices  
- **Six-vertex**: HOTRG, SLoopTNR can handle the symmetries well
- **3D models**: ATRG_3D is the primary choice

Always experiment with different `truncdim` values and iteration counts to balance accuracy and computational cost.