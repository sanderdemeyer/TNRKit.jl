using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

Ising_βc = log(1.0 + sqrt(2)) / 2.0
scheme = Loop_TNR(classical_ising(Ising_βc), classical_ising(Ising_βc))

entanglement_filtering!(scheme, 200, 1e-20)
entanglement_filtering!(scheme, 200, 1e-20)



