using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

Ising_βc = log(1.0+sqrt(2))/2.0
scheme = Loop_TNR(gross_neveu_start(1,1,1), gross_neveu_start(1,1,1))

entanglement_filtering!(scheme, 100, 1e-10)
