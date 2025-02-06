using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

bound_trg_blk(steps::Int, data) = abs(log(data[end][1]) * 4.0^((1-steps)))
bound_trg_env(steps::Int, data) = abs(log(data[end][2]) * 2.0^((1-steps)))
stopping_criterion_boundary = maxiter(7)&convcrit(1e-20, bound_trg_env)&convcrit(1e-20, bound_trg_blk)
stopping_criterion_trg = maxiter(20)
#stopping_criterion = convcrit(1e-20, bound_trg_blk)&maxiter(40)
Ising_βc = log(1.0+sqrt(2))/2.0

T = classical_ising_symmetric(Ising_βc)
# scheme = TRG(T)
# run!(scheme, truncdim(16), stopping_criterion_trg; finalize_beginning=true)
# T = scheme.T

V = Vect[Z2Irrep](0=>1,1=>1)
V_trivial = Vect[Z2Irrep](0=>1)
E1 = TensorMap(ones, V_trivial⊗V←V_trivial)
E2 = TensorMap(ones, V_trivial←V_trivial⊗V)
scheme = Boundary_TRG(T, E1, E2)
data_boundtrg = run!(scheme, truncdim(24) ,stopping_criterion_boundary; finalize_beginning=true)