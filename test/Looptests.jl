using Revise, TensorKit, Plots, QuadGK, DataFrames, CSV
includet("../src/TRGKit.jl")
using .TRGKit
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^((1-steps)))

stopping_criterion_tnr = convcrit(1e-20, trg_f)&maxiter(9)
Ising_βc = log(1.0 + sqrt(2)) / 2.0

scheme = Loop_TNR(classical_ising_symmetric(Ising_βc), classical_ising_symmetric(Ising_βc))  
data_tnr = []
@info "Finalizing beginning"
push!(data_tnr, scheme.finalize!(scheme))

stopping_criterion_tnr = convcrit(1e-20, trg_f)&maxiter(9)

steps = 0
crit = true
while crit
    @info "Step $(steps + 1), data_tnr[end]: $(!isempty(data_tnr) ? data_tnr[end] : "empty")"
    step!(scheme, 16, 100, 1e-20, 30, 1e-12)
    push!(data_tnr, scheme.finalize!(scheme))
    steps += 1
    crit = stopping_criterion_tnr(steps, data_tnr)
end
lnz_tnr = 0
for (i,d) in enumerate(data_tnr)
    lnz_tnr += log(d) * 2.0^(-i)
end
@show lnz_tnr

@tensor opt=true transfer_ten[-1 -2; -3 -4] := scheme.TA[-1 1; 2 5]*scheme.TB[2 3; -3 6]*scheme.TB[-2 5; 4 1]*scheme.TA[4 6; -4 3]
#@tensor opt=true transfer_ten[-1 -2; -3 -4] := scheme_atrg.T[-1 1; 2 5]*scheme_atrg.T[2 3; -3 6]*scheme_atrg.T[-2 5; 4 1]*scheme_atrg.T[4 6; -4 3]
D, V = eig(transfer_ten)
diag = []
for (i,d) in blocks(D)
    push!(diag, d...)
end
diag = sort!(real(diag))