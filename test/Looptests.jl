using Revise, TensorKit, Plots, QuadGK, DataFrames, CSV
includet("../src/TRGKit.jl")
using .TRGKit
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^((1-steps)))

stopping_criterion_tnr = convcrit(1e-20, trg_f)&maxiter(9)
Ising_βc = log(1.0 + sqrt(2)) / 2.0

scheme = Loop_TNR(gross_neveu_start(0,0,0), gross_neveu_start(0,0,0))   
data_tnr = []
@info "Finalizing beginning"
push!(data_tnr, scheme.finalize!(scheme))

stopping_criterion_tnr = convcrit(1e-20, trg_f)&maxiter(9)

steps = 0
crit = true
while crit
    @info "Step $(steps + 1), data_tnr[end]: $(!isempty(data_tnr) ? data_tnr[end] : "empty")"
    step!(scheme, 16, 100, 1e-20, 50, 1e-5)
    push!(data_tnr, scheme.finalize!(scheme))
    steps += 1
    crit = stopping_criterion_tnr(steps, data_tnr)
end
lnz_tnr = 0
for (i,d) in enumerate(data_tnr)
    lnz_tnr += log(d) * 2.0^(-i)
end
@show lnz_tnr