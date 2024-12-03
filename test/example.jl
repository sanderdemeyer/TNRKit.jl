using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-20, trg_f)&maxiter(200)

# initialize the TRG scheme
scheme = TRG(triangle_bad())

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
data = run!(scheme, truncdim(64), stopping_criterion; finalize_beginning=false)
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)
lnz_triangle_bad = 0
for (i, d) in enumerate(data)
    lnz_triangle_bad += log(d) * 2.0^(-i)        
end
@show lnz_triangle_bad

# initialize the BTRG scheme
scheme = BTRG(triangle_bad(1.0), -0.5)

# run the BTRG scheme
data = run!(scheme, truncdim(16), stopping_criterion; finalize_beginning=true)
lnz_triangle_bad = 0
for (i, d) in enumerate(data)
    lnz_triangle_bad += log(d) * 4.0^(1-i)        
end
@show lnz_triangle_bad