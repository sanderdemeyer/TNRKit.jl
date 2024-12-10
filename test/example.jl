using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

# stop when converged or after 50 steps, whichever comes first
<<<<<<< HEAD
stopping_criterion = convcrit(1e-20, trg_f)&maxiter(200)

β = 0
=======
stopping_criterion = convcrit(1e-16, trg_f)&maxiter(20)

# choose a TensorKit truncation scheme
trunc = truncdim(16)&truncbelow(1e-40)
>>>>>>> dev

# initialize the TRG scheme
scheme_trg = TRG(triangle_good(β))

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
<<<<<<< HEAD
data_trg = run!(scheme_trg, truncdim(16), stopping_criterion; finalize_beginning=false)
=======
data = run!(scheme, trunc, stopping_criterion; finalize_beginning=true)
>>>>>>> dev
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)
lnz_triangle_trg = 0
for (i, d) in enumerate(data_trg)
    lnz_triangle_trg += log(d) * 2.0^(-i)        
end
@show lnz_triangle_trg

# initialize the BTRG scheme
scheme_btrg = BTRG(triangle_good(β), -0.5)

# run the BTRG scheme
<<<<<<< HEAD
data_btrg = run!(scheme_btrg, truncdim(16), stopping_criterion; finalize_beginning=false)
lnz_triangle_btrg = 0
for (i, d) in enumerate(data_btrg)
    lnz_triangle_btrg += log(d) * 2.0^(-i)        
end
@show lnz_triangle_btrg
=======
data = run!(scheme, trunc, stopping_criterion)

# initialize the HOTRG scheme
scheme = HOTRG(classical_ising(1.0))

# run the HOTRG scheme
data = run!(scheme, trunc, stopping_criterion)
>>>>>>> dev
