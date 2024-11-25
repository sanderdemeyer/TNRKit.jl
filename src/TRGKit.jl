module TRGKit
    using TensorKit, LinearAlgebra
    using Printf

    # stop criteria
    include("utility/stopping.jl")
    export maxiter, convcrit

    # schemes
    include("schemes/trgscheme.jl")
    include("schemes/trg.jl")
    include("schemes/btrg.jl")
    include("schemes/hotrg.jl")
    
    export TRG, BTRG, HOTRG, trg_convcrit, btrg_convcrit
    export run!

    # models
    # include("models/ising.jl")

    # export the different models (do we really want to ship those ourselves?)    
end 
