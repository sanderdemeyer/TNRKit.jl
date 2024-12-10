module TRGKit
    using TensorKit, LinearAlgebra, OMEinsum
    using Printf

    # stop criteria
    include("utility/stopping.jl")
    export maxiter, convcrit

    # schemes
    include("schemes/trgscheme.jl")
    include("schemes/trg.jl")
    include("schemes/btrg.jl")
    include("schemes/hotrg.jl")
    include("schemes/Loop-TNR.jl")
    
    export TRGScheme, TRG, BTRG, HOTRG, trg_convcrit, btrg_convcrit, Loop_TNR
    export run!, step!, pseudopow, entanglement_filtering!

    # models
    include("models/ising.jl")
    export classical_ising, classical_ising_symmetric

    include("models/gross-neveu.jl")
    export gross_neveu_start

    include("models/trianguar-ising.jl")
    export triangle_bad, triangle_good, triangle_bad_2, triangle_bad_3

    include("models/Hubbard_2D.jl")
    export Hubbard2D_start
    # utility functions
    include("utility/cft.jl")
    export cft_data

    # export the different models (do we really want to ship those ourselves?)    
end 
