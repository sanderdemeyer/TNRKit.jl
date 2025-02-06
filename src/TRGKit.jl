module TRGKit
    using TensorKit, LinearAlgebra, OMEinsum, KrylovKit
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
    include("schemes/hotrg_robust.jl")
    include("schemes/boundary_trg.jl")
    include("schemes/Anisotropic_trg.jl")
    
    export TRGScheme, TRG, BTRG, ATRG, ATRG_3D, HOTRG, HOTRG_robust, trg_convcrit, btrg_convcrit, Loop_TNR, Boundary_TRG, boundary_subroutine, bulk_subroutine
    export run!, step!, pseudopow, entanglement_filtering!, find_projectors, make_psi

    # models
    include("models/ising.jl")
    export classical_ising, classical_ising_symmetric

    include("models/gross-neveu.jl")
    export gross_neveu_start

    include("models/trianguar-ising.jl")
    export triangle_bad, triangle_good, triangle_bad_2, triangle_bad_3

    include("models/Hubbard_2D.jl")
    export Hubbard2D_start, Hubbard2D
    # utility functions
    include("utility/cft.jl")
    export cft_data

    # export the different models (do we really want to ship those ourselves?)    
end 
