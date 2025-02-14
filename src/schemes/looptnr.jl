mutable struct Loop_TNR <: TNRScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function Loop_TNR(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end