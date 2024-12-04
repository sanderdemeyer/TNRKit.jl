mutable struct Loop_TNR <: TRGScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function TRG(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end