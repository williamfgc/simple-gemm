
import GemmDenseAMDGPU

function julia_main(args)::Cint
    GemmDenseAMDGPU.main(args)
    #GemmDenseAMDGPU.main64(args)
    #GemmDenseAMDGPU.main16(args)
    return 0
end

julia_main(ARGS)