
import GemmDenseThreads

function julia_main(args)::Cint
    GemmDenseThreads.main(args)
    #GemmDenseThreads.main64(args)
    #GemmDenseThreads.main16(args)
    return 0
end

julia_main(ARGS)