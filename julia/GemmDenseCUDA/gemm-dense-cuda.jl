
import GemmDenseCUDA

function julia_main(args)::Cint
    GemmDenseCUDA.main(args)
    #GemmDenseCUDA.main64(args)
    #GemmDenseCUDA.main16(args)
    return 0
end

julia_main(ARGS)