
import GemmDenseCUBLAS

function julia_main(args)::Cint
    GemmDenseCUBLAS.main(args)
    #GemmDenseCUBLAS.main64(args)
    #GemmDenseCUBLAS.main16(args)
    return 0
end

julia_main(ARGS)