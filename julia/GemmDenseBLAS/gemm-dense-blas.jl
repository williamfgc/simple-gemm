
import GemmDenseBLAS

function julia_main(args)::Cint
    GemmDenseBLAS.main(args)
    return 0
end

julia_main(ARGS)