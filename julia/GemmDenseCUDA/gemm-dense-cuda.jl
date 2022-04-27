
import GemmDenseCUDA

function julia_main(args)::Cint
    GemmDenseCUDA.main(args)
    return 0
end

julia_main(ARGS)