
import GemmDenseThreads

function julia_main(args)::Cint
    GemmDenseThreads.main(args)
    return 0
end

julia_main(ARGS)