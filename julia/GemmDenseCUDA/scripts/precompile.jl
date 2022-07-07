
using PackageCompiler
create_sysimage([:CUDA, :GemmDenseCUDA], sysimage_path = "GemmDenseCUDA_jll.so")
exit()
