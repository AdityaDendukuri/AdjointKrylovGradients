using CairoMakie, CSV, DataFrames
const ROOT=get(ENV,"CME_ROOT",normpath(joinpath(@__DIR__,"..")))
const INPUT=get(ENV,"LIBRARY_FRONTIER",joinpath(ROOT,"benchmark_results/sisc_library_accuracy_20260911_frontier.csv"))
const OUTPUT=get(ENV,"LIBRARY_FIGURE",joinpath(ROOT,"paper/figures/library_accuracy_bw.pdf"))
p=CSV.read(INPUT,DataFrame)
set_theme!(theme_latexfonts())
update_theme!(fontsize=17,Axis=(topspinevisible=false,rightspinevisible=false,xgridvisible=false,ygridvisible=false))
f=Figure(size=(1050,410))
for (i,tol) in enumerate((1e-6,1e-8))
 ax=Axis(f[1,i],xlabel="Distinct coefficient directions M",ylabel="Median evaluation time (ms)",xscale=log2,yscale=log10,xticks=([16,64,256,512],["16","64","256","512"]),title="Tolerance $(tol)")
 for (j,(method,label)) in enumerate((("directional_block","Per-direction block"),("shared_direct","Shared, direct contraction"),("shared_quadrature","Shared, fused contraction")))
  sub=sort(filter(r->r.method==method && r.tolerance==tol,p),:M)
  scatterlines!(ax,sub.M,sub.median_ms,color=:black,marker=[:circle,:rect,:utriangle][j],linestyle=[:solid,:dash,:dot][j],markersize=9,linewidth=2,label=label)
 end
end
Legend(f[2,1:2],f.content[1],orientation=:horizontal,labelsize=15,framevisible=false)
save(OUTPUT,f)
