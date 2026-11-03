# Replot saved gradient and application results in monochrome.
# julia --project=experiments experiments/plot_sisc_gradient_bw.jl
using CairoMakie, CSV, DataFrames, Statistics
const ROOT=normpath(joinpath(@__DIR__, ".."))
const DATA=joinpath(ROOT,"paper/figures")
const OUT=get(ENV, "SISC_FIGURE_DIR", joinpath(ROOT,"paper","figures"))
mkpath(OUT)
set_theme!(theme_latexfonts())
update_theme!(fontsize=15,Axis=(topspinevisible=false,rightspinevisible=false,xgridvisible=false,ygridvisible=false),Lines=(linewidth=2,))
readcsv(n)=CSV.read(joinpath(DATA,n*".csv"),DataFrame)
styles=[:solid,:dash,:dot,:dashdot]; markers=[:circle,:rect,:utriangle,:diamond]
function curve(ax,x,y,k,label)
 scatterlines!(ax,x,y,color=:black,linestyle=styles[mod1(k,4)],marker=markers[mod1(k,4)],markersize=7,label=label)
end
savefig(n,f)=save(joinpath(OUT,n*".pdf"),f)
# Matfree portable same-process GPU sweep
p=CSV.read(joinpath(ROOT,"benchmark_results/portable_iso/portable_iso_rtx4070super_gpu.csv"),DataFrame)
f=Figure(size=(1100,340))
for (i,tol) in enumerate([1e-6,1e-8,1e-10])
 ax=Axis(f[1,i],xlabel="State dimension n",ylabel="Median gradient time (ms)",xscale=log10,yscale=log10,xticks=([100,1000,10000],["100","1,000","10,000"]),yticks=([1,2,5,10],["1","2","5","10"]),title="($(Char(96+i))) tolerance $(tol)")
 df=filter(r->r.max_vjp_relerr<=tol,p)
 println("GPU methods ",unique(df.method))
 for (j,method) in enumerate(["JAX_dense_AD","Matfree_finite_Arnoldi_VJP","Structured_semigroup_core","Structured_semigroup_quadrature"])
  sub=sort(combine(groupby(filter(r->r.method==method,df),:n),:runtime_median_ms=>minimum=>:runtime_median_ms),:n);curve(ax,sub.n,sub.runtime_median_ms,j,replace(method,"JAX_dense_AD"=>"Dense AD","Matfree_finite_Arnoldi_VJP"=>"Matfree","Structured_semigroup_core"=>"Shared core","Structured_semigroup_quadrature"=>"Shared quadrature"))
 end
 i==1 && axislegend(ax,position=:lt,labelsize=10)
end
savefig("gradient_matfree_bw",f)
# Explicitly separate quadrature and Krylov errors
p=readcsv("gradient_accuracy");f=Figure(size=(1000,340))
for (i,system) in enumerate(["FPE","Lindblad"])
 ax=Axis(f[1,i],xlabel="Krylov depth m",ylabel="Relative parameter-gradient error",yscale=log10,title="($(Char(96+i))) $system")
 df=filter(r->r.system==system && r.dt==(system=="FPE" ? 0.1 : 1.0),p)
 for (j,nq) in enumerate([5,8,16,24])
  sub=sort(filter(r->r.n_quad==nq,df),:m);isempty(sub) && continue
  curve(ax,sub.m,max.(sub.parameter_vjp_relerr,1e-17),j,"quadrature $(nq)")
 end
 if system=="FPE"
  # Higher-order sweep reported in Table 1 of the paper.
  curve(ax,[20,30,40,60],[1.0e-5,1.1e-8,1.7e-13,1.8e-15],3,"quadrature 24")
  curve(ax,[20,30,40,60],[1.0e-5,1.1e-8,1.7e-13,2.0e-15],4,"quadrature 32")
 end
 axislegend(ax,position=:rt,labelsize=12)
end
savefig("gradient_accuracy_bw",f)
# Neural applications: recorded context fits and grouped support
f=Figure(size=(1050,680));p=readcsv("brusselator_gates")
ax=Axis(f[1,1],xlabel="Candidate direction",ylabel="Channel simplex weight",title="(a) Brusselator: grouped selection")
barplot!(ax,p.dir,p.pi_r,color=:white,strokecolor=:black,strokewidth=0.8)
r=filter(r->r.is_true,p);scatter!(ax,r.dir,r.pi_r,marker=:star5,color=:black,markersize=13,label="True directions");axislegend(ax,position=:rc,labelsize=11)
p=readcsv("brusselator_rates");ax=Axis(f[1,2],xlabel="Held-out context index",ylabel="Inferred / true rate",title="(b) Brusselator: rate recovery")
hlines!(ax,[1.],color=:gray,linestyle=:dot)
for (i,k) in enumerate(["A","B","k2","k4"]);curve(ax,1:nrow(p),p[!,k*"_pred"]./p[!,k*"_true"],i,k);end
axislegend(ax,position=:lb,labelsize=11,nbanks=2)
p=readcsv("sl_recovery");ax=Axis(f[2,1],xlabel="Context μ",ylabel="Recovered coefficient",title="(c) Stuart–Landau: context learning")
curve(ax,p.mu_ctx,p.mu_pred,1,"μ inferred");lines!(ax,p.mu_ctx,p.mu_true,color=:black,linestyle=:dot,label="μ true")
curve(ax,p.mu_ctx,p.omega_pred,2,"ω inferred");hlines!(ax,[1.],color=:gray,linestyle=:dashdot,label="ω true");axislegend(ax,position=:lt,labelsize=11)
p=readcsv("lindblad_observation_compare");ax=Axis(f[2,2],xlabel="Qubits",ylabel="Held-out rate error (%)",title="(d) Lindblad: observation protocols",xticks=[2,3,4])
for (i,mode) in enumerate(["full_state","pauli_local"]);sub=sort(filter(r->r.obs_mode==mode,p),:n);curve(ax,sub.n,100sub.mean_param_err,i,mode=="full_state" ? "Full state" : "Partial observations");end
axislegend(ax,position=:lt,labelsize=11)
savefig("neural_results_bw",f)
# Semilinear trajectory: shared grayscale scale, error map distinct scale
p=readcsv("kuramoto_sivashinsky_composable_trajectory");println("KS columns ",names(p))
# Discrete support stability under Krylov/quadrature choices
p=CSV.read(joinpath(ROOT,"benchmark_results/support_stability.csv"),DataFrame)
f=Figure(size=(1000,340));ax=Axis(f[1,1],xlabel="Krylov depth",ylabel="Exact support fraction",title="(a) Support stability across settings")
for (i,nq) in enumerate(sort(unique(p.n_quad)))
 sub=filter(r->r.n_quad==nq,p);agg=sort(combine(groupby(sub,:m_krylov),:exact_support=>mean=>:rate),:m_krylov);curve(ax,agg.m_krylov,agg.rate,i,"quadrature $nq")
end
axislegend(ax,position=:rb,labelsize=11)
p=CSV.read(joinpath(ROOT,"benchmark_results/initializer_robustness.csv"),DataFrame);ax=Axis(f[1,2],xlabel="Initialization perturbation σ",ylabel="Exact support fraction",title="(b) Initialization sensitivity")
p.exact=(p.n_recovered.==p.true_support).&(p.false_positives.==0).&(p.lost.==0)
for (i,sys) in enumerate(unique(p.system));sub=filter(r->r.system==sys,p);agg=sort(combine(groupby(sub,:sigma),:exact=>mean=>:rate),:sigma);curve(ax,agg.sigma,agg.rate,i,sys);end
axislegend(ax,position=:lb,labelsize=11,nbanks=2)
savefig("support_controls_bw",f)
println("Completed monochrome figures")
# Dense versus sparse original backward sweeps
f=Figure(size=(1050,360))
for (i,suffix,xcol) in [(1,"n",:n),(2,"m",:M)]
 py=readcsv("benchmark_python_"*suffix);jl=readcsv("benchmark_julia_"*suffix)
 println("Dense columns ",names(py)," / ",names(jl))
 xc=Symbol(names(py)[1]);xj=Symbol(names(jl)[1])
 ax=Axis(f[1,i],xlabel=i==1 ? "State dimension n" : "Library directions M",ylabel="Backward time (ms)",xscale=log10,yscale=log10,title=i==1 ? "(a) Ten directions" : "(b) 80 states")
 for (j,col) in enumerate(names(py)[2:end]);curve(ax,py[!,xc],py[!,col],j,occursin("jax",col) ? "JAX dense (compiled)" : "PyTorch dense (eager)");end
 curve(ax,jl[!,xj],jl.shared_ms,3,"Julia shared sparse")
 if "dense_cpu_ms" in names(jl);curve(ax,jl[!,xj],jl.dense_cpu_ms,4,"Julia dense");end
 axislegend(ax,position=:lt,labelsize=11)
end
savefig("dense_scaling_bw",f)
# KS independent reference and fitted multi-step prediction as print-safe line profiles.
p=readcsv("kuramoto_sivashinsky_composable_trajectory")
xs=sort(unique(p.x));ts=sort(unique(p.time))
z=reshape(p.reference,length(xs),length(ts))
pred=reshape(p.prediction,length(xs),length(ts))
f=Figure(size=(1050,330))
for (i,tstar) in enumerate([0.125,0.25])
 k=argmin(abs.(ts .- tstar))
 ax=Axis(f[1,i],xlabel="Space x",ylabel="State u",
         title="($(Char(96+i))) t = $(ts[k])")
 lines!(ax,xs,z[:,k],color=:black,linewidth=2.2,label="RK4 reference")
 lines!(ax,xs,pred[:,k],color=:black,linewidth=1.8,linestyle=:dash,
        label="Learned prediction")
 scatter!(ax,xs[1:8:end],pred[1:8:end,k],color=:white,strokecolor=:black,
          strokewidth=1.1,marker=:circle,markersize=7)
 i==1 && axislegend(ax,position=:lb,labelsize=10)
end
relerr=[sqrt(sum(abs2,pred[:,k].-z[:,k]))/sqrt(sum(abs2,z[:,k]))
        for k in eachindex(ts)]
ax=Axis(f[1,3],xlabel="Time",ylabel="Relative L² error (%)",
        title="(c) Multi-step error")
lines!(ax,ts,100relerr,color=:black,linewidth=2.2)
scatter!(ax,ts[1:2:end],100relerr[1:2:end],color=:white,strokecolor=:black,
         strokewidth=1.1,marker=:diamond,markersize=7)
ylims!(ax,0,0.15)
savefig("ks_trajectory_bw",f)
