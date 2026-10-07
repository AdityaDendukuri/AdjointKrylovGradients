using LinearAlgebra, SparseArrays, Statistics, Random, Printf, CSV, DataFrames
using ExponentialUtilities: expv
const ROOT = get(ENV, "CME_ROOT", normpath(joinpath(@__DIR__, "..")))
include(joinpath(ROOT,"src/shared_krylov.jl"))
include(joinpath(ROOT,"experiments/optimized_shared_krylov_utils.jl"))
BLAS.set_num_threads(1)
const OUT = get(ENV,"LIBRARY_ACCURACY_OUT",joinpath(ROOT,"benchmark_results/sisc_library_accuracy_20260911.csv"))
const REPEATS = 7
n=320; dt=0.3
rng=MersenneTwister(20260911)
lo=15 .+ 5 .* rand(rng,n-1); hi=12 .+ 5 .* rand(rng,n-1)
A=spdiagm(-1=>lo, 1=>hi, 0=>-vcat(lo,0).-vcat(0,hi))
v=rand(rng,n); v/=norm(v)
u=randn(rng,n); u/=norm(u)
# Distinct local edge-rate perturbations; fixed generator and seeds across M.
all_dirs=SparseMatrixCSC{Float64,Int}[]
for j in randperm(rng,2(n-1))[1:512]
    col = j<=n-1 ? j : j-(n-1)+1
    row = j<=n-1 ? col+1 : col-1
    push!(all_dirs,sparse([row,col],[col,col],[1.0,-1.0],n,n))
end
# Dense adjoint Frechet reference: no Arnoldi or quadrature approximation.
Z=zeros(n,n)
G=exp(dt .* [Matrix(A') u*v'; Z Matrix(A')])[1:n,n+1:2n]
reference_all=[dot(G,D) for D in all_dirs]
yref=exp(dt*Matrix(A))*v; sref=exp(dt*Matrix(A'))*u
# Independent directional dense-block cross-check of the adjoint identity.
for j in (1,257,512)
    gj=dot(u,exp(dt .* [Matrix(A) Matrix(all_dirs[j]); Z Matrix(A)])[1:n,n+1:2n]*v)
    @assert isapprox(gj,reference_all[j];atol=2e-14,rtol=2e-11)
end
rel(x,y)=norm(x-y)/max(norm(y),eps())
function measure(f)
    f()
    ts=[(@elapsed f())*1000 for _ in 1:REPEATS]
    (median(ts),quantile(ts,.25),quantile(ts,.75))
end
rows=NamedTuple[]
for M in (16,64,256,512)
    dirs=all_dirs[1:M]; ref=reference_all[1:M]
    tbuild=@elapsed blocks=[[A D;spzeros(n,n) A] for D in dirs]
    seed=vcat(zeros(n),v)
    for m in (12,20,30,40)
        function directional()
            g=[dot(u,view(expv(dt,B,seed;m=m,tol=1e-14),1:n)) for B in blocks]
            (parameter_vjp=g,prediction=expv(dt,A,v;m=m,tol=1e-14),state_vjp=expv(dt,A',u;m=m,tol=1e-14))
        end
        r=directional(); pe=rel(r.parameter_vjp,ref); se=rel(r.state_vjp,sref); fe=rel(r.prediction,yref)
        med,q25,q75=measure(directional)
        push!(rows,(M=M,n=n,dt=dt,method="directional_block",m=m,nq=0,param_error=pe,state_error=se,forward_error=fe,median_ms=med,q25_ms=q25,q75_ms=q75,setup_ms=tbuild*1000,repeats=REPEATS))
        for nq in (8,16,24)
            st=@elapsed ws=OptimizedQuadratureWorkspace(dirs,n,m,nq)
            nodes,weights=gauss_legendre_01(nq)
            f=()->optimized_linear_loss_and_vjp!(ws,A,v,u,dt,nodes,weights;m_krylov=m)
            r=f(); pe=rel(r.parameter_vjp,ref); se=rel(r.state_vjp,sref); fe=rel(r.prediction,yref)
            med,q25,q75=measure(f)
            push!(rows,(M=M,n=n,dt=dt,method="shared_quadrature",m=m,nq=nq,param_error=pe,state_error=se,forward_error=fe,median_ms=med,q25_ms=q25,q75_ms=q75,setup_ms=st*1000,repeats=REPEATS))
            # Existing unfused primitive plus independently evaluated endpoints.
            # Retains redundant basis work for these endpoint actions.
            direct=()->(parameter_vjp=krylov_frechet_vjp(A,dirs,v,dt,u;m_krylov=m,n_quad=nq),prediction=expv(dt,A,v;m=m,tol=1e-14),state_vjp=expv(dt,A',u;m=m,tol=1e-14))
            r=direct(); pe=rel(r.parameter_vjp,ref); se=rel(r.state_vjp,sref); fe=rel(r.prediction,yref)
            med,q25,q75=measure(direct)
            push!(rows,(M=M,n=n,dt=dt,method="shared_direct",m=m,nq=nq,param_error=pe,state_error=se,forward_error=fe,median_ms=med,q25_ms=q25,q75_ms=q75,setup_ms=0.0,repeats=REPEATS))
        end
        @printf("M=%d m=%d finished\n",M,m); flush(stdout)
        CSV.write(OUT,DataFrame(rows))
    end
end
df=DataFrame(rows); front=NamedTuple[]
for M in (16,64,256,512), tol in (1e-6,1e-8), method in ("directional_block","shared_quadrature","shared_direct")
    sub=filter(r->r.M==M && r.method==method && max(r.param_error,r.state_error,r.forward_error)<=tol,df)
    isempty(sub) && error("No qualifying configuration: $M $tol $method")
    r=sub[argmin(sub.median_ms),:]
    push!(front,(M=M,tolerance=tol,method=method,m=r.m,nq=r.nq,median_ms=r.median_ms,max_error=max(r.param_error,r.state_error,r.forward_error)))
end
CSV.write(replace(OUT,".csv"=>"_frontier.csv"),DataFrame(front))
show(stdout,DataFrame(front);allrows=true,allcols=true);println()
