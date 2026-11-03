using LinearAlgebra, SparseArrays, Statistics, Printf
using CairoMakie, CSV, DataFrames, Zygote

const CUDA_LOADED = try
    @eval using CUDA
    true
catch
    false
end

include("../src/shared_krylov.jl")
include("../src/fpe.jl")

const HAS_GPU = CUDA_LOADED && CUDA.functional()
HAS_GPU && println("GPU: ", CUDA.name(CUDA.CuDevice(0)))

set_theme!(theme_latexfonts())
update_theme!(
    fontsize = 10,
    Axis = (
        spinewidth        = 0.7,
        xgridwidth        = 0.4,
        ygridwidth        = 0.4,
        xgridcolor        = (:black, 0.10),
        ygridcolor        = (:black, 0.10),
        topspinevisible   = false,
        rightspinevisible = false,
    ),
    Lines = (linewidth = 1.8,),
)

const BLUE   = RGBf(0.000, 0.447, 0.698)
const ORANGE = RGBf(0.835, 0.369, 0.000)
const GREEN  = RGBf(0.000, 0.620, 0.451)
const MGRAY  = RGBf(0.50,  0.50,  0.50)

const L_DOM = 2.0
const D_DIFF = 0.5
const DT = 0.3
const T_OBS = 8

function build_problem(n, M)
    xs = collect(range(-L_DOM, L_DOM, length=n))
    drift_fns = [let k = j, L = L_DOM; x -> (x / L)^k end for j in 1:M]
    A0, dirs = fpe_library_1d(xs, drift_fns; D = D_DIFF)

    θ = zeros(M)
    θ[1] = -2 * L_DOM
    M >= 3 && (θ[3] = 0.5 * L_DOM^3)
    A_true = A0 + sum(θ[j] * dirs[j] for j in 1:M)

    p0 = exp.(-0.5 .* xs.^2)
    p0 ./= sum(p0)
    expA = exp(Matrix(A_true) * DT)
    p_data = Vector{Vector{Float64}}(undef, T_OBS + 1)
    p_data[1] = p0
    for t in 1:T_OBS
        pn = max.(expA * p_data[t], 0.0)
        pn ./= sum(pn)
        p_data[t + 1] = pn
    end
    dts = fill(DT, T_OBS)
    return A_true, dirs, p_data, dts
end

# ── Dense CPU baseline: explicit block-matrix Fréchet (no AD required) ────────
# Same mathematics as the Zygote path but implemented analytically.
# G = exp([[Aᵀdt, dt·r⊗p_t], [0, Aᵀdt]])[1:n, n+1:end]  is the Fréchet kernel.
function dense_frechet_gradient(A_sparse, dirs, p_data, dts)
    n    = size(A_sparse, 1)
    A    = Matrix(A_sparse)
    At   = A'
    Z    = zeros(n, n)
    grad = zeros(length(dirs))

    for t in eachindex(dts)
        dt     = dts[t]
        p_t    = p_data[t];  p_next = p_data[t+1]
        expAdt = exp(A * dt)
        r      = p_next - expAdt * p_t
        # The loss cotangent is p_pred - p_next = -r.
        X      = -dt .* (r * p_t')         # n×n outer product
        B      = [At*dt  X; Z  At*dt]      # 2n×2n block matrix
        G      = exp(B)[1:n, n+1:end]      # Fréchet kernel

        for j in eachindex(dirs)
            rows, cols, vals = findnz(dirs[j])
            s = 0.0
            @inbounds for k in eachindex(rows)
                s += G[rows[k], cols[k]] * vals[k]
            end
            grad[j] += s
        end
    end
    grad
end

# Note: Julia's exp(CuArray) requires scalar indexing (Higham Padé uses pivoting
# that CUDA.jl can't vectorise). GPU dense Fréchet is covered by the Python
# baselines (PyTorch GPU, JAX GPU) — see benchmark_dense_vjp.py.

function fd_gradient_krylov(A, dirs, p_data, dts; m_krylov = 25, ε = 1e-5)
    M = length(dirs)
    _, loss0 = shared_gradient(A, [dirs[1]], p_data, dts; m_krylov = m_krylov)
    grad = zeros(M)
    for j in 1:M
        A_p = A + ε * dirs[j]
        _, loss_p = shared_gradient(A_p, [dirs[j]], p_data, dts; m_krylov = m_krylov)
        grad[j] = (loss_p - loss0) / ε
    end
    return grad
end

time_ms(f; n_trials = 5) = median(begin
    t0 = time_ns()
    f()
    (time_ns() - t0) * 1e-6
end for _ in 1:n_trials)

function benchmark_n_sweep(;
        n_vals_cpu = [20, 40, 80, 150, 250, 400],
        n_vals_gpu = [200, 500, 1000, 2000, 4000],
        M_fixed = 10, n_trials = 7)

    println("\n=== Julia CPU n-sweep (M=$M_fixed): Shared-Krylov vs dense Fréchet ===")
    @printf("%-6s  %-14s  %-14s\n", "n", "shared-CPU (ms)", "dense-CPU (ms)")
    cpu_rows = NamedTuple[]
    for n in n_vals_cpu
        A, dirs, p_data, dts = build_problem(n, M_fixed)
        shared_gradient(A, dirs, p_data, dts; m_krylov=30)
        dense_frechet_gradient(A, dirs, p_data, dts)

        t_sk  = time_ms(() -> shared_gradient(A, dirs, p_data, dts; m_krylov=30); n_trials=n_trials)
        t_den = time_ms(() -> dense_frechet_gradient(A, dirs, p_data, dts);        n_trials=n_trials)
        @printf("%-6d  %-14.2f  %-14.2f\n", n, t_sk, t_den)
        push!(cpu_rows, (n=n, shared_ms=t_sk, dense_cpu_ms=t_den))
        flush(stdout)
    end

    # Large-n sweep: Shared-Krylov (all n) + dense Fréchet CPU (up to 1000 only)
    # Dense CPU at n>1000 would take minutes per trial; shown as NaN above that.
    cpu_max_n = 1000
    gpu_rows = NamedTuple[]
    println("\n=== Julia large-n sweep (M=$M_fixed): Shared-Krylov + dense CPU baselines ===")
    @printf("%-6s  %-14s  %-14s\n", "n", "shared-CPU (ms)", "dense-CPU (ms)")
    for n in n_vals_gpu
        A, dirs, p_data, dts = build_problem(n, M_fixed)
        shared_gradient(A, dirs, p_data, dts; m_krylov=30)  # warmup

        t_sk  = time_ms(() -> shared_gradient(A, dirs, p_data, dts; m_krylov=30); n_trials=min(n_trials, 5))
        t_den = n ≤ cpu_max_n ?
            time_ms(() -> dense_frechet_gradient(A, dirs, p_data, dts); n_trials=min(n_trials, 3)) :
            NaN
        @printf("%-6d  %-14.2f  %-14s\n", n, t_sk, isnan(t_den) ? "—" : @sprintf("%.2f", t_den))
        push!(gpu_rows, (n=n, shared_ms=t_sk, dense_cpu_ms=t_den))
        flush(stdout)
    end

    DataFrame(cpu_rows), DataFrame(gpu_rows)
end

function benchmark_m_sweep(; M_vals = [10, 25, 50, 100, 150, 200], n_fixed = 80, n_trials = 5)
    println("\n=== Julia M-sweep (n=$n_fixed): Shared-Krylov vs dense Fréchet ===")
    @printf("%-6s  %-14s  %-14s  %-12s\n", "M", "shared-CPU (ms)", "dense-CPU (ms)", "FD (ms)")
    rows = NamedTuple[]
    for M in M_vals
        A, dirs, p_data, dts = build_problem(n_fixed, M)
        shared_gradient(A, dirs, p_data, dts; m_krylov=30)
        dense_frechet_gradient(A, dirs, p_data, dts)

        t_sk  = time_ms(() -> shared_gradient(A, dirs, p_data, dts; m_krylov=30); n_trials=n_trials)
        t_den = time_ms(() -> dense_frechet_gradient(A, dirs, p_data, dts);        n_trials=n_trials)
        t_fd  = time_ms(() -> fd_gradient_krylov(A, dirs, p_data, dts; m_krylov=25); n_trials=min(3, n_trials))
        @printf("%-6d  %-14.2f  %-14.2f  %-12.2f\n", M, t_sk, t_den, t_fd)
        push!(rows, (M=M, shared_ms=t_sk, dense_cpu_ms=t_den, fd_ms=t_fd))
        flush(stdout)
    end
    DataFrame(rows)
end

function _col(df, name)
    hasproperty(df, Symbol(name)) ? Float64.(getproperty(df, Symbol(name))) : nothing
end

# Plot a series, skipping NaN/Inf
function _line!(ax, xs, ys, col; lw=1.8, ls=:solid, mkr=:circle, ms=6, lbl=nothing)
    ys === nothing && return
    mask = isfinite.(ys)
    any(mask) || return
    kw = lbl === nothing ? (;) : (label=lbl,)
    lines!(ax, xs[mask], ys[mask]; color=col, linewidth=lw, linestyle=ls, kw...)
    scatter!(ax, xs[mask], ys[mask]; color=col, marker=mkr, markersize=ms)
end

function make_figure(df_py_n, df_py_m, df_jl_n, df_jl_m,
                    df_jl_ngpu=nothing, df_py_ngpu=nothing)
    fig = Figure(size = (680, 260), figure_padding = (4, 10, 4, 4))

    # ── (a) CPU n-sweep (moderate n, show O(n) vs O(n³) scaling) ─────
    ax_n = Axis(fig[1, 1],
        xlabel = L"State-space size $n$",
        ylabel = "Gradient wall time (ms)",
        title = "(a) CPU scaling  (M=10)",
        titlealign = :left, titlegap = 4,
        xscale = log10, yscale = log10,
        xticks = ([20, 40, 80, 150, 250, 400], ["20","40","80","150","250","400"]),
        yticks = ([0.1, 1, 10, 100, 1000], ["0.1","1","10","100","1000"]))

    # ── (a) CPU n-sweep ───────────────────────────────────────────────
    jl_n = Float64.(df_jl_n.n)
    py_n = Float64.(df_py_n.n)
    i80  = findfirst(==(80), df_jl_n.n)
    ref  = [jl_n[1], jl_n[end]]

    lines!(ax_n, ref, df_jl_n.shared_ms[i80] .* (ref ./ 80);
        color=(MGRAY,0.3), linewidth=0.8, linestyle=:dash)
    text!(ax_n, ref[end]*0.72, df_jl_n.shared_ms[i80]*(ref[end]/80)*1.9;
        text=L"O(n)", color=(MGRAY,0.55), fontsize=7.5)
    lines!(ax_n, ref, df_jl_n.dense_cpu_ms[i80] .* (ref ./ 80).^3;
        color=(MGRAY,0.3), linewidth=0.8, linestyle=:dot)
    text!(ax_n, ref[end]*0.52, df_jl_n.dense_cpu_ms[i80]*(ref[end]/80)^3*0.22;
        text=L"O(n^3)", color=(MGRAY,0.55), fontsize=7.5)

    _line!(ax_n, py_n, _col(df_py_n,"pytorch_ms"),  ORANGE;
        lw=2.0, mkr=:rect, ms=7, lbl="PyTorch CPU (eager)")
    _line!(ax_n, py_n, _col(df_py_n,"jax_scan_ms"), GREEN;
        lw=1.6, mkr=:utriangle, ms=6, lbl="JAX CPU (compiled)")
    _line!(ax_n, jl_n, df_jl_n.dense_cpu_ms, MGRAY;
        lw=1.5, ls=:dashdot, mkr=:diamond, ms=6, lbl="Dense CPU (Julia)")
    _line!(ax_n, jl_n, df_jl_n.shared_ms, BLUE;
        lw=2.5, ms=8, lbl="Shared-Krylov (ours)")
    axislegend(ax_n; position=:lt, framevisible=false, labelsize=7.5, rowgap=1)

    # ── (b) M-sweep ────────────────────────────────────────────────────
    ax_m = Axis(fig[1, 2],
        xlabel = L"Library size $M$",
        ylabel = "Gradient wall time (ms)",
        title = "(b) library-size scaling  (n=80)",
        titlealign = :left, titlegap = 4,
        xscale = log10, yscale = log10,
        xticks = ([10, 25, 50, 100, 200], ["10","25","50","100","200"]),
        yticks = ([0.1, 1, 10, 100, 1000], ["0.1","1","10","100","1000"]))

    py_m = Float64.(df_py_m.M)
    jl_m = Float64.(df_jl_m.M)

    _line!(ax_m, py_m, _col(df_py_m,"pytorch_ms"),  ORANGE;
        lw=2.0, mkr=:rect, ms=7, lbl="PyTorch CPU (eager)")
    _line!(ax_m, py_m, _col(df_py_m,"jax_scan_ms"), GREEN;
        lw=1.6, mkr=:utriangle, ms=6, lbl="JAX CPU (compiled)")
    _line!(ax_m, jl_m, df_jl_m.dense_cpu_ms, MGRAY;
        lw=1.5, ls=:dashdot, mkr=:diamond, ms=6, lbl="Dense CPU (Julia)")
    _line!(ax_m, jl_m, df_jl_m.shared_ms, BLUE;
        lw=2.5, ms=8, lbl="Shared-Krylov (ours)")
    axislegend(ax_m; position=:lt, framevisible=false, labelsize=7.5, rowgap=1)

    return fig
end

function main()
    parse_ints(name, default) = parse.(Int, split(get(ENV, name, default), ","))
    out_dir = get(ENV, "GRAD_BENCH_OUTPUT_DIR", joinpath("paper", "figures"))
    jl_n_path = joinpath(out_dir, "benchmark_julia_n.csv")
    jl_m_path = joinpath(out_dir, "benchmark_julia_m.csv")
    rebuild_only = get(ENV, "GRAD_BENCH_REBUILD_ONLY", "0") == "1"
    run_large = get(ENV, "GRAD_BENCH_RUN_LARGE", "1") == "1"
    make_plot = get(ENV, "GRAD_BENCH_MAKE_PLOT", "1") == "1"

    jl_ngpu_path = joinpath(out_dir, "benchmark_julia_n_gpu.csv")

    if rebuild_only
        isfile(jl_n_path)    || error("Missing $jl_n_path.")
        isfile(jl_m_path)    || error("Missing $jl_m_path.")
        df_jl_n   = CSV.read(jl_n_path, DataFrame)
        df_jl_m   = CSV.read(jl_m_path, DataFrame)
        df_jl_ngpu = isfile(jl_ngpu_path) ? CSV.read(jl_ngpu_path, DataFrame) : nothing
    else
        n_cpu = parse_ints("GRAD_BENCH_N_CPU", "20,40,80,150,250,400")
        n_large = run_large ?
            parse_ints("GRAD_BENCH_N_LARGE", "200,500,1000,2000,4000") :
            Int[]
        m_vals = parse_ints("GRAD_BENCH_M", "10,25,50,100,150,200")
        n_trials = parse(Int, get(ENV, "GRAD_BENCH_TRIALS", "7"))
        df_jl_n, df_jl_ngpu = benchmark_n_sweep(
            n_vals_cpu=n_cpu,
            n_vals_gpu=n_large,
            n_trials=n_trials,
        )
        df_jl_m = benchmark_m_sweep(M_vals=m_vals, n_trials=n_trials)
        mkpath(out_dir)
        CSV.write(jl_n_path, df_jl_n)
        CSV.write(jl_m_path, df_jl_m)
        nrow(df_jl_ngpu) > 0 && CSV.write(jl_ngpu_path, df_jl_ngpu)
        println("Saved Julia CSV benchmarks.")
    end

    make_plot || return
    py_n_path    = joinpath(out_dir, "benchmark_python_n.csv")
    py_ngpu_path = joinpath(out_dir, "benchmark_python_n_gpu.csv")
    py_m_path    = joinpath(out_dir, "benchmark_python_m.csv")
    isfile(py_n_path) || error("Missing $py_n_path. Run benchmark_dense_vjp.py first.")
    isfile(py_m_path) || error("Missing $py_m_path. Run benchmark_dense_vjp.py first.")
    df_py_n    = CSV.read(py_n_path, DataFrame)
    df_py_m    = CSV.read(py_m_path, DataFrame)
    df_py_ngpu = isfile(py_ngpu_path) ? CSV.read(py_ngpu_path, DataFrame) : nothing

    fig = make_figure(df_py_n, df_py_m, df_jl_n, df_jl_m, df_jl_ngpu, df_py_ngpu)
    save(joinpath(out_dir, "fig_gradient_benchmark.pdf"), fig, pt_per_unit = 1)
    save(joinpath(out_dir, "fig_gradient_benchmark.png"), fig, px_per_unit = 2)
    println("Saved paper/figures/fig_gradient_benchmark.{pdf,png}")
end

main()
