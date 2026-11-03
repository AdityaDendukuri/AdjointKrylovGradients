# =============================================================================
# SIAM Black-and-White Figure Style
#
# All figures in this paper target SIAM journal specs:
#   Single column : 3.40 in (245 pt)
#   Double column : 6.85 in (494 pt)
#   Max height    : 9.00 in (648 pt)
#   Font          : 9 pt body / 8 pt captions
#
# Color rules: NO color.  Distinguish series by:
#   line style  : solid, dashed, dotted, dash-dot
#   marker      : circle, square, triangle-up, diamond
#   gray level  : black (0.0), dark (0.25), mid (0.55), light (0.75)
#   fill        : open vs. filled markers
# =============================================================================

using CairoMakie

# ── Dimensions (points at 72 dpi) ───────────────────────────────────────────
const SW1 = 245          # single-column width  (pt)
const SW2 = 494          # double-column width  (pt)
const SH_HALF  = 200     # comfortable half-height
const SH_FULL  = 380     # comfortable full-height (golden-ratio-ish for SW2)
const SH_TALL  = 520

# ── Font sizes ───────────────────────────────────────────────────────────────
const FS_BODY  = 9       # axis labels, tick labels
const FS_SMALL = 8       # legends, annotations
const FS_TITLE = 9       # subplot titles (bold)

# ── Gray levels (0 = black, 1 = white) ──────────────────────────────────────
const GK = RGBf(0.00, 0.00, 0.00)   # black
const GD = RGBf(0.25, 0.25, 0.25)   # dark gray
const GM = RGBf(0.50, 0.50, 0.50)   # mid gray
const GL = RGBf(0.72, 0.72, 0.72)   # light gray

# ── Line styles ──────────────────────────────────────────────────────────────
const LS1 = :solid
const LS2 = :dash
const LS3 = :dot
const LS4 = :dashdot

# ── Marker shapes ────────────────────────────────────────────────────────────
const MK1 = :circle
const MK2 = :rect
const MK3 = :utriangle
const MK4 = :diamond

# ── Convenience: 4-series style bundles ─────────────────────────────────────
#   S1-S4  : for scatterlines!(ax, x, y; S1..., label="...")
#   L1-L4  : for lines!(ax, x, y; L1..., label="...")
#   SC1-SC4: for scatter!(ax, x, y; SC1..., label="...")
const S1 = (color=GK, linestyle=LS1, marker=MK1, linewidth=1.6, markersize=5)
const S2 = (color=GD, linestyle=LS2, marker=MK2, linewidth=1.6, markersize=5)
const S3 = (color=GM, linestyle=LS3, marker=MK3, linewidth=1.6, markersize=5)
const S4 = (color=GL, linestyle=LS4, marker=MK4, linewidth=1.6, markersize=5)

# Line-only (no marker attributes)
const L1 = (color=GK, linestyle=LS1, linewidth=1.6)
const L2 = (color=GD, linestyle=LS2, linewidth=1.6)
const L3 = (color=GM, linestyle=LS3, linewidth=1.6)
const L4 = (color=GL, linestyle=LS4, linewidth=1.6)

# Scatter-only (no line attributes)
const SC1 = (color=GK, marker=MK1, markersize=6)
const SC2 = (color=GD, marker=MK2, markersize=6)
const SC3 = (color=GM, marker=MK3, markersize=6)
const SC4 = (color=GL, marker=MK4, markersize=6)

# bars / histograms
const B1 = (color=GK,               strokecolor=GK, strokewidth=0.8)
const B2 = (color=(:white,0),       strokecolor=GD, strokewidth=1.2)   # open/outline
const B3 = (color=(GM,0.5),         strokecolor=GM, strokewidth=0.8)
const B4 = (color=(GL,0.5),         strokecolor=GL, strokewidth=0.8)

# step-histogram style (no fill, just outline)
const STEP_DATA = (color=GK, linewidth=1.2)
const STEP_CME  = (color=GD, linewidth=1.4, linestyle=LS2)
const STEP_MC   = (color=GM, linewidth=1.4, linestyle=LS3)

# ── Global theme ─────────────────────────────────────────────────────────────
function set_siam_theme!()
    set_theme!(
        fontsize    = FS_BODY,
        fonts       = (; regular="Computer Modern", bold="Computer Modern Bold",
                          italic="Computer Modern Italic"),
        Axis        = (
            spinewidth        = 0.7,
            xgridvisible      = false,
            ygridvisible      = false,
            xticklabelsize    = FS_SMALL,
            yticklabelsize    = FS_SMALL,
            xlabelsize        = FS_BODY,
            ylabelsize        = FS_BODY,
            titlesize         = FS_TITLE,
            titlefont         = :bold,
            xtickwidth        = 0.7,
            ytickwidth        = 0.7,
        ),
        Legend      = (
            labelsize   = FS_SMALL,
            framewidth  = 0.7,
            framecolor  = GK,
            padding     = (4f0, 4f0, 3f0, 3f0),
            patchsize   = (16f0, 8f0),
        ),
        Lines       = (linewidth=1.6,),
        Scatter     = (markersize=5,),
    )
end

# ── Step histogram from probability vector ───────────────────────────────────
# Returns (xs_step, ys_step) suitable for lines!()
function step_hist(counts::AbstractVector{Int},
                   probs::AbstractVector{Float64})
    n = length(counts)
    xs = Float64[]
    ys = Float64[]
    for i in 1:n
        push!(xs, counts[i] - 0.5, counts[i] + 0.5)
        push!(ys, probs[i], probs[i])
    end
    xs, ys
end

# ── Integer-power-of-10 ticks for log-scale axes ────────────────────────────
# Returns a tick vector [10^pmin, ..., 10^pmax] covering [vmin, vmax].
# Usage:  Axis(...; yscale=log10, yticks=logticks(ymin, ymax))
function logticks(vmin::Real, vmax::Real)
    vmin = max(vmin, 1e-300)
    vmax = max(vmax, vmin * 10)
    pmin = floor(Int, log10(vmin))
    pmax = ceil(Int,  log10(vmax))
    [10.0^p for p in pmin:pmax]
end

# ── Panel label helper: (a), (b), … ─────────────────────────────────────────
function panel_label!(ax, letter; pos=:lt, fontsize=FS_BODY+1)
    halign = pos in (:lt,:lb) ? :left  : :right
    valign = pos in (:lt,:rt) ? :top   : :bottom
    x = pos in (:lt,:lb) ? 0.03 : 0.97
    y = pos in (:lt,:rt) ? 0.97 : 0.03
    text!(ax, "($letter)"; position=(x,y), space=:relative,
          align=(halign,valign), fontsize=fontsize, font=:bold)
end
