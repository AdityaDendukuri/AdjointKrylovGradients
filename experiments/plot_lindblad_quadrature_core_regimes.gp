set terminal pdfcairo enhanced monochrome font "Helvetica,9" size 7.0in,2.7in
set output "paper/figures/lindblad_quadrature_core_regimes.pdf"
set datafile separator comma
set border linewidth 1
set tics out nomirror
set key top left
set multiplot layout 1,2 margins 0.09,0.98,0.18,0.88 spacing 0.13

set title "(a) Runtime at 1024 directions"
set xlabel "Krylov depth m"
set ylabel "Time (ms)"
set logscale y
set format y "10^{%L}"
plot "benchmark_results/lindblad_quadrature_core_depth.csv" using 4:7 \
    with linespoints lw 1.6 pt 4 title "quadrature (n_q=8)", \
    "" using 4:8 with linespoints lw 1.6 pt 8 dt 3 title "projected core"

unset logscale y
set format y "%g"
set title "(b) Core/quadrature resources"
set xlabel "Krylov depth m"
set ylabel "Ratio"
set yrange [0:*]
plot "benchmark_results/lindblad_quadrature_core_depth.csv" using 4:9 \
    with linespoints lw 1.6 pt 8 title "runtime", \
    "" using 4:13 with linespoints lw 1.6 pt 4 dt 3 title "allocated bytes", \
    "" using 4:16 with linespoints lw 1.6 pt 6 dt 2 title "evaluation storage"

unset multiplot
