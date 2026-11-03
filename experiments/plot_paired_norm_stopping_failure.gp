set datafile separator comma
set terminal pdfcairo enhanced monochrome font "Helvetica,9" size 7.0in,2.8in
set output "paper/figures/paired_norm_stopping_failure.pdf"

set multiplot layout 1,2 margins 0.09,0.98,0.19,0.88 spacing 0.12
set logscale y
set format y "10^{%L}"
set title "(a) Stationary CTMC: one-sided termination"
set xrange [1:30]
set xtics 5
set yrange [1e-2:100]
set xlabel "Adjoint depth (forward depth = 1)"
set ylabel "Absolute Frobenius error or bound"
set key at graph 0.98,0.23 right spacing 1.1
set arrow 2 from 3.2,0.21 to 1.4,0.011 head filled size screen 0.008,15,45 lw 1.2
set label 2 "incorrectly stops at depth 1\n(true error 20.7)" at 3.5,0.23 left
plot \
  "paper/figures/one_sided_stopping_failure.csv" using 1:2 \
    with linespoints lw 2.2 pt 6 ps 0.65 title "true error", \
  "" using 1:($3 > 0 ? $3 : 1.05e-2) \
    with linespoints lw 2.2 pt 8 ps 0.65 dt 4 title "multiplicative estimate (59): identically zero", \
  "" using 1:($4 > 0 ? $4 : 1e-16) \
    with linespoints lw 2.2 pt 4 ps 0.65 dt 2 title "additive residual bound"

set title "(b) Stiff Brusselator CME: successive-difference failure"
unset arrow 2
unset label 2
set logscale x 2
set xrange [2:128]
set xtics ("2" 2, "4" 4, "8" 8, "16" 16, "32" 32, "64" 64, "128" 128)
set yrange [1e-10:1e-3]
set xlabel "Krylov depth"
set ylabel "Normalized coordinate error or bound"
set key at graph 0.02,0.02 left bottom opaque spacing 0.9 samplen 1.5 font ",7"
set arrow 1 from 16,1e-10 to 16,1e-3 nohead dt 3 lw 1
set label 1 "would stop here" at 18,2e-7 left
plot \
  "paper/figures/brusselator_error_control.csv" using 1:2 \
    with linespoints lw 2.2 pt 6 ps 0.55 title "true gradient error", \
  "" using 1:8 \
    with linespoints lw 2.2 pt 4 ps 0.55 dt 2 title "successive difference (60)", \
  "" using 1:3 \
    with lines lw 2.2 dt 4 title "additive residual bound", \
  "" every 2 using 1:3 \
    with points pt 8 ps 0.48 notitle, \
  1e-8 with lines dt 3 lw 1.3 title "tolerance"

unset multiplot
