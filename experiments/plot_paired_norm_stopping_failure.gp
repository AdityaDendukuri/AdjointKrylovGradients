set datafile separator comma
set terminal pdfcairo enhanced monochrome font "Helvetica,9" size 7.1in,3.0in
set output "paper/figures/paired_norm_stopping_failure.pdf"

set multiplot layout 1,2 margins 0.09,0.98,0.18,0.91 spacing 0.11,0.04
set logscale y
set xrange [2:25]
set yrange [1e-6:2]
set xlabel "Krylov depth"
set ylabel "Frobenius norm"
set title "(a) Successive-core stopping rule"
set key bottom right spacing 1.1
set arrow 1 from 3,1e-6 to 3,2 nohead dt 3 lw 1
set label 1 "would stop" at 3.5,1.6e-4 left
plot \
  "paper/figures/paired_norm_stopping_failure.csv" using 1:7 every ::1 \
    with linespoints lw 1.5 pt 6 title "true Fréchet error", \
  "" using 1:8 every ::1 \
    with linespoints lw 1.5 pt 4 dt 2 title "successive difference", \
  1e-4 with lines dt 3 lw 1 title "tolerance"

unset arrow 1
unset label 1
set xrange [2:50]
set yrange [1e-16:2e2]
set xlabel "Krylov depth"
set ylabel "Absolute sensitivity error"
set title "(b) Contractive paired-norm error bound"
set key top right spacing 1.1
plot \
  "paper/figures/paired_norm_stopping_failure.csv" using 1:($4 > 1e-16 ? $4 : 1e-16) \
    with linespoints lw 1.5 pt 6 title "true gradient error", \
  "" using 1:($5 > 1e-16 ? $5 : 1e-16) every ::1 \
    with linespoints lw 1.5 pt 4 dt 2 title "successive difference", \
  "" using 1:6 \
    with lines lw 1.5 dt 4 title "paired-norm bound"

unset multiplot
