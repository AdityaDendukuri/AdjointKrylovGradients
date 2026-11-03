$errors << EOD
11 1.777397e-6 2.204841e-6 1.972877e-6
22 1.481614e-5 1.464216e-5 1.047249e-5
33 2.560734e-6 3.161814e-6 3.598533e-6
44 1.973799e-6 2.386103e-6 3.483714e-6
55 2.145515e-7 3.045391e-7 4.231740e-7
EOD
$wc << EOD
4 8.553837e-1
6 1.020276e-3
16 2.710904e-5
76 1.846325e-5
EOD
$ic << EOD
4 5.475796e-1
6 1.027953
16 4.390408e-1
76 2.075196e-2
EOD
$ft << EOD
4 1.777397e-6
6 2.331481e-6
16 3.724581e-5
76 7.785404e-5
EOD

set terminal pdfcairo monochrome enhanced font "Helvetica,10" size 6.5in,2.5in
set output "paper/figures/brusselator_headliner_bw.pdf"
set multiplot layout 1,2 margins 0.07,0.99,0.19,0.91 spacing 0.10,0.04
set border 3
set tics nomirror
set key opaque box reverse Left font ",8"

set title "(a) Ungrouped finite-time identification"
set xlabel "Seed"
set ylabel "Relative error"
set xtics ("11" 11, "22" 22, "33" 33, "44" 44, "55" 55)
set logscale y
set yrange [1e-7:3e-5]
set format y "10^{%L}"
set key at graph 0.5, graph 0.08 center bottom horizontal font ",7.5"
plot $errors using 1:2 with linespoints lw 1.5 ps 0.7 pt 7 title "Coefficients", \
     $errors using 1:3 with linespoints lw 1.5 ps 0.7 pt 5 dt 2 title "Held-out", \
     $errors using 1:4 with linespoints lw 1.5 ps 0.7 pt 9 dt 3 title "Long horizon"

set title "(b) Matched comparison under temporal thinning"
set xlabel "Number of snapshots"
set ylabel "Held-out coefficient error"
set logscale y
set logscale x
set xrange [3.5:85]
set autoscale y
set xtics ("4" 4, "6" 6, "16" 16, "76" 76)
set format y "10^{%L}"
set key default
set key top right opaque box reverse Left font ",7.5"
plot $ft with linespoints lw 1.5 ps 0.7 pt 7 title "Finite-time Krylov", \
     $wc with linespoints lw 1.5 ps 0.7 pt 5 dt 2 title "Weak SINDy", \
     $ic with linespoints lw 1.5 ps 0.7 pt 9 dt 3 title "Interval weak"

unset multiplot
