set terminal pdfcairo enhanced color font "Helvetica,10" size 4.2in,3.0in
set output "paper/figures/lindblad_library_scaling.pdf"
set datafile separator comma
set monochrome
set border linewidth 1
set tics out nomirror
set key top left
set xlabel "Number of Lindblad directions"
set ylabel "Time (ms)"
set logscale xy
set xtics (16,64,256,510)
set format x "%g"
set format y "10^{%L}"
set xrange [12:700]
set yrange [0.8:600]
plot "benchmark_results/lindblad_library_scaling.csv" using 3:8 \
    with linespoints lw 1.7 pt 6 title "one action per direction", \
     "" using 3:7 with linespoints lw 1.7 pt 4 dt 2 title "quadrature streaming", \
     "" using 3:13 with linespoints lw 1.7 pt 8 dt 3 \
        title "projected core (Kandolf et al., Alg. 2)", \
     "" using 3:(sqrt($7*$8)):(sprintf("%.0fx",$9)) \
        with labels boxed offset 0.2,0 notitle
