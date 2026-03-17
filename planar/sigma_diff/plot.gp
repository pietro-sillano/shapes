plot 'diff.csv' u 2:3 index 0 w lp title "Eun,sigma=0.1",\
'diff.csv' u 2:3 index 1 w lp title "Eun,sigma=0.025",\
'diff.csv' u 2:3 index 2 w lp title "Eun,sigma=0.15",\
'diff.csv' u 2:3 index 3 w lp title "Eun,sigma=0.2"


pause -1

plot 'diff.csv' u 2:($3+$4+$5) index 0 w lp title "Etot,sigma=0.1",\
'diff.csv' u 2:($3+$4+$5) index 1 w lp title "Etot,sigma=0.025",\
'diff.csv' u 2:($3+$4+$5) index 2 w lp title "Etot,sigma=0.15",\
'diff.csv' u 2:($3+$4+$5) index 3 w lp title "Etot,sigma=0.2"

pause -1

# plot 'diff.csv' u 2:4 index 0 w lp title "Ebo,sigma=0.1",\
# 'diff.csv' u 2:4 index 1 w lp title "Ebo,sigma=0.025",\
# 'diff.csv' u 2:4 index 2 w lp title "Ebo,sigma=0.15",\
# 'diff.csv' u 2:4 index 3 w lp title "Ebo,sigma=0.2"


# pause -1

# plot 'diff.csv' u 2:5 index 0 w lp title "Ead,sigma=0.1",\
# 'diff.csv' u 2:5 index 1 w lp title "Ead,sigma=0.025",\
# 'diff.csv' u 2:5 index 2 w lp title "Ead,sigma=0.15",\
# 'diff.csv' u 2:5 index 3 w lp title "Ead,sigma=0.2"
#
#
# pause -1
