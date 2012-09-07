# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""

import numpy as np
from statsmodels.tools.tools import add_constant

#data from Rdatasets, doesn't match robust examples
"idx","sal","lag","trend","dis"
dta_salin0 = np.array([
1,7.6,8.2,4,23.01,
2,7.7,7.6,5,22.87,
3,4.3,4.6,0,26.42,
4,5.9,4.3,1,24.87,
5,5,5.9,2,29.9,
6,6.5,5,3,24.2,
7,8.3,6.5,4,23.22,
8,8.2,8.3,5,22.86,
9,13.2,10.1,0,22.27,
10,12.6,13.2,1,23.83,
11,10.4,12.6,2,25.14,
12,10.8,10.4,3,22.43,
13,13.1,10.8,4,21.79,
14,12.3,13.1,5,22.38,
15,10.4,13.3,0,23.93,
16,10.5,10.4,1,33.44,
17,7.7,10.5,2,24.86,
18,9.5,7.7,3,22.69,
19,12,10,0,21.79,
20,12.6,12,1,22.04,
21,13.6,12.1,4,21.03,
22,14.1,13.6,5,21.01,
23,13.5,15,0,25.87,
24,11.5,13.5,1,26.29,
25,12,11.5,2,22.93,
26,13,12,3,21.31,
27,14.1,13,4,20.77,
28,15.1,14.1,5,21.39]).reshape(-1,5)

#from book
ss_salin = '''\
 1   8.2  4  23.005   7.6
 2   7.6  5  23.873   7.7
 3   4.6  0  26.417   4.3
 4   4.3  1  24.868   5.9
 5   5.9  2  29.895   5.0
 6   5.0  3  24.200   6.5
 7   6.5  4  23.215   8.3
 8   8.3  5  21.862   8.2
 9  10.1  0  22.274  13.2
10  13.2  1  23.830  12.6
11  12.6  2  25.144  10.4
12  10.4  3  22.430  10.8
13  10.8  4  21.785  13.1
14  13.1  5  22.380  12.3
15  13.3  0  23.927  10.4
16  10.4  1  33.443  10.5
17  10.5  2  24.859   7.7
18   7.7  3  22.686   9.5
19  10.0  0  21.789  12.0
20  12.0  1  22.041  12.6
21  12.1  4  21.033  13.6
22  13.6  5  21.005  14.1
23  15.0  0  25.865  13.5
24  13.5  1  26.290  11.5
25  11.5  2  22.932  12.0
26  12.0  3  21.313  13.0
27  13.0  4  20.769  14.1
28  14.1  5  21.393  15.1'''

dta_salin = np.array(ss_salin.split(), float).reshape(-1,5)


outl_sal = [1, 5, 8, 9, 10, 11, 13, 16, 23, 24, 25, 28] #1-based indices
inl_sal = [i for i in range(28) if i+1 not in outl_sal]

nobs_used = 28
endog_sal0 = dta_salin0[-nobs_used:,1]
exog_sal0 = add_constant(dta_salin0[-nobs_used:,2:])

endog_salin = dta_salin[-nobs_used:, -1]
exog_salin = add_constant(dta_salin[-nobs_used:, 1:-1])

ss_aircraft = '''\
 1  6.3  1.7   8176   4500   2.76
 2  6.0  1.9   6699   3120   4.76
 3  5.9  1.5   9663   6300   8.75
 4  3.0  1.2  12837   9800   7.78
 5  5.0  1.8  10205   4900   6.18
 6  6.3  2.0  14890   6500   9.50
 7  5.6  1.6  13836   8920   5.14
 8  3.6  1.2  11628  14500   4.76
 9  2.0  1.4  15225  14800  16.70
10  2.9  2.3  18691  10900  27.68
11  2.2  1.9  19350  16000  26.64
12  3.9  2.6  20638  16000  13.71
13  4.5  2.0  12843   7800  12.31
14  4.3  9.7  13384  17900  15.73
15  4.0  2.9  13307  10500  13.59
16  3.2  4.3  29855  24500  51.90
17  4.3  4.3  29277  30000  20.78
18  2.4  2.6  24651  24500  29.82
19  2.8  3.7  28539  34000  32.78
20  3.9  3.3   8085   8160  10.12
21  2.8  3.9  30328  35800  27.84
22  1.6  4.1  46172  37000 107.10
23  3.4  2.5  17836  19600  11.19'''

dta_aircraft = np.array(ss_aircraft.split(), float).reshape(-1,6)

endog_aircraft = dta_aircraft[-nobs_used:, -1]
exog_aircraft = add_constant(dta_aircraft[-nobs_used:, 1:-1])

#> wood
#      x1     x2    x3    x4    x5     y
ss_wood = '''\
1  0.573 0.1059 0.465 0.538 0.841 0.534
2  0.651 0.1356 0.527 0.545 0.887 0.535
3  0.606 0.1273 0.494 0.521 0.920 0.570
4  0.437 0.1591 0.446 0.423 0.992 0.450
5  0.547 0.1135 0.531 0.519 0.915 0.548
6  0.444 0.1628 0.429 0.411 0.984 0.431
7  0.489 0.1231 0.562 0.455 0.824 0.481
8  0.413 0.1673 0.418 0.430 0.978 0.423
9  0.536 0.1182 0.592 0.464 0.854 0.475
10 0.685 0.1564 0.631 0.564 0.914 0.486
11 0.664 0.1588 0.506 0.481 0.867 0.554
12 0.703 0.1335 0.519 0.484 0.812 0.519
13 0.653 0.1395 0.625 0.519 0.892 0.492
14 0.586 0.1114 0.505 0.565 0.889 0.517
15 0.534 0.1143 0.521 0.570 0.889 0.502
16 0.523 0.1320 0.505 0.612 0.919 0.508
17 0.580 0.1249 0.546 0.608 0.954 0.520
18 0.448 0.1028 0.522 0.534 0.918 0.506
19 0.417 0.1687 0.405 0.415 0.981 0.401
20 0.528 0.1057 0.424 0.566 0.909 0.568'''

dta_wood = np.array(ss_wood.split(), float).reshape(-1,7)

endog_wood = dta_wood[-nobs_used:, -1]
exog_wood = add_constant(dta_wood[-nobs_used:, 1:-1])

from statsmodels.datasets.stackloss import load
data = load()   # class attributes for subclasses
exog_stackloss = add_constant(data.exog)
endog_stackloss = data.endog

