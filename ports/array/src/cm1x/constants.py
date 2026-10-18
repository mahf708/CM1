"""Physical constants, ported from src/constants.F (set_constants, testcase=0)."""

import math

pi = 3.1415926535897932384626433
th0r = 300.0
to = 273.15
p00 = 1.0e5
rp00 = 1.0 / p00
clwsat = 1.0e-6
epsilon = 1.0e-18
karman = 0.40

g = 9.81
rd = 287.04
cp = 1005.7
cpv = 1870.0
xlv = 2501000.0
xls = 2834000.0
govtwo = 0.5 * g
rv = 461.5
cpinv = 1.0 / cp
cv = cp - rd
cvv = cpv - rv
rcp = 1.0 / cp
cpdcv = cp / cv
cvdcp = cv / cp
rovcp = rd / cp
rddcp = rd / cp
rddcv = rd / cv
rddrv = rd / rv
cvdrd = cv / rd
cpdrd = cp / rd
eps = rd / rv
reps = rv / rd
repsm1 = rv / rd - 1.0
cpt = (cpv / cp) - 1.0
cvt = (cvv / cv) - 1.0
pnum = (cp * rv) - (cpv * rd)
lathv = xlv
lvdcp = xlv / cp
condc = xlv * xlv / (rv * cp)
cpl = 4190.0
cpi = 2106.0
lv1 = xlv + (cpl - cpv) * to
lv2 = cpl - cpv
ls1 = xls + (cpi - cpv) * to
ls2 = cpi - cpv
xlf = xls - xlv
rhow = 1.0e3
svp1 = 0.6112
svp2 = 17.67
svp3 = 29.65
svpt0 = 273.15

del math
