"""Physical constants, ported from src/constants.F (set_constants).

The module attributes hold the values of the current test case:
``set_constants(testcase)`` (called by cm1x.state.initialize and before each
model step) switches them for testcase 4, 5 (constants.F) and back.  Every
cm1x module reads them as ``constants.<name>`` at call time.
"""

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


_testcase = 0


def set_constants(testcase=0):
    """constants.F ``set_constants``: g, rd, cp, cpv, xlv, xls for
    testcase 4 (Stevens et al 2005), 5 (Ackerman et al 2009) or the CM1
    defaults, and everything derived from them."""
    global _testcase, g, rd, cp, cpv, xlv, xls, govtwo, cpinv, cv, cvv, rcp
    global cpdcv, cvdcp, rovcp, rddcp, rddcv, rddrv, cvdrd, cpdrd, eps, reps
    global repsm1, cpt, cvt, pnum, lathv, lvdcp, condc, lv1, lv2, ls1, ls2, xlf
    testcase = int(testcase or 0)
    if testcase == _testcase:
        return
    _testcase = testcase
    if testcase == 4:
        g, rd, cp, cpv, xlv, xls = 9.81, 287.0, 1015.0, 1870.0, 2.47e6, 2834000.0
    elif testcase == 5:
        g, rd, cp, cpv, xlv, xls = 9.81, 287.0, 1004.0, 1870.0, 2.50e6, 2834000.0
    else:
        g, rd, cp, cpv, xlv, xls = 9.81, 287.04, 1005.7, 1870.0, 2501000.0, 2834000.0
    govtwo = 0.5 * g
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
    lv1 = xlv + (cpl - cpv) * to
    lv2 = cpl - cpv
    ls1 = xls + (cpi - cpv) * to
    ls2 = cpi - cpv
    xlf = xls - xlv
    if testcase in (4, 5):
        lv1 = xlv
        lv2 = 0.0


del math
