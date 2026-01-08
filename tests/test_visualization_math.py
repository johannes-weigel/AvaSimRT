from __future__ import annotations

import numpy as np

from avasimrt.result import ComplexReading
from avasimrt.visualization.math import amps_to_db, magnitude, phase_deg

FREQ = 0                # not relevant
ACCEPTED_DELTA = 0.01   # allowed rounding error to manual calculation

c1 = ComplexReading(freq=FREQ, real= 0.8, imag= 0.1)   # |H| = 0.806.. ~  -1.87 dB
c2 = ComplexReading(freq=FREQ, real= 0.6, imag= 0.4)   # |H| = 0.721.. ~  -2.84 dB
c3 = ComplexReading(freq=FREQ, real=-0.3, imag= 0.7)   # |H| = 0.761.. ~  -2.37 dB
c4 = ComplexReading(freq=FREQ, real=-0.5, imag=-0.2)   # |H| = 0.538.. ~  -5.38 dB
c5 = ComplexReading(freq=FREQ, real= 0.2, imag=-0.9)   # |H| = 0.922.. ~  -0.71 dB
c6 = ComplexReading(freq=FREQ, real= 0.1, imag= 0.0)   # |H| = 0.1     = -20.00 dB

values = [c1, c2, c3, c4, c5, c6]

def test_magnitude_base():
    assert magnitude(ComplexReading(FREQ, 0, 0)) == 0
    assert magnitude(ComplexReading(FREQ, 1, 0)) == 1
    assert magnitude(ComplexReading(FREQ, 0, 1)) == 1
    assert magnitude(ComplexReading(FREQ, 4, 3)) == 5

def test_magnitude_example():
    assert abs(magnitude(c1) - 0.806) < ACCEPTED_DELTA
    assert abs(magnitude(c2) - 0.721) < ACCEPTED_DELTA
    assert abs(magnitude(c3) - 0.761) < ACCEPTED_DELTA
    assert abs(magnitude(c4) - 0.538) < ACCEPTED_DELTA
    assert abs(magnitude(c5) - 0.922) < ACCEPTED_DELTA
    assert abs(magnitude(c6) - 0.1) < ACCEPTED_DELTA

def test_phase():
    assert phase_deg(ComplexReading(FREQ,  0,  0)) == 0
    assert phase_deg(ComplexReading(FREQ,  1,  1)) == 45
    assert phase_deg(ComplexReading(FREQ, -1,  1)) == 135
    assert phase_deg(ComplexReading(FREQ, -1, -1)) == -135
    assert phase_deg(ComplexReading(FREQ,  1, -1)) == -45

def test_amps_to_db_example():
    actual = amps_to_db(np.array([0.806, 0.721, 0.761, 0.538, 0.922, 0.1], dtype=float))

    assert abs(actual[0] - ( -1.87)) < ACCEPTED_DELTA
    assert abs(actual[1] - ( -2.84)) < ACCEPTED_DELTA
    assert abs(actual[2] - ( -2.37)) < ACCEPTED_DELTA
    assert abs(actual[3] - ( -5.38)) < ACCEPTED_DELTA
    assert abs(actual[4] - ( -0.71)) < ACCEPTED_DELTA
    assert abs(actual[5] - (-20.00)) < ACCEPTED_DELTA
    