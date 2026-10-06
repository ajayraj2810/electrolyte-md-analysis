import numpy as np
from electrolyte_md.pbc import minimum_image, unwrap_step


def test_minimum_image():
    dr = np.array([9.0, -9.0, 1.0])
    box = np.array([10.0, 10.0, 10.0])
    assert np.allclose(minimum_image(dr, box), [-1.0, 1.0, 1.0])


def test_unwrap_step():
    prev_wrapped = np.array([9.5, 1.0, 1.0])
    current_wrapped = np.array([0.5, 1.0, 1.0])
    prev_unwrapped = prev_wrapped.copy()
    got = unwrap_step(prev_unwrapped, prev_wrapped, current_wrapped, [10.0, 10.0, 10.0])
    assert np.allclose(got, [10.5, 1.0, 1.0])
