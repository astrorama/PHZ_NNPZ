import numpy as np
from nnpz.utils.SedGenerator import SedGenerator


def test_sedGenerator():
    """
    Test the on-the-fly generation of redshifted SEDs
    """

    # Given
    sed1 = np.array([[1., 2., 3., 4.], [0., 1., 1.5, 0.5]]).T
    zs1 = [.0, .5, 1.]
    sed2 = np.array([[10., 25., 30., 35.], [0., 4., 5.5, 1.5]]).T
    zs2 = [1., 1.5]

    # When
    generator = SedGenerator()
    generator.add(sed1, zs1)
    generator.add(sed2, zs2)

    generated = [sed.sed for sed in generator]

    # Then
    assert len(generated) == 5

    np.testing.assert_array_equal(sed1, generated[0])
    np.testing.assert_array_equal(sed1[:, 0] * 1.5, generated[1][:, 0])
    np.testing.assert_array_equal(sed1[:, 0] * 2.0, generated[2][:, 0])
    np.testing.assert_array_equal(sed2[:, 0] * 2.0, generated[3][:, 0])
    np.testing.assert_array_equal(sed2[:, 0] * 2.5, generated[4][:, 0])
