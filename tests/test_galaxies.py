from EXPANSE import ResolvedGalaxy, MockResolvedGalaxy
import pytest

# getcurrent dir
import os
current_dir = os.path.dirname(os.path.realpath(__file__))


def test_init_from_h5():
    path = f'{current_dir}/galaxies/GalaxyTest.h5'

    galaxy = ResolvedGalaxy.init_from_h5(path)
    del galaxy
    # open again - to check it doesn't crash again
    galaxy = ResolvedGalaxy.init_from_h5(path)
    
    print(galaxy)
    
def test_mock_init_from_h5():
    path = f'{current_dir}/galaxies/MockGalaxyTest.h5'
    galaxy = MockResolvedGalaxy.init_from_h5(path)
    del galaxy
    # open again - to check it doesn't crash again
    galaxy = MockResolvedGalaxy.init_from_h5(path)

    print(galaxy)





