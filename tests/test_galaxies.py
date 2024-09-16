from EXPANSE import ResolvedGalaxy, MockResolvedGalaxy, expanse_viewer
import pytest
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor


# getcurrent dir
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
mock_galaxy_path = f"{current_dir}/galaxies/MockGalaxyTest.h5"
galaxy_path = f"{current_dir}/galaxies/GalaxyTest.h5"


def init_from_h5(obj, path):
    """
    Test that the galaxy is correctly initialized from an h5 file,
    and that the file size does not increase when the galaxy is simply
    opened and closed. Tests that a serialzed galaxy can then be opened again.

    """

    # Measure size of file
    initial_size = os.path.getsize(path)

    galaxy = obj.init_from_h5(path, save_out=False)
    del galaxy

    # Measure size of file
    final_size = os.path.getsize(path)

    # open again - to check it doesn't crash again
    galaxy = obj.init_from_h5(path, save_out=False)

    assert (
        initial_size >= final_size
    ), "Memory leak! File size should not increase just by opening it"


import asyncio
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor


def run_server(gal_path, tab="Cutouts"):
    galaxies_dir = os.path.dirname(gal_path)
    galaxy = os.path.basename(gal_path)
    error = None
    event = threading.Event()

    async def run_server_async():
        try:
            await expanse_viewer(
                [
                    "--galaxy",
                    galaxy,
                    "--gal_dir",
                    galaxies_dir,
                    "--test_mode",
                    "True",
                    "--tab",
                    tab,
                ],
                standalone_mode=False,
            )
        except Exception as e:
            nonlocal error
            error = e
        finally:
            event.set()

    def run_in_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server_async())

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(run_in_thread, loop)

    # Wait for the server to finish or timeout
    timeout = 5  # Adjust this value as needed
    event.wait(timeout)

    if not event.is_set():
        print("Server thread timed out")
        loop.call_soon_threadsafe(loop.stop)

    executor.shutdown(wait=True)

    if error:
        raise AssertionError(f"Server crashed with error: {error}")


# Usage


def test_galaxy():
    init_from_h5(ResolvedGalaxy, galaxy_path)


def test_mock_galaxy():
    init_from_h5(MockResolvedGalaxy, mock_galaxy_path)


test_mock_galaxy()

# def test_galaxy_server():
"""
This should run the server and open the galaxy in the Cutouts tab

"""
#    run_server(galaxy_path)

# def test_mock_galaxy_server():
"""
This should run the server and open the mock galaxy in the Cutouts tab
"""
#    run_server(mock_galaxy_path)
