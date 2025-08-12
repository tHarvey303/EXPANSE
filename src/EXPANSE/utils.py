import copy
import curses
import glob
import os
import pathlib
import shutil
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Optional, Tuple

import h5py as h5
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.wcs import WCS
from matplotlib.colors import (
    ListedColormap,
    to_rgba,
)
from matplotlib.patches import FancyArrow
from matplotlib.patheffects import Normal, SimpleLineShadow
from photutils.aperture import (
    CircularAperture,
    EllipticalAperture,
    aperture_photometry,
)
from scipy.interpolate import interp1d
from tqdm import tqdm

file_path = os.path.abspath(__file__)

if os.path.exists("/.singularity.d/Singularity"):
    computer = "singularity"
elif "nvme" in file_path:
    computer = "morgan"
elif "Users" in file_path:
    computer = "mac"
else:
    computer = "unknown"


def update_mpl(tex_on=True):
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["axes.labelsize"] = 18.0
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["figure.facecolor"] = "#f7f7f7"
    # mpl.rcParams["figure.edgecolor"] = 'k'
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["figure.dpi"] = 300

    if tex_on:
        # mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        mpl.rc("text", usetex=True)
        mpl.rcParams["text.usetex"] = True

    else:
        mpl.rcParams["text.usetex"] = False


# recursively go through the old file and copy over any metadata attributes which are not in the new file (could be on any level and on a group or dataset)
def copy_attrs(old_group, new_group):
    for key in old_group.keys():
        if key in new_group.keys():
            if type(old_group[key]) is h5.Group:
                copy_attrs(old_group[key], new_group[key])
            else:
                for attr_key in old_group[key].attrs.keys():
                    if attr_key not in new_group[key].attrs.keys():
                        new_group[key].attrs[attr_key] = old_group[key].attrs[attr_key]
        else:
            if type(old_group[key]) is h5.Group:
                new_group.create_group(key)
                copy_attrs(old_group[key], new_group[key])
            else:
                new_group.create_dataset(key, data=old_group[key])
                for attr_key in old_group[key].attrs.keys():
                    new_group[key].attrs[attr_key] = old_group[key].attrs[attr_key]


def calculate_distance_to_bin_from_center(
    bin_map, center, value, type="mean", weight_map=None, scale=None
):
    """
    Given a binmap, where background is 0, and positive integer values are different bins on the map,
    calculate the average scalar distance from the center to all points with the given value.


    """

    bin_map = copy.deepcopy(bin_map)

    # Calculate the distance from the center to each point in the bin map

    y, x = np.indices(bin_map.shape)

    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Mask the distances to only include the points with the given value

    distances = np.ma.masked_where(bin_map != value, distances)

    # Calculate the average distance

    if type == "mean":
        value = np.mean(distances)
    elif type == "median":
        value = np.median(distances)
    elif type == "max":
        value = np.max(distances)
    elif type == "min":
        value = np.min(distances)
    elif type == "weighted_mean":
        assert weight_map is not None, "Must provide a weight map for weighted mean"
        # Get values for value from weight map
        weights = np.ma.masked_where(bin_map != value, weight_map)
        # Calculate the weighted mean
        value = np.ma.average(distances, weights=weights)

    if scale is not None:
        value = value * scale

    return value


def scale_fluxes(
    mag_aper,
    mag_auto,
    band,
    a,
    b,
    theta,
    kron_radius,
    psf,
    flux_type="mag",  # 'mag' or 'flux'
    zero_point=28.08,
    aper_diam=0.32 * u.arcsec,
    pix_scale=0.03 * u.arcsec,
    max_auto_correction=None,  # If set, will not correct if the auto flux is larger than this value
):
    """Scale fluxes to total flux using PSF
    Scale mag_aper to mag_auto (ideally measured in LW stack), and then derive PSF correction by placing a elliptical aperture around the PSF.
    """
    a = kron_radius * a
    b = kron_radius * b
    area_of_aper = np.pi * (aper_diam.to(u.arcsec) / (2 * pix_scale)) ** 2
    area_of_ellipse = np.pi * a * b
    # Don't use the scale factor if the ellipse is smaller than the aperture
    scale_factor = area_of_aper / area_of_ellipse
    if flux_type == "mag":
        flux_auto = 10 ** ((zero_point - mag_auto) / 2.5)
        flux_aper = 10 ** ((zero_point - mag_aper) / 2.5)
    else:
        flux_auto = mag_auto
        flux_aper = mag_aper

    # print(flux_auto, flux_aper, scale_factor)
    if (scale_factor > 1) and flux_auto > flux_aper:
        factor = flux_auto / flux_aper
        clip = False
    else:
        factor = 1
        clip = True

    if max_auto_correction is not None and factor > max_auto_correction:
        factor = 1 # Don't apply the correction if the auto flux is larger than this value - typically only happens for very faint objects anyway
        clip = True

    if clip:
        # Make Elliptical Aperture be the circle
        a = aper_diam.to(u.arcsec) / (2 * pix_scale)
        b = aper_diam.to(u.arcsec) / (2 * pix_scale)
        theta = 0

    # print(f"Corrected aperture flux by {factor} for kron ellipse flux.")
    # Scale for PSF
    assert type(psf) is np.ndarray, "PSF must be a numpy array"
    assert np.sum(psf) < 1, "PSF should not be normalised, some flux is outside the footprint."
    # center = (psf.shape[0] - 1) / 2
    from photutils.centroids import centroid_com

    center = centroid_com(psf)
    #
    # from photutils import CircularAperture, EllipticalAperture, aperture_photometry
    # circular_aperture = CircularAperture(center, center, r=aper_diam/(2*pixel_scale))
    # circular_aperture_phot = aperture_photometry(psf, circular_aperture)

    # get path of this file
    file_path = os.path.abspath(__file__)
    psf_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path))) + "/psfs"
    file1 = f"{psf_path}/Encircled_Energy_LW_ETCv2.txt"
    tab1 = Table.read(file1, format="ascii")

    file2 = f"{psf_path}/ACS_WFC_EE.txt"
    tab2 = Table.read(file2, format="ascii.commented_header")

    if band in tab1.colnames:
        tab = tab1
    elif band in tab2["band"]:
        tab = tab2
        # Current tab is bands in row and aper_radius in column, need to swap this
        tab = tab.to_pandas().T
        # Change row 0 to column names
        tab.columns = tab.iloc[0]
        tab = tab.drop(tab.index[0])
        # Rename band to aper_radius
        tab["aper_radius"] = [float(i) * pix_scale for i in tab.index]

    x = tab["aper_radius"] / pix_scale
    y = tab[band]
    f = interp1d(x, y, kind="linear", fill_value=(0, 1.0), bounds_error=False)

    a = float(a)
    b = float(b)

    if a > psf.shape[0] or b > psf.shape[0]:
        # approximate as a circle
        r = np.sqrt(a * b)
        encircled_energy = f(r)

        # encircled_energy = # enclosed_energy in F444W from band
    else:
        elliptical_aperture = EllipticalAperture(center, a=a, b=b, theta=theta)
        elliptical_aperture_phot = aperture_photometry(psf, elliptical_aperture)
        encircled_energy = elliptical_aperture_phot["aperture_sum"][0]

    factor_total = factor / encircled_energy

    return factor_total, factor


def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [
        {"code": code, "templates": templates, "lowz_zmax": lowz_zmax}
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr)
        for lowz_zmax in lowz_zmaxs
    ]


class CLIInterface:
    def __init__(self):
        self.screen = None
        self.lines = [[]]
        self.current_task = None
        self.progress = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.start_time = 0
        # self.keyboard_interrupt_event = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.start_time = time.time()

        # signal.signal(signal.SIGINT, self._signal_handler)

    def stop(self):
        self.running = False
        self.thread.join()
        # self.keyboard_interrupt_event.set()
        self.end_time = time.time()

    def update(self, lines=None, current_task=None, progress=None):
        with self.lock:
            if lines is not None:
                self.lines = lines

            if progress is not None:
                self.progress = progress

    def _run(self):
        curses.wrapper(self._curses_main)

    def _signal_handler(self, signum, frame):
        # self.keyboard_interrupt_event.set()
        self.stop()

    def _curses_main(self, stdscr):
        self.screen = stdscr
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

        while self.running:
            self._update_display()
            # if self.keyboard_interrupt_event.is_set():
            #    self._handle_keyboard_interrupt()
            #    break
            time.sleep(0.1)

    def _update_display(self):
        if not self.screen:
            return

        height, width = self.screen.getmaxyx()

        self.screen.clear()

        with self.lock:
            # Add divider
            self.screen.addstr(0, 0, "-" * width, curses.color_pair(2))

            top_line = [
                "ResolvedGalaxy CLI",
                f"Time elapsed: {time.time() - self.start_time:.2f}s",
                f"Computer: {computer}",
            ]

            # top_line.append(f'RAM usage: {psutil.virtual_memory().percent:.2f}%')
            # self.screen.addstr(0, 0, f"ResolvedGalaxy  Time elapsed: {time.time() - self.start_time:.2f}s  Computer: {computer}", curses.color_pair(1))
            for i, item in enumerate(top_line):
                self.screen.addstr(
                    1,
                    i * (width // len(top_line)),
                    item[: width // len(top_line) - 1],
                    curses.color_pair(i),
                )

            info_line = []
            # add a RAM usage line
            try:
                import psutil

                info_line.append(f"Total RAM usage: {psutil.virtual_memory().percent:.2f}%")
                pid = os.getpid()
                python_process = psutil.Process(pid)
                memoryUse = python_process.memory_info()[0] / 2.0**30  # memory use in GB...I think
                info_line.append(f"Script RAM usage: {memoryUse:.2f} GB")
                # script CPU usage
                info_line.append(f"Script CPU usage: {python_process.cpu_percent(interval = 0.1)}%")
                # Which core is being used
                info_line.append(f"Core = {python_process.cpu_num()}")
                core_num = python_process.cpu_num()

                index = np.argwhere(
                    [i[0] == f"Core {core_num}" for i in psutil.sensors_temperatures()["coretemp"]]
                )
                if len(index) > 0:
                    # flatten index
                    index = index.flatten()

                    temp = psutil.sensors_temperatures()["coretemp"][index[0]][1]
                else:
                    temp = "N/A"
                # Core clock and temperature
                # info_line.append(f'(Clock: {python_process.cpu_freq()}, ')
                info_line.append(f"Temp: {temp}°C ")
            except ImportError:
                pass

            for i, item in enumerate(info_line):
                self.screen.addstr(
                    2,
                    i * (width // len(info_line)),
                    item[: width // len(info_line) - 1],
                    curses.color_pair(i),
                )

            # Add divider
            self.screen.addstr(3, 0, "-" * width, curses.color_pair(2))

            for i, line in enumerate(self.lines):
                for j, item in enumerate(line):
                    self.screen.addstr(
                        i + 4,
                        j * (width // len(line)),
                        item[: width // len(line) - 1],
                        curses.color_pair(i),
                    )

            # Show current task.
            if self.current_task is not None:
                self.screen.addstr(
                    height - 2,
                    0,
                    f"Current task: {self.current_task}",
                    curses.color_pair(3),
                )

            progress_width = int(self.progress * (width - 2))
            progress_bar = "[" + "#" * progress_width + " " * (width - 2 - progress_width) + "]"
            self.screen.addstr(height - 1, 0, progress_bar[: width - 1], curses.color_pair(3))

        self.screen.refresh()

    def _handle_keyboard_interrupt(self):
        height, width = self.screen.getmaxyx()
        self.screen.clear()
        self.screen.addstr(height // 2, width // 2 - 11, "Keyboard Interrupt", curses.A_BOLD)
        self.screen.addstr(height // 2 + 1, width // 2 - 9, "Exiting...", curses.A_BOLD)
        self.screen.refresh()
        time.sleep(2)  # Show the message for 2 seconds before exiting


def is_cli():
    # Check if running in IPython/Jupyter
    try:
        __IPYTHON__
        return False
    except NameError:
        pass

    # Check for common notebook environments
    if "JUPYTER_RUNTIME_DIR" in os.environ or "IPYTHON_KERNEL_PATH" in os.environ:
        return False

    # Check if it's an interactive Python shell
    if hasattr(sys, "ps1"):
        return False

    # Check if the script is being piped or redirected
    if sys.stdin is not None and not sys.stdin.isatty():
        return False

    # If none of the above, it's likely a CLI environment
    return True


def send_email(contents, subject="", address="tharvey303@gmail.com"):
    """
    except Exception as e:
            # Email me if you crash
            ctime  = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            send_email(contents=f'{e}', subject = f'{sys.argv[0]} crash at {ctime}')
            raise e

    """
    import yagmail

    if computer == "morgan":
        oauth2_file = "/nvme/scratch/work/tharvey/scripts/testing/client_secret.json"
    elif computer == "singularity":
        oauth2_file = glob.glob("/home/scripts/client_secret_*.json")[0]
    else:
        print("No oauth2 file found, skipping email.")
        print(contents)
        return

    yagmail.SMTP(
        "tcharvey303",
        oauth2_file=oauth2_file,
    ).send(address, subject, contents)
    print("Sent email.")


def calculate_ab_zeropoint(unknown_counts, known_jy=None, ab_mag_known=None):
    # Convert μJy to AB magnitude
    if known_jy is None and ab_mag_known is None:
        raise ValueError("Must provide either known_jy or ab_mag_known.")

    if ab_mag_known is None:
        ab_mag_known = -2.5 * np.log10(known_jy) + 8.90

    # Calculate zeropoint
    zeropoint = ab_mag_known + 2.5 * np.log10(unknown_counts)

    return zeropoint


class PhotometryBandInfo:
    """
    A class to store information about a photometry band.
    File Paths to image, wht, err, and segmentation maps.
    As well as hdu extensions for each.

    """

    def __init__(
        self,
        band_name,
        survey,
        image_path,
        instrument="auto",
        wht_path=None,
        err_path=None,
        seg_path=None,
        psf_path=None,
        psf_type=None,
        psf_kernel_path=None,
        psf_matched_image_path=None,
        psf_matched_err_path=None,
        im_pixel_scale="HEADER",
        image_zp="HEADER",
        image_unit="HEADER",
        im_hdu_ext=0,
        wht_hdu_ext=0,
        err_hdu_ext=0,
        seg_hdu_ext=0,
        detection_image=False,
        psf_matched=False,
        auto_photometry={},
        aperture_photometry={},
    ):
        """
        band_name : str
            The name of the band, e.g. 'F444W'
        survey : str
            The survey the band is from, e.g. 'HST'
        image_path : str
            The path to the image file. Can be a folder, in which case the code will attempt to auto-detect the file.
        instrument : str
            The instrument the band is from, e.g. 'WFC3'
        wht_path : str
            The path to the weight file. Can be a folder, in which case the code will attempt to auto-detect the file.
            Can be 'im', which will look for a 'WHT' extension in the image file. Or 'im_folder', which will look for a weight file in the same folder as the image file.
        err_path : str
            Optional: The path to the error file. Can be a folder, in which case the code will attempt to auto-detect the file.
            Can be 'im', which will look for a 'ERR' extension in the image file. Or 'im_folder', which will look for a error file in the same folder as the image file.
        seg_path : str or None
            Optional: The path to the segmentation file. Can be a folder, in which case the code will attempt to auto-detect the file.
            Can be 'im', which will look for a 'SEG' extension in the image file. Or 'im_folder', which will look for a segmentation file in the same folder as the image file.
        psf_path : str or None
            Optional: The path to the PSF file. Can be a folder, in which case the code will attempt to auto-detect the file.
        psf_type : str or None
            Optional: Name of PSF model, e.g. 'webbpsf', 'empirical' etc. Allows for multiple PSF models to be used.
            Must be provided if psf_path or psf_kernel_path is provided.
        psf_kernel_path : str or None
            Optional: Path to the PSF kernel file. Can be a folder, in which case the code will attempt to auto-detect the file.
        psf_matched_image_path : str or None
            Only required if providing both PSF matched images and un-matched images. The path to the PSF matched image file.
            Otherwise provide this as image_path and set psf_matched = True.
        psf_matched_err_path : str or None
            Only required if providing both PSF matched images and un-matched images. The path to the PSF matched error file.
            Otherwise provide this as err_path and set psf_matched = True.
        im_pixel_scale : str
            The pixel scale of the image (arcsec/pixel). Can be 'HEADER', in which case the code will attempt to read the pixel scale from the image header.
        image_zp : str
            The AB zero point of the image. Can be 'HEADER', in which case the code will attempt to read the zero point from the image header.
        image_unit : str
            The unit of the image. Can be 'HEADER', in which case the code will attempt to read the unit from the image header.
        im_hdu_ext : int
            The HDU extension of the image file. If JWST image structure is detected, this will be set to 1.
        wht_hdu_ext : int
            The HDU extension of the weight file. If JWST image structure is detected, this will be set to 2.
        err_hdu_ext : int
            The HDU extension of the error file. If JWST image structure is detected, this will be set to 3.
        seg_hdu_ext : int
            The HDU extension of the segmentation file.
        detection_image : bool
            If True, this band is the detection image.
        psf_matched : string or False
            Important! This only refers to the basic image arguments here - e.g. image_path, wht_path, err_path.
            If providing both PSF matched and un-matched images, set this to False.
            If False, the images are not PSF matched. Otherwise provide the name of the band that the image is PSF matched to.
        auto_photometry : dict
            auto photometry for one galaxy of interest
            Optional - can all be measured with sep once loaded.
            A dictionary of parameters of already derived autophotometry. Can have any values and will be saved.
            E.g. {'MAG_AUTO': 25.0, 'FLUX_AUTO': 1e-20, 'FLUXERR_AUTO': 1e-21, 'MAGERR_AUTO': 0.1,
                    'FLUX_RADIUS': 1.0, 'KRON_RADIUS': 1.0, 'ISOAREA_IMAGE': 1.0, 'ISOAREAF_IMAGE': 1.0}

        aperture_photometry : dict
            aperture photometry for one galaxy of interest
            Optional - can all be measured with sep once loaded.
            A dictionary of aperture photometry. Each key should be the aperture diameter in arcsec as an astropy unit.
            e.g. {str(0.32*u.arcsec):{'flux':1e-20, 'fluxerr':1e-21, 'depths':[25]}}
                Photometry is expected to be in Jy
                The apertures are assumed to be centered on the SkyCoord provided in the ResolvedGalaxy init_from_basics


        """

        self.band_name = band_name
        self.instrument = instrument
        self.auto_photometry = auto_photometry
        self.aperture_photometry = aperture_photometry

        if wht_path == "im_folder":
            wht_path = image_path

        if err_path == "im_folder":
            err_path = image_path

        if seg_path == "im_folder":
            seg_path = image_path

        if psf_path == "im_folder":
            psf_path = image_path

        # Perform same checks for err, wht and seg
        self.detection_image = detection_image

        if wht_path is not None and os.path.isdir(wht_path):
            wht_path_finder = glob.glob(os.path.join(wht_path, f"*{band_name.upper()}*wht*.fits"))
            wht_path_finder.extend(
                glob.glob(os.path.join(wht_path, f"*{band_name.lower()}*wht*.fits"))
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(
                        wht_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*wht*.fits",
                    )
                )
            )
            wht_path_finder.extend(
                glob.glob(os.path.join(wht_path, f"*{band_name.upper()}*weight*.fits"))
            )
            wht_path_finder.extend(
                glob.glob(os.path.join(wht_path, f"*{band_name.lower()}*weight*.fits"))
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(
                        wht_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*weight*.fits",
                    )
                )
            )
            wht_path_finder.extend(
                glob.glob(os.path.join(wht_path, f"*{band_name.upper()}*WHT*.fits"))
            )
            wht_path_finder.extend(
                glob.glob(os.path.join(wht_path, f"*{band_name.lower()}*WHT*.fits"))
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(
                        wht_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*WHT*.fits",
                    )
                )
            )

            # Remove duplicates
            wht_path = list(set(wht_path_finder))

            if len(wht_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(wht_path) == 0:
                raise ValueError(
                    f"No wht files found for band {band_name}. Please provide full path."
                )
            else:
                wht_path = wht_path[0]
                print(f"Auto detected weight path {wht_path} for band {band_name}.")

        if err_path is not None and os.path.isdir(err_path):
            err_path_finder = glob.glob(os.path.join(err_path, f"*{band_name.upper()}*err*.fits"))
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.lower()}*err*.fits"))
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*err*.fits",
                    )
                )
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.upper()}*error*.fits"))
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.lower()}*error*.fits"))
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*error*.fits",
                    )
                )
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.upper()}*ERR*.fits"))
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.lower()}*ERR*.fits"))
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*ERR*.fits",
                    )
                )
            )
            # Same for RMS
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.upper()}*rms*.fits"))
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.lower()}*rms*.fits"))
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*rms*.fits",
                    )
                )
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.upper()}*RMS*.fits"))
            )
            err_path_finder.extend(
                glob.glob(os.path.join(err_path, f"*{band_name.lower()}*RMS*.fits"))
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*RMS*.fits",
                    )
                )
            )

            # Remove duplicates
            err_path = list(set(err_path_finder))

            if len(err_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(err_path) == 0:
                raise ValueError(
                    f"No err files found for band {band_name}. Please provide full path."
                )
            else:
                err_path = err_path[0]
                print(f"Auto detected error path {err_path} for band {band_name}.")

        if seg_path is not None and os.path.isdir(seg_path):
            seg_path_finder = glob.glob(os.path.join(seg_path, f"*{band_name.upper()}*seg*.fits"))
            seg_path_finder.extend(
                glob.glob(os.path.join(seg_path, f"*{band_name.lower()}*seg*.fits"))
            )
            seg_path_finder.extend(
                glob.glob(
                    os.path.join(
                        seg_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*seg*.fits",
                    )
                )
            )
            seg_path_finder.extend(
                glob.glob(os.path.join(seg_path, f"*{band_name.upper()}*SEG*.fits"))
            )
            seg_path_finder.extend(
                glob.glob(os.path.join(seg_path, f"*{band_name.lower()}*SEG*.fits"))
            )
            seg_path_finder.extend(
                glob.glob(
                    os.path.join(
                        seg_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*SEG*.fits",
                    )
                )
            )

            # Remove duplicates
            seg_path = list(set(seg_path_finder))
            if len(seg_path) > 1:
                seg_path = [path for path in seg_path if "+" not in path]

            if len(seg_path) > 1:
                print(seg_path)
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(seg_path) == 0:
                raise ValueError(
                    f"No seg files found for band {band_name}. Please provide full path."
                )
            else:
                seg_path = seg_path[0]
                print(f"Auto detected segmentation path {seg_path} for band {band_name}.")
        elif seg_path is not None and not os.path.exists(seg_path):
            raise ValueError(f"Segmentation path {seg_path} does not exist.")

        if psf_path is not None and os.path.isdir(psf_path):
            psf_path_finder = glob.glob(os.path.join(psf_path, f"*{band_name.upper()}*psf*.fits"))
            psf_path_finder.extend(
                glob.glob(os.path.join(psf_path, f"*{band_name.lower()}*psf*.fits"))
            )
            psf_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*psf*.fits",
                    )
                )
            )
            psf_path_finder.extend(
                glob.glob(os.path.join(psf_path, f"*{band_name.upper()}*PSF*.fits"))
            )
            psf_path_finder.extend(
                glob.glob(os.path.join(psf_path, f"*{band_name.lower()}*PSF*.fits"))
            )
            psf_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*PSF*.fits",
                    )
                )
            )

            # Remove duplicates
            psf_path = list(set(psf_path_finder))

            if len(psf_path) > 1:
                psf_path = [path for path in psf_path if "norm" not in path]

            if len(psf_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(psf_path) == 0:
                raise ValueError(
                    f"No PSF files found for band {band_name}. Please provide full path."
                )
            else:
                psf_path = psf_path[0]
                print(f"Auto detected PSF path {psf_path} for band {band_name}.")
        elif psf_path is not None and not os.path.exists(psf_path):
            raise ValueError(f"PSF path {psf_path} does not exist.")

        if psf_kernel_path is not None and os.path.isdir(psf_kernel_path):
            psf_kernel_path_finder = glob.glob(
                os.path.join(psf_kernel_path, f"*{band_name.upper()}*psf*.fits")
            )
            psf_kernel_path_finder.extend(
                glob.glob(os.path.join(psf_kernel_path, f"*{band_name.lower()}*psf*.fits"))
            )
            psf_kernel_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_kernel_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*psf*.fits",
                    )
                )
            )
            psf_kernel_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_kernel_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*kernel*.fits",
                    )
                )
            )
            psf_kernel_path_finder.extend(
                glob.glob(os.path.join(psf_kernel_path, f"*{band_name.upper()}*kernel*.fits"))
            )
            psf_kernel_path_finder.extend(
                glob.glob(os.path.join(psf_kernel_path, f"*{band_name.lower()}*PSF*.fits"))
            )

            psf_kernel_path_finder.extend(
                glob.glob(os.path.join(psf_kernel_path, f"*{band_name.upper()}*PSF*.fits"))
            )
            psf_kernel_path_finder.extend(
                glob.glob(os.path.join(psf_kernel_path, f"*{band_name.lower()}*PSF*.fits"))
            )
            psf_kernel_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_kernel_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*PSF*.fits",
                    )
                )
            )

            # Remove duplicates
            psf_kernel_path = list(set(psf_kernel_path_finder))

            if len(psf_kernel_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(psf_kernel_path) == 0:
                raise ValueError(
                    f"No PSF kernel files found for band {band_name}. Please provide full path."
                )
            else:
                psf_kernel_path = psf_kernel_path[0]
                print(f"Auto detected PSF kernel path {psf_kernel_path} for band {band_name}.")
        elif psf_kernel_path is not None and not os.path.exists(psf_kernel_path):
            raise ValueError(f"PSF kernel path {psf_kernel_path} does not exist.")

        if psf_matched_image_path is not None and os.path.isdir(psf_matched_image_path):
            psf_matched_image_path_finder = glob.glob(
                os.path.join(psf_matched_image_path, f"*{band_name.upper()}*psf*.fits")
            )
            psf_matched_image_path_finder.extend(
                glob.glob(os.path.join(psf_matched_image_path, f"*{band_name.lower()}*psf*.fits"))
            )
            psf_matched_image_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_matched_image_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*psf*.fits",
                    )
                )
            )
            psf_matched_image_path_finder.extend(
                glob.glob(os.path.join(psf_matched_image_path, f"*{band_name.upper()}*PSF*.fits"))
            )
            psf_matched_image_path_finder.extend(
                glob.glob(os.path.join(psf_matched_image_path, f"*{band_name.lower()}*PSF*.fits"))
            )
            psf_matched_image_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_matched_image_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*PSF*.fits",
                    )
                )
            )

            # Remove duplicates
            psf_matched_image_path = list(set(psf_matched_image_path_finder))

            if len(psf_matched_image_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(psf_matched_image_path) == 0:
                raise ValueError(
                    f"No PSF-matched images files found for band {band_name}. Please provide full path."
                )
            else:
                psf_matched_image_path = psf_matched_image_path[0]
                print(
                    f"Auto detected PSF matched image path {psf_matched_image_path} for band {band_name}."
                )
        elif psf_matched_image_path is not None and not os.path.exists(psf_matched_image_path):
            raise ValueError(f"PSF matched image path {psf_matched_image_path} does not exist.")

        if psf_matched_err_path is not None and os.path.isdir(psf_matched_err_path):
            psf_matched_err_path_finder = glob.glob(
                os.path.join(psf_matched_err_path, f"*{band_name.upper()}*psf*.fits")
            )
            psf_matched_err_path_finder.extend(
                glob.glob(os.path.join(psf_matched_err_path, f"*{band_name.lower()}*psf*.fits"))
            )
            psf_matched_err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_matched_err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*psf*.fits",
                    )
                )
            )
            psf_matched_err_path_finder.extend(
                glob.glob(os.path.join(psf_matched_err_path, f"*{band_name.upper()}*PSF*.fits"))
            )
            psf_matched_err_path_finder.extend(
                glob.glob(os.path.join(psf_matched_err_path, f"*{band_name.lower()}*PSF*.fits"))
            )
            psf_matched_err_path_finder.extend(
                glob.glob(
                    os.path.join(
                        psf_matched_err_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*PSF*.fits",
                    )
                )
            )

            # Remove duplicates
            psf_matched_err_path = list(set(psf_matched_err_path_finder))

            if len(psf_matched_err_path) > 1:
                raise ValueError(
                    f"Multiple files found for band {band_name}. Please provide full path."
                )
            elif len(psf_matched_err_path) == 0:
                raise ValueError(
                    f"No PSF-matched err files found for band {band_name}. Please provide full path."
                )
            else:
                psf_matched_err_path = psf_matched_err_path[0]
                print(
                    f"Auto detected PSF matched error path {psf_matched_err_path} for band {band_name}."
                )
        elif psf_matched_err_path is not None and not os.path.exists(psf_matched_err_path):
            raise ValueError(f"PSF matched error path {psf_matched_err_path} does not exist.")
        # Check if imagepath is a folder
        if os.path.isdir(image_path):
            # Use glob to find the image
            image_path_finder = glob.glob(os.path.join(image_path, f"*{band_name.upper()}*.fits"))
            image_path_finder.extend(
                glob.glob(os.path.join(image_path, f"*{band_name.lower()}*.fits"))
            )
            # Try lowercase first letter
            image_path_finder.extend(
                glob.glob(
                    os.path.join(
                        image_path,
                        f"*{band_name[0].lower()}{band_name[1:].upper()}*.fits",
                    )
                )
            )
            # Remove duplicates
            image_path = list(set(image_path_finder))

            if len(image_path) > 1:
                im_endings = [
                    "drz.fits",
                    "sci.fits",
                    "drz_aligned.fits",
                    "sci_aligned.fits",
                    "drz_sci.fits",
                    "drz_sci_aligned.fits",
                    "drz_matched.fits",
                    "sci_matched.fits",
                    "scixy.fits",
                    "scicutoutxy.fits",
                    "drz_xy.fits",
                    "drzxy.fits",
                    "i2dnobg.fits",
                    "i2d.fits",
                    "i2dnobgnobg.fits",
                    "i2dnobgxy.fits",
                ]
                # See if any one of the files has a common ending
                good_endings = []
                for ending in im_endings:
                    test_image_path = [im for im in image_path if ending in im]
                    good_endings.extend(test_image_path)
                if len(good_endings) == 1:
                    image_path = good_endings[0]
                    print(f"Auto detected image path {image_path} for band {band_name}.")
                elif len(good_endings) > 1:
                    raise ValueError(
                        f"Multiple files found for band {band_name} even when guessing ending. Please provide full path."
                    )
                else:
                    raise ValueError(
                        f"No image files found for band {band_name}. Please provide full path."
                    )

            elif len(image_path) == 0:
                raise ValueError(
                    f"No image files found for band {band_name}. Please provide full path."
                )
            else:
                image_path = image_path[0]

        if psf_matched_image_path is not None:
            assert (
                psf_matched_err_path is not None
            ), "Must provide a PSF matched error path if PSF matched image path is provided."
            assert (
                not psf_matched
            ), "If providing both PSF matched and un-matched images, set psf_matched = False."
            assert (
                image_path != psf_matched_image_path
            ), "PSF matched image path and image path must be different. If only providing PSF matched images, set image_path  only."

        self.survey = survey
        self.image_path = image_path
        self.wht_path = wht_path
        self.err_path = err_path
        self.seg_path = seg_path
        self.psf_path = psf_path
        self.psf_kernel_path = psf_kernel_path

        self.psf_type = psf_type

        self.im_pixel_scale = im_pixel_scale
        self.image_zp = image_zp
        self.image_unit = image_unit
        self.im_hdu_ext = im_hdu_ext
        self.wht_hdu_ext = wht_hdu_ext
        self.err_hdu_ext = err_hdu_ext
        self.seg_hdu_ext = seg_hdu_ext

        self.psf_matched = psf_matched
        self.psf_matched_image_path = psf_matched_image_path
        self.psf_matched_err_path = psf_matched_err_path

        if self.psf_path is not None or self.psf_kernel_path is not None:
            assert self.psf_type is not None, "Must provide a PSF type if PSF path is provided."

        if self.instrument == "HEADER":
            instruments = {
                "F435W": "ACS_WFC",
                "F606W": "ACS_WFC",
                "F775W": "ACS_WFC",
                "F814W": "ACS_WFC",
                "F850LP": "ACS_WFC",
                "F090W": "NIRCam",
                "F105W": "WFC3IR",
                "F110W": "WFC3IR",
                "F115W": "NIRCam",
                "F125W": "WC3IR",
                "F140W": "NIRCam",
                "F140M": "WFC3IR",
                "F150W": "WFC3IR",
                "F160W": "WFC3IR",
                "F162M": "NIRCam",
                "F182M": "WFC3IR",
                "F200W": "NIRCam",
                "F210M": "NIRCam",
                "F250M": "NIRCam",
                "F277W": "NIRCam",
                "F300M": "NIRCam",
                "F335M": "NIRCam",
                "F356W": "NIRCam",
                "F360M": "NIRCam",
                "F410M": "NIRCam",
                "F430M": "NIRCam",
                "F444W": "NIRCam",
                "F460M": "NIRCam",
                "F480M": "NIRCam",
                "F560W": "MIRI",
                "F770W": "MIRI",
                "F1000W": "MIRI",
                "F1130W": "MIRI",
                "F1280W": "MIRI",
                "F1500W": "MIRI",
                "F1800W": "MIRI",
                "F2100W": "MIRI",
                "F2550W": "MIRI",
            }
            self.instrument = instruments[self.band_name.upper()]
            print(f"Auto detected instrument {self.instrument} for band {self.band_name}.")

        hdulist = fits.open(self.image_path, ignore_missing_simple=True)
        names = [hdu.name for hdu in hdulist]

        if "SCI" in names and "WHT" in names and "ERR" in names:
            print(f"Detected single HDUList for {self.band_name}.")
            print(
                "Assuming JWST style HDUList with PrimaryHDU (0), SCI [1],  WHT [2], and ERR [3]."
            )
            if self.im_hdu_ext == 0:
                self.im_hdu_ext = names.index("SCI")
            if self.wht_hdu_ext == 0:
                self.wht_hdu_ext = names.index("WHT")
            if self.err_hdu_ext == 0:
                self.err_hdu_ext = names.index("ERR")

            self.err_path = self.image_path
            self.wht_path = self.image_path

        # Open image to get header

        im_header = fits.open(self.image_path, ignore_missing_simple=True)[self.im_hdu_ext].header
        if self.image_zp == "HEADER":
            phot_zp_keywords = [
                "PHOTZP",
                "MAGZPT",
                "MAGZEROPOINT",
                "MAGZP",
                "MAGZERO",
                "ZEROPNT",
                "ZP",
                "ZEROPOINT",
                "MAGABZPT",
                "MAGABZERO",
                "MAGABZP",
                "MAGABZEROPOINT",
            ]
            for key in phot_zp_keywords:
                if key in im_header:
                    self.image_zp = im_header[key]
                    print(
                        f"Auto detected zero point {self.image_zp} with keyword {key} for band {self.band_name}."
                    )
                    break

            if self.image_zp == "HEADER":
                # Attempt to calculate from photometry keywords
                if "PHOTFLAM" in im_header and "PHOTPLAM" in im_header:
                    self.image_zp = (
                        -2.5 * np.log10(im_header["PHOTFLAM"])
                        - 21.10
                        - 5 * np.log10(im_header["PHOTPLAM"])
                        + 18.6921
                    )
                    print(
                        f"Auto calculated zero point {self.image_zp} from PHOTFLAM and PHOTPLAM for band {self.band_name}."
                    )
                    print("WARNING! THIS IS CALIBRATED ONLY TO ACS_WFC")
        if self.image_zp == "HEADER":
            self.image_zp = None

        if self.image_unit == "HEADER":
            phot_unit_keywords = ["BUNIT", "UNITS", "UNIT", "BUNITS"]
            for key in phot_unit_keywords:
                if key in im_header:
                    self.image_unit = im_header[key]
                    print(
                        f"Auto detected unit {self.image_unit} with keyword {key} for band {self.band_name}."
                    )
                    break
        if self.image_unit == "HEADER":
            self.image_unit = None

        assert (self.image_zp != "HEADER") or (
            self.image_unit != "HEADER"
        ), f"Failed to detect zero point or unit for band {self.band_name}."

        # if both found, print warning and prefer unit

        if (self.image_zp not in ["HEADER", None]) and (self.image_unit not in ["HEADER", None]):
            print(
                f"Both zero point and unit found for band {self.band_name}. Using ZP {self.image_zp}."
            )
            self.image_unit = None

        # Get pixel scale from header

        if self.im_pixel_scale == "HEADER":
            pixel_scale_keywords = [
                "PIXSCALE",
                "PIXELSCL",
                "PIXELSC",
                "PIXSCAL",
                "PIXSCA",
                "PIXSC",
            ]
            for key in pixel_scale_keywords:
                if key in im_header:
                    self.im_pixel_scale = im_header[key]
                    print(
                        f"Auto detected pixel scale {self.im_pixel_scale} with keyword {key} for band {self.band_name}."
                    )
                    break
            if self.im_pixel_scale == "HEADER":
                # Get from WCS
                wcs = WCS(im_header)
                # Get the pixel scale matrix
                cd = wcs.pixel_scale_matrix

                # Calculate the pixel scale in degrees
                scale_deg = np.sqrt(np.abs(np.linalg.det(cd)))

                # Convert to arcseconds
                scale_arcsec = scale_deg * 3600
                self.im_pixel_scale = scale_arcsec * u.arcsec

                if scale_arcsec == 3600 * u.arcsec:
                    print(
                        "Warning! Detected pixel scale from WCS is 1 degree/pixel, which is likely incorrect. Setting to None."
                    )
                    self.im_pixel_scale = None
                else:
                    print(
                        f"Auto detected pixel scale {self.im_pixel_scale:.4f} arcsec/pixel from WCS for band {self.band_name}."
                    )

        if self.image_unit == "MJy/sr":
            # compute zeropoint given pixel scale
            area_sr = (self.im_pixel_scale.to(u.radian).value) ** 2
            self.image_zp = -2.5 * np.log10(1e6 * area_sr) + 8.90

    @property
    def phot_data(self):
        with fits.open(self.image_path, ignore_missing_simple=True) as hdul:
            data = hdul[self.im_hdu_ext].data
            if data is None:
                raise ValueError(f"No data found in image {self.image_path}.")
            return data

    @property
    def wht_data(self):
        if self.wht_path is None:
            return None
        with fits.open(self.wht_path, ignore_missing_simple=True) as hdul:
            data = hdul[self.wht_hdu_ext].data
            if data is None:
                raise ValueError(f"No data found in weight image {self.wht_path}.")
            return data

    @property
    def err_data(self):
        if self.err_path is None:
            return None
        with fits.open(self.err_path, ignore_missing_simple=True) as hdul:
            data = hdul[self.err_hdu_ext].data
            if data is None:
                raise ValueError(f"No data found in error image {self.err_path}.")
            return data

    @property
    def seg_data(self):
        if self.seg_path is None:
            return None
        with fits.open(self.seg_path, ignore_missing_simple=True) as hdul:
            data = hdul[self.seg_hdu_ext].data
            if data is None:
                raise ValueError(f"No data found in segmentation image {self.seg_path}.")
            return data

    @property
    def psf_data(self):
        if self.psf_path is None:
            return None
        with fits.open(self.psf_path, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No data found in PSF image {self.psf_path}.")
            return data

    @property
    def psf_kernel_data(self):
        if self.psf_kernel_path is None:
            return None
        with fits.open(self.psf_kernel_path, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No data found in PSF kernel image {self.psf_kernel_path}.")
            return data

    @property
    def psf_matched_image_data(self):
        if self.psf_matched_image_path is None:
            return None
        with fits.open(self.psf_matched_image_path, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(
                    f"No data found in PSF matched image {self.psf_matched_image_path}."
                )
            return data

    @property
    def psf_matched_err_data(self):
        if self.psf_matched_err_path is None:
            return None
        with fits.open(self.psf_matched_err_path, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(
                    f"No data found in PSF matched error image {self.psf_matched_err_path}."
                )
            return data


class FieldInfo:
    """
    Just a container for a list of PhotometryBandInfo objects.
    """

    def __init__(self, band_info_list):
        self.band_info_list = band_info_list
        if len(band_info_list) == 0:
            raise ValueError("band_info_list cannot be empty.")
        # Check only one or no detection image
        detection_images = [band.detection_image for band in band_info_list]
        if detection_images.count(True) > 1:
            raise ValueError("More than one detection image found in band_info_list.")

        elif detection_images.count(True) == 1:
            self.detection_band_name = [
                band.band_name for band in band_info_list if band.detection_image
            ][0]
            self.detection_band = band_info_list[detection_images.index(True)]
            # Remove detection image from list
            self.band_info_list = [band for band in band_info_list if not band.detection_image]
        else:
            self.detection_band = None
            self.detection_band_name = None
            self.band_info_list = band_info_list

        # Get list of bands
        surveys = [band.survey for band in self.band_info_list]
        assert len(set(surveys)) == 1, f"All bands must be from the same survey: {surveys}"
        self.survey = surveys[0]

        self.band_names = [band.band_name for band in self.band_info_list]
        # check unique
        assert len(self.band_names) == len(set(self.band_names)), "Band names must be unique."
        # Get list of instruments
        self.instruments = [band.instrument for band in self.band_info_list]

        self.im_pixel_scales = {band.band_name: band.im_pixel_scale for band in self.band_info_list}
        self.im_zps = {band.band_name: band.image_zp for band in self.band_info_list}
        self.im_units = {band.band_name: band.image_unit for band in self.band_info_list}
        self.im_exts = {band.band_name: band.im_hdu_ext for band in self.band_info_list}
        self.wht_exts = {band.band_name: band.wht_hdu_ext for band in self.band_info_list}
        self.rms_err_exts = {band.band_name: band.err_hdu_ext for band in self.band_info_list}
        self.seg_exts = {band.band_name: band.seg_hdu_ext for band in self.band_info_list}

        self.im_paths = {band.band_name: band.image_path for band in self.band_info_list}
        self.wht_paths = {band.band_name: band.wht_path for band in self.band_info_list}
        self.err_paths = {band.band_name: band.err_path for band in self.band_info_list}
        self.seg_paths = {band.band_name: band.seg_path for band in self.band_info_list}

        self.psf_paths = {band.band_name: band.psf_path for band in self.band_info_list}

        self.psf_kernel_paths = {
            band.band_name: band.psf_kernel_path for band in self.band_info_list
        }

        self.psf_matched = {band.band_name: band.psf_matched for band in self.band_info_list}
        self.all_psf_types = {band.band_name: band.psf_type for band in self.band_info_list}
        self.all_psf_matched = all(self.psf_matched.values())
        self.any_psf_matched = any(self.psf_matched.values())
        self.psf_matched_band = set(self.psf_matched.values())
        assert (
            len(self.psf_matched_band) < 2
        ), "All bands must be matched to the same band if matched."
        if len(self.psf_matched_band) == 1 and list(self.psf_matched_band)[0] is not False:
            print(f"Matched band is {list(self.psf_matched_band)[0]}.")
            self.psf_matched_band = list(self.psf_matched_band)[0]
            assert self.psf_matched_band in self.band_names, "Matched band must be in band list."
            assert (
                self.psf_matched_band == self.band_names[-1]
            ), "The assumed convention is to match to the last band."
        else:
            self.psf_matched_band = None

        one_type = list(set(self.all_psf_types.values()))
        # remove None
        one_type = [ptype for ptype in one_type if ptype is not None]
        if len(one_type) == 1:
            self.psf_type = one_type[0]
        elif len(one_type) == 0:
            self.psf_type = None
        else:
            raise ValueError(f"All PSFs must be of the same type, got {one_type}.")

        self.any_psf_kernel = any([path is not None for path in self.psf_kernel_paths.values()])
        self.all_psf_kernel = all([path is not None for path in self.psf_kernel_paths.values()])

        all_auto_keys = set(
            [key for band in self.band_info_list for key in band.auto_photometry.keys()]
        )
        all_aperture_sizes = set(
            [key for band in self.band_info_list for key in band.aperture_photometry.keys()]
        )
        # Now get all keys inside each aperture size
        all_aperture_size_keys = set(
            [
                key
                for band in self.band_info_list
                for key in band.aperture_photometry.keys()
                for key in band.aperture_photometry[key].keys()
            ]
        )

        # fix
        self.auto_photometry = {
            key: [band.auto_photometry.get(key, None) for band in self.band_info_list]
            for key in all_auto_keys
        }
        self.aperture_photometry = {
            key: [band.aperture_photometry.get(key, None) for band in self.band_info_list]
            for key in all_aperture_sizes
        }

        self.psf_folder = "internal"
        self.psf_kernel_folder = "internal"

        if self.any_psf_matched:
            # Check if all PSFs are in the same folder
            psf_folders = set([os.path.dirname(path) for path in self.psf_paths.values()])
            if len(psf_folders) == 1:
                self.psf_folder = psf_folders.pop()

        if self.any_psf_kernel:
            # Check if all PSFs are in the same folder
            psf_kernel_folders = []
            for path in self.psf_kernel_paths.values():
                if path is not None:
                    psf_kernel_folders.append(os.path.dirname(path))
            if len(psf_kernel_folders) == 1:
                self.psf_kernel_folder = psf_kernel_folders.pop()

        self.psf_matched_image_paths = {
            band.band_name: band.psf_matched_image_path for band in self.band_info_list
        }
        self.psf_matched_err_paths = {
            band.band_name: band.psf_matched_err_path for band in self.band_info_list
        }

        self.any_psf_matched_image = any(
            [path is not None for path in self.psf_matched_image_paths.values()]
        )
        self.all_psf_matched_image = all(
            [path is not None for path in self.psf_matched_image_paths.values()]
        )
        self.any_psf_matched_err = any(
            [path is not None for path in self.psf_matched_err_paths.values()]
        )
        self.all_psf_matched_err = all(
            [path is not None for path in self.psf_matched_err_paths.values()]
        )

    def __str__(self):
        """
        Produces an ASCII style table. List bands, instruments, pixel scales, zeropoints, PSF_matched status, and PSF type.
        Also shows availability of error maps, weight maps, segmentation maps, PSFs, PSF kernels,
        auto photometry, and aperture photometry for each band.
        If None fill with --
        """
        # Define headers
        headers = [
            "Band",
            "Instrument",
            "Pixel Scale",
            "ZP/Unit",
            "PSF Matched",
            "PSF Type",
            "Err",
            "Wht",
            "Seg",
            "PSF",
            "PSF Kernel",
            "Auto Phot",
            "Aper Phot",
        ]

        # Collect data rows
        rows = []
        for band_name in self.band_names:
            instrument = next(
                (band.instrument for band in self.band_info_list if band.band_name == band_name),
                "--",
            )
            pixel_scale = self.im_pixel_scales.get(band_name, "--")
            zero_point = self.im_zps.get(band_name, "--")
            if zero_point == "--" or zero_point is None:
                zero_point = self.im_units.get(band_name, "--")
            psf_matched = "Yes" if self.psf_matched.get(band_name, False) else "No"
            psf_type = self.all_psf_types.get(band_name, "--")

            # Check for existence of various data products
            has_err = "Yes" if self.err_paths.get(band_name) is not None else "No"
            has_wht = "Yes" if self.wht_paths.get(band_name) is not None else "No"
            has_seg = "Yes" if self.seg_paths.get(band_name) is not None else "No"
            has_psf = "Yes" if self.psf_paths.get(band_name) is not None else "No"
            has_psf_kernel = "Yes" if self.psf_kernel_paths.get(band_name) is not None else "No"

            # Check for auto and aperture photometry
            has_auto_phot = "No"
            for key in self.auto_photometry:
                if self.auto_photometry[key][self.band_names.index(band_name)] is not None:
                    has_auto_phot = "Yes"
                    break

            has_aper_phot = "No"
            for aperture_size in self.aperture_photometry:
                if (
                    self.aperture_photometry[aperture_size][self.band_names.index(band_name)]
                    is not None
                ):
                    has_aper_phot = "Yes"
                    break

            # Replace None values with "--"
            pixel_scale = "--" if pixel_scale is None else f"{pixel_scale:.3f}"
            zero_point = (
                "--"
                if zero_point is None
                else f"{np.round(zero_point, 3) if isinstance(zero_point, float) else zero_point}"
            )
            psf_type = "--" if psf_type is None else psf_type

            rows.append(
                [
                    band_name,
                    instrument,
                    pixel_scale,
                    zero_point,
                    psf_matched,
                    psf_type,
                    has_err,
                    has_wht,
                    has_seg,
                    has_psf,
                    has_psf_kernel,
                    has_auto_phot,
                    has_aper_phot,
                ]
            )

        # Calculate column widths based on content
        col_widths = []
        for i in range(len(headers)):
            col_width = len(headers[i])
            for row in rows:
                col_width = max(col_width, len(str(row[i])))
            col_widths.append(col_width + 2)  # Add padding

        # Create the table string
        result = []
        result.append(f"Field Info for Survey: {self.survey}")

        # Add detection band info if present
        if self.detection_band:
            result.append(f"Detection Band: {self.detection_band_name}")
            result.append("")

        # Add header row
        header_row = "".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        result.append(header_row)

        # Add separator
        separator = "".join("-" * width for width in col_widths)
        result.append(separator)

        # Add data rows
        for row in rows:
            row_str = "".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))
            result.append(row_str)

        # Add additional information
        result.append("")
        result.append(f"All PSF Matched: {self.all_psf_matched}")
        result.append(f"Any PSF Matched: {self.any_psf_matched}")
        result.append(f"PSF Type: {self.psf_type or '--'}")

        # Add information about availability of various data products
        result.append("")
        result.append(f"All PSF Kernels Available: {self.all_psf_kernel}")
        result.append(f"Any PSF Kernels Available: {self.any_psf_kernel}")

        result.append("")
        result.append(f"All PSF Matched Images Available: {self.all_psf_matched_image}")
        result.append(f"Any PSF Matched Images Available: {self.any_psf_matched_image}")

        result.append("")
        result.append(f"All PSF Matched Error Maps Available: {self.all_psf_matched_err}")
        result.append(f"Any PSF Matched Error Maps Available: {self.any_psf_matched_err}")

        # Add path information
        if self.any_psf_kernel:
            result.append("")
            result.append(f"PSF Kernel Folder: {self.psf_kernel_folder}")

        if self.any_psf_matched:
            result.append("")
            result.append(f"PSF Folder: {self.psf_folder}")

        # Available photometry information
        if self.auto_photometry:
            result.append("")
            result.append("Auto Photometry Keys: " + ", ".join(self.auto_photometry.keys()))

        if self.aperture_photometry:
            result.append("")
            result.append(
                "Aperture Sizes: " + ", ".join(str(k) for k in self.aperture_photometry.keys())
            )

        return "\n".join(result)

    def __repr__(self):
        return self.__str__()

    # Make supscribale over band_info_list

    def __iter__(self):
        return iter(self.band_info_list)

    def __len__(self):
        return len(self.band_info_list)

    def __getitem__(self, key):
        return self.band_info_list[key]

    def __next__(self):
        return next(self.band_info_list)

    def __contains__(self, item):
        return item in self.band_info_list


def compass(
    ra,
    dec,
    wcs,
    axis,
    arrow_length=0.5 * u.arcsec,
    x_ax="ra",
    ang_text=False,
    arrow_width=200,
    arrow_color="black",
    text_color="black",
    fontsize="large",
    return_ang=False,
    pix_scale=0.03 * u.arcsec,
    compass_text_scale_factor=1.15,
):
    def calculate_arrow_vectors(angle_radians, length):
        dx = length * np.sin(angle_radians)
        dy = length * np.cos(angle_radians)
        return dx, dy

    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    origin_coord = SkyCoord(ra, dec, unit=u.deg)
    bottom_coord = wcs.wcs_pix2world(xlim[0], ylim[0], 1)
    top_coord = wcs.wcs_pix2world(xlim[1], ylim[1], 1)

    size_x = abs(bottom_coord[0] - top_coord[0])
    size_y = abs(bottom_coord[1] - top_coord[1])
    north_coord = SkyCoord(ra, dec + size_y, unit=u.deg)
    east_coord = SkyCoord(ra + size_x, dec, unit=u.deg)

    origin_coord_pix = wcs.wcs_world2pix(origin_coord.ra.degree, origin_coord.dec.degree, 1)
    north_coord_pix = wcs.wcs_world2pix(north_coord.ra.degree, north_coord.dec.degree, 1)
    east_coord_pix = wcs.wcs_world2pix(east_coord.ra.degree, east_coord.dec.degree, 1)

    if x_ax == "ra":
        ang = np.arctan2(
            north_coord_pix[0] - origin_coord_pix[0],
            north_coord_pix[1] - origin_coord_pix[1],
        )
    elif x_ax == "dec":
        ang = np.arctan2(
            north_coord_pix[1] - origin_coord_pix[1],
            north_coord_pix[0] - origin_coord_pix[0],
        )
    else:
        raise ValueError("x_ax must be 'ra' or 'dec'.")

    if return_ang:
        return np.degrees(ang)

    arrow_length = arrow_length.to(u.arcsec).value / pix_scale.to(u.arcsec).value

    dx_north, dy_north = calculate_arrow_vectors(ang, arrow_length)

    # Calculate east vector (perpendicu
    #
    # lar to north)
    east_angle = ang - np.pi / 2

    dx_east, dy_east = calculate_arrow_vectors(east_angle, arrow_length)

    # Check if east is within ±π/2 of north and flip if necessary

    offset = 1.3 * (arrow_width / 2)  # Half the arrow width
    offset_angle = ang - np.pi / 4  # 45 degrees between north and east
    dx_offset, dy_offset = calculate_arrow_vectors(offset_angle, offset)

    north = FancyArrow(
        origin_coord_pix[0],
        origin_coord_pix[1],
        dx_north,
        dy_north,
        color=arrow_color,
        width=arrow_width,
        length_includes_head=True,
        head_width=2.2 * arrow_width,
        head_length=1.3 * 2.2 * arrow_width,
    )

    east = FancyArrow(
        origin_coord_pix[0] + dx_offset,
        origin_coord_pix[1] + dy_offset,
        dx_east,
        dy_east,
        color=arrow_color,
        width=arrow_width,
        length_includes_head=True,
        head_width=2.2 * arrow_width,
        head_length=1.3 * 2.2 * arrow_width,
    )

    axis.add_patch(north)
    axis.add_patch(east)

    if x_ax == "ra":
        north_label, east_label = "N", "E"
    elif x_ax == "dec":
        north_label, east_label = "E", "N"

    north_text_coord = (
        origin_coord_pix[0] + compass_text_scale_factor * dx_north,
        origin_coord_pix[1] + compass_text_scale_factor * dy_north,
    )
    east_text_coord = (
        origin_coord_pix[0] + compass_text_scale_factor * (dx_east + dx_offset),
        origin_coord_pix[1] + compass_text_scale_factor * (dy_east + dx_offset),
    )
    axis.text(
        *north_text_coord,
        north_label,
        color=text_color,
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight="bold",
    )
    axis.text(
        *east_text_coord,
        east_label,
        color=text_color,
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight="bold",
    )


# Example usage
"""
fig, ax = plt.subplots()
wcs_compass(ra, dec, wcs, ax)
plt.show()
"""
# Example usage
"""
fig, ax = plt.subplots()
wcs_compass(ra, dec, wcs, ax)
plt.show()
"""
# axis.text(north_coord_pix[0] + scale_x_text*np.cos((90-ang)*np.pi/180), north_coord_pix[1]+scale_x_text * np.sin((90-ang)*np.pi/180), x_label, color=text_color, fontsize=fontsize, ha='center', va='center')# fontweight="bold")
# axis.text(east_coord_pix[0] - scale_y_text * np.cos(ang*np.pi/180), east_coord_pix[1] + scale_y_text * np.sin(ang*np.pi/180), y_label, color=text_color, fontsize=fontsize, ha='center', va='center')# fontweight="bold")
# north_angle = wcs.wcs.cd[1,1]
# east_angle = wcs.wcs.cd[0,0]

# print(height_y)
# print(east_coord_pix[0]-base_coord_pix[0])

# size_x_pix = abs(xlim[0]-xlim[1])
# size_y_pix = abs(ylim[0]-ylim[1])


def find_dict_differences(dict1, dict2, path=""):
    """
    Compare two nested dictionaries and return their differences.

    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
        path: Current path in the nested structure (used for recursion)

    Returns:
        dict: Dictionary containing three keys:
            - 'added': Keys present in dict2 but not in dict1
            - 'removed': Keys present in dict1 but not in dict2
            - 'modified': Keys present in both but with different values
    """
    differences = {"added": {}, "removed": {}, "modified": {}}

    # Handle cases where either input is None
    if dict1 is None:
        if dict2 is not None:
            differences["added"] = dict2
        return differences
    if dict2 is None:
        differences["removed"] = dict1
        return differences

    # Compare keys in both dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        # Key only in dict2
        if key not in dict1:
            differences["added"][current_path] = dict2[key]
            continue

        # Key only in dict1
        if key not in dict2:
            differences["removed"][current_path] = dict1[key]
            continue

        # If both values are dictionaries, recurse
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_diff = find_dict_differences(dict1[key], dict2[key], current_path)

            # Merge nested differences with current level
            for diff_type in ["added", "removed", "modified"]:
                differences[diff_type].update(nested_diff[diff_type])

        # If values are list or numpy array, compare element-wise
        elif type(dict1[key]) in [list, np.ndarray, tuple]:
            if not np.array_equal(dict1[key], dict2[key]):
                differences["modified"][current_path] = {
                    "old": dict1[key],
                    "new": dict2[key],
                }
        elif dict1[key] != dict2[key]:
            differences["modified"][current_path] = {
                "old": dict1[key],
                "new": dict2[key],
            }

    return differences


def measure_cog(sci_cutout, pos, nradii=20, minrad=1, maxrad=10):
    """Measure the curve of growth of the galaxy in the cutout image"""
    # make COG by apertures

    radii = np.linspace(minrad, maxrad, nradii)

    apertures = [CircularAperture(pos, r) for r in radii]
    phot_tab = aperture_photometry(sci_cutout, apertures)
    # print(phot_tab)
    cog = np.array([[phot_tab[coln][0] for coln in phot_tab.colnames[3:]]][0])

    return radii, cog


def get_bounding_box(array, scale_border=0, square=False, init_buffer=2, debug=False):
    """Get bounding box coordinates of non-NaN values in 2D array.

    Parameters
    ----------
    array : ndarray
        2D numpy array
    scale_border : float
        Factor to scale the border by, as a fraction of box size
    square : bool
        Whether to make the bounding box square

    Returns
    -------
    tuple
        (xmin, xmax, ymin, ymax) coordinates. Returns None if all values are NaN.
    """
    valid = ~np.isnan(array)
    if not np.any(valid):
        return None

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)

    # Find the indices of the first and last True values
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Add a 2 pixel buffer by default as the bounding box tends to overlap the edge pixels

    ymin -= init_buffer
    ymax += init_buffer
    xmin -= init_buffer
    xmax += init_buffer

    # print(f"Bounding box: {xmin}, {xmax}, {ymin}, {ymax}")

    if debug:
        fig, ax = plt.subplots()
        from matplotlib.patches import Rectangle

        ax.imshow(array, origin="lower", cmap="gray")
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor="red",
                facecolor="none",
                label="Original",
            )
        )

    # Store original dimensions for border scaling
    diff_x = xmax - xmin
    diff_y = ymax - ymin

    # Apply border scaling
    xmin = max(0, xmin - int(scale_border * diff_x))
    xmax = min(array.shape[1] - 1, xmax + int(scale_border * diff_x))
    ymin = max(0, ymin - int(scale_border * diff_y))
    ymax = min(array.shape[0] - 1, ymax + int(scale_border * diff_y))

    if debug:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor="blue",
                facecolor="none",
                label="Scaled",
            )
        )

    if square:
        # Recalculate dimensions after border scaling
        diff_x = xmax - xmin
        diff_y = ymax - ymin

        if diff_x > diff_y:
            # Need to expand y dimension
            diff = diff_x - diff_y
            pad_top = diff // 2
            pad_bottom = diff - pad_top  # Handle odd differences correctly

            ymin = max(0, ymin - pad_top)
            ymax = min(array.shape[0] - 1, ymax + pad_bottom)

        elif diff_y > diff_x:
            # Need to expand x dimension
            diff = diff_y - diff_x
            pad_left = diff // 2
            pad_right = diff - pad_left  # Handle odd differences correctly

            xmin = max(0, xmin - pad_left)
            xmax = min(array.shape[1] - 1, xmax + pad_right)

        if debug:
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    edgecolor="green",
                    facecolor="none",
                    label="Square",
                )
            )

    if debug:
        ax.legend()
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plt.savefig(os.path.join(current_dir, "plots/bounding_box.png"))

    return xmin, xmax, ymin, ymax


def calculate_half_radius(
    map_data,
    center="com",
    mask=None,
    replace_nan=True,
    replace_val=0,
    radius_step=1.0,
    max_radius=None,
    method="exact",
):
    """Calculate radius containing half of total value using circular aperture photometry.

    Parameters
    ----------
    map_data : array_like
        2D array containing surface density values (e.g. Msun/kpc^2 or SFR/kpc^2)
    center : tuple, optional
        (x,y) coordinates of center. If None, uses center of array.
    mask : array_like, optional
        Boolean mask of valid pixels. Invalid pixels ignored.
    radius_step : float
        Step size in pixels for radius search
    max_radius : float, optional
        Maximum radius to try. If None, uses half the image size.
    method : {'exact', 'center'}
        Method for treating pixels that span aperture boundaries:
        - 'exact': Exact overlap fraction (slower)
        - 'center': Include if center is in aperture (faster)

    Returns
    -------
    radius : float
        Radius in pixels containing half the total value
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    # Set maximum radius if not provided
    if max_radius is None:
        max_radius = min(map_data.shape) / 2

    # Calculate radii to try
    radii = np.arange(radius_step, max_radius, radius_step)

    if replace_nan:
        map_data = np.nan_to_num(map_data, nan=replace_val)

    # Set center if not provided
    if center is None:
        center = (map_data.shape[1] / 2, map_data.shape[0] / 2)
    elif center == "com":
        x, y = np.meshgrid(np.arange(map_data.shape[1]), np.arange(map_data.shape[0]))
        x = x.flatten()
        y = y.flatten()
        m = map_data.flatten()
        x_com = np.nansum(x * m) / np.nansum(m)
        y_com = np.nansum(y * m) / np.nansum(m)
        center = (x_com, y_com)

    # Get masked data if mask provided
    if mask is not None:
        data = map_data.copy()
        data[~mask] = 0
    else:
        data = map_data

    # Get total value using large aperture
    total_aper = CircularAperture(center, r=max_radius)
    total_phot = aperture_photometry(data, total_aper, method=method)
    total = float(total_phot["aperture_sum"])

    # Try apertures of increasing size
    cumsum = []
    for r in radii:
        aper = CircularAperture(center, r=r)
        phot = aperture_photometry(data, aper, method=method)
        cumsum.append(float(phot["aperture_sum"]))

    cumsum = np.array(cumsum)

    # Find radius where cumsum crosses half total
    idx = np.searchsorted(cumsum, total / 2)

    if idx == 0:
        return radii[0], center
    elif idx == len(radii):
        print("Warning: Half-radius search reached maximum radius")
        return radii[-1], center

    # Interpolate between bracketing radii
    r1, r2 = radii[idx - 1 : idx + 1]
    m1, m2 = cumsum[idx - 1 : idx + 1]
    radius = r1 + (r2 - r1) * (total / 2 - m1) / (m2 - m1)

    return radius, center


def calculate_radial_profile(
    map_2d, center, bin_size=1.0, max_radius=None, nrad=None, statistic="mean"
):
    """
    Calculate the radial profile of a 2D map.

    Parameters:
    -----------
    map_2d : 2D numpy array
        The input 2D map/image
    center : tuple of (x, y)
        The center coordinates for the radial calculation
    bin_size : float
        The width of each radial bin
    nrad: int
        Number of radial bins
    max_radius : float or None
        Maximum radius to calculate. If None, will use the maximum possible radius
    statistic : str
        The statistic to calculate in each radial bin ('mean' or 'median')

    Returns:
    --------
    radii : numpy array
        The radial distances (center of each bin)
    profile : numpy array
        The calculated profile values
    std_profile : numpy array
        The standard deviation in each radial bin
    """
    # Create coordinate grids
    y, x = np.indices(map_2d.shape)

    # Calculate radial distance for each pixel
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Determine maximum radius if not specified
    if max_radius is None:
        max_radius = np.max(r)

    # Create radial bins
    if nrad is not None:
        bin_size = max_radius / nrad

    rbins = np.arange(0, max_radius + bin_size, bin_size)

    # Initialize arrays for results
    profile = np.zeros(len(rbins) - 1)
    std_profile = np.zeros(len(rbins) - 1)
    radii = (rbins[1:] + rbins[:-1]) / 2

    # Calculate statistics for each radial bin
    for i in range(len(rbins) - 1):
        # Create mask for current annulus
        mask = (r >= rbins[i]) & (r < rbins[i + 1])

        if np.any(mask):
            values = map_2d[mask]

            if statistic == "mean":
                profile[i] = np.nanmean(values)
            elif statistic == "median":
                profile[i] = np.nanmedian(values)
            else:
                raise ValueError("Statistic must be either 'mean' or 'median'")

            std_profile[i] = np.std(values)
        else:
            profile[i] = np.nan
            std_profile[i] = np.nan

    return radii, profile  # , std_profile


def create_padded_colormap(custom_colors_dict, base_cmap_name="nipy_spectral_r", total_length=256):
    """
    Create a new colormap that starts with specified colors and is padded with an existing colormap.

    Parameters:
    -----------
    base_cmap_name : str
        Name of the matplotlib colormap to use for padding
    custom_colors : dict
        Dict of color and integer key pairs to place at the start of the colormap
    total_length : int, optional
        Total number of colors in the final colormap (default: 256)

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        The new combined colormap
    """
    # Convert input colors to RGB if they aren't already
    n_custom = len(custom_colors_dict)

    keys = list(custom_colors_dict.keys())
    assert np.max(keys) <= total_length, "Custom colors exceed total length of requested colors"
    assert [type(key) == int for key in keys], "Keys must be integer values"
    ordered_keys = np.sort(keys)
    custom_colors = [to_rgba(custom_colors_dict[key]) for key in ordered_keys]

    # Can assume ordered keys are integer values. Need to fill missing integer positions up to total_length with cmap colors
    base_cmap = plt.get_cmap(base_cmap_name)

    n_base = total_length - n_custom

    missing_keys = np.setdiff1d(np.arange(total_length), ordered_keys)

    if n_base == 0:
        return ListedColormap(custom_colors)

    base_colors = base_cmap(np.linspace(0, 1, n_base))
    # Need to insert custom colors in correct positions
    new_colors = np.zeros((total_length, 4))
    new_colors[ordered_keys] = custom_colors
    new_colors[missing_keys] = base_colors

    new_cmap = ListedColormap(new_colors)

    return new_cmap


import numpy as np


def optimize_sfh_xlimit(ax, mass_threshold=0.001, buffer_fraction=0.2):
    """
    Optimizes the x-axis limits of a matplotlib plot containing SFR histories
    to focus on periods after each galaxy has formed a certain fraction of its final mass.
    Calculates cumulative mass from SFR data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object containing the SFR plots (SFR/yr vs time)
    mass_threshold : float, optional
        Fraction of final stellar mass to use as threshold (default: 0.01 for 1%)
    buffer_fraction : float, optional
        Fraction of the active time range to add as buffer (default: 0.1)

    Returns:
    --------
    float
        The optimal maximum x value for the plot
    """

    # Get all lines from the plot
    lines = ax.get_lines()
    if not lines:
        raise ValueError("No lines found in the plot")

    # Initialize variables to track the earliest time reaching mass threshold
    earliest_activity = 0

    # Check each line
    for line in lines:
        # Get the x and y data
        xdata = line.get_xdata()
        ydata = line.get_ydata()  # This is SFR/yr

        # Calculate time intervals (assuming uniform spacing)
        dt = np.abs(xdata[1] - xdata[0])

        # Calculate cumulative mass formed
        # Integrate SFR from observation time (x=0) backwards
        # Remember: x-axis is negative lookback time, so we need to flip the integration
        cumulative_mass = np.cumsum(ydata[::-1] * dt)[::-1]

        # Normalize by total mass formed
        total_mass = cumulative_mass[0]  # Mass at observation time
        normalized_mass = cumulative_mass / total_mass

        # Find indices where normalized mass exceeds threshold
        active_indices = np.where(normalized_mass >= mass_threshold)[0]

        if len(active_indices) > 0:
            # Find the earliest time reaching threshold for this line
            earliest_this_line = xdata[active_indices[-1]]  # Using -1 since time goes backwards

            earliest_activity = max(earliest_activity, earliest_this_line)

    if earliest_activity == 0:
        raise ValueError("No galaxies found reaching the mass threshold")

    # Add buffer to the range
    buffer = abs(earliest_activity) * buffer_fraction
    new_xlimit = earliest_activity + buffer

    return new_xlimit


if __name__ == "__main__":
    # Test the create_padded_colormap function
    # custom_colors = {0: 'red', 100: 'blue'}
    # cmap = create_padded_colormap(custom_colors_dict=custom_colors, base_cmap_name='nipy_spectral_r', total_length=256)
    fig, ax = plt.subplots()

    n = 100
    len_sfh1 = 33
    len_sfh2 = 10

    mock_sfh = np.pad(np.array([10] * len_sfh1), (0, n - len_sfh1), mode="constant")
    ax.plot(np.arange(0, n), mock_sfh)

    mock_sfh = np.pad(np.array([10] * len_sfh2), (0, n - len_sfh2), mode="constant")
    ax.plot(np.arange(0, n), mock_sfh)

    x = optimize_sfh_xlimit(ax)

    print(x)


def create_fitsmap(
    survey,
    field_info,
    overwrite=False,
    catalogue_path=None,
    out_dir="/nvme/scratch/work/tharvey/fitsmap/",
    filter_val=None,
    filter_field=None,
    plot_path_column=None,
    other_image_paths=[],
    wcs_band="F444W",
    kron_band="F444W",
    rgb_bands={"red": ["F444W"], "green": ["F277W"], "blue": ["F150W"]},
    fitsmap_columns={
        "id": "NUMBER",
        "ra": "ALPHA_J2000",
        "dec": "DELTA_J2000",
        "a": "A_IMAGE_kron_band",
        "b": "B_IMAGE_kron_band",
        "theta": "THETA_IMAGE_kron_band",
        "kron_radius": "KRON_RADIUS_kron_band",
        "x": "X_IMAGE",
        "y": "Y_IMAGE",
    },
    use_wcs=True,
    extra_columns={
        "MAG_AUTO_F444W": "F444W Mag",
        "MAG_AUTO_F277W": "F277W Mag",
        "zbest_fsps_larson_zfree": "photo-z",
        "chi2_best_fsps_larson_zfree": "chi2",
    },
):  # 'zbest_fsps_larson_zfree':'Photo-z',
    """
    Create a fitsmap for a given survey and field_info object.

    Parameters:
    -----------

    survey : str
        The name of the survey (e.g. 'CANDELS', 'GOODS-S', etc.)
    field_info : FieldInfo
        The FieldInfo object containing the band information for the survey
    overwrite : bool, optional
        Whether to overwrite existing fitsmap files (default: False)
    catalogue_path : str, optional
        Path to the catalogue file to include in the fitsmap (default: None)
    out_dir : str, optional
        The output directory for the fitsmap files (default: '/nvme/scratch/work/tharvey/fitsmap/')
    extra_columns_to_show : list, optional
        List of additional columns to include in the fitsmap table (default: [])
    filter_val : str or None, optional
        Value to filter the fitsmap table by (default: None)
    filter_field : str or None, optional
        Field to filter the fitsmap table by (default: None)
    plot_path_column : str or dict or list(len(catalogue)) or None, optional
        Column name containing paths to SED plots (default: None) or a string to use as a folder path with {id}.png format
        Or a dict matching ID to path
        Or a list of paths matching the length of the catalogue
    other_image_paths : list, optional
        List of additional image paths to include in the fitsmap (default: [])
    wcs_band : str, optional
        Band to use for WCS information (default: 'F444W')
    rgb_bands : dict, optional
        Dictionary of RGB bands to use for creating RGB images (default: {'red': ['F444W'], 'green': ['F277W'], 'blue': ['F150W']})
    default_columns_to_show : dict, optional
        Dictionary of default columns to include in the fitsmap table. Maps needed columns (id, ra, dec) to catalogue names (default: {'UNIQUE_ID':'id', 'ALPHA_J2000':'ra', 'DELTA_J2000':'dec','MAG_AUTO_F444W':'f444W Mag', 'MAG_AUTO_F277W':'f277W Mag', 'zbest_fsps_larson_zfree':'photo-z', 'chi2_best_fsps_larson_zfree':'chi2'})

    """
    print(f"Creating fitsmap for {survey}...")

    sys.path.append("/nvme/scratch/software/trilogy/")
    from fitsmap import convert
    from trilogy3 import Trilogy

    folder_name = f"{survey}/"
    out_path = out_dir + folder_name
    if overwrite:
        try:
            shutil.rmtree(out_path)
        except FileNotFoundError:
            pass

    os.makedirs(out_path, exist_ok=True)

    bands = field_info.band_names
    paths = [field_info.im_paths[band] for band in bands]

    # Catalogue path
    if catalogue_path is not None:
        nhdu = len(fits.open(catalogue_path))
        table = Table.read(catalogue_path)

        if nhdu > 1:
            # check same number of rows and hstack
            for i in range(2, nhdu):
                table_i = Table.read(catalogue_path, hdu=i)
                if len(table) == len(table_i):
                    table = hstack([table, table_i])

        # Check if table has multiple HDUs

        print(table.colnames)

        if filter_val != None and filter_field != None and filter_field != "None":
            if type(filter_val) in [list, np.ndarray]:
                mask = [True if i in filter_val else False for i in table[filter_field]]
                filtered_table = table[mask]
            else:
                filtered_table = table[table[filter_field] == filter_val]
        else:
            filtered_table = table

        output_table = Table()
        filtered_output_table = Table()
        for column in fitsmap_columns:
            cat_colname = fitsmap_columns[column]
            if column in [
                "id",
                "ra",
                "dec",
                "a",
                "b",
                "theta",
                "kron_radius",
                "x",
                "y",
            ]:
                if column == "a" or column == "b":
                    output_table[column] = table[cat_colname.replace("kron_band", kron_band)]
                    filtered_output_table[column] = filtered_table[
                        cat_colname.replace("kron_band", kron_band)
                    ]
                    if (
                        f'{fitsmap_columns["kron_radius"].replace("kron_band", kron_band)}'
                        in table.colnames
                    ):
                        print(f"Converting dimensionless a/b to pixels for {kron_band}...")
                        output_table[column] *= table[
                            fitsmap_columns["kron_radius"].replace("kron_band", kron_band)
                        ]
                        filtered_output_table[column] *= filtered_table[
                            fitsmap_columns["kron_radius"].replace("kron_band", kron_band)
                        ]

                    continue
                elif column == "theta":
                    output_table[column] = table[cat_colname.replace("kron_band", kron_band)]
                    filtered_output_table[column] = filtered_table[
                        cat_colname.replace("kron_band", kron_band)
                    ]
                    continue
                elif column == "kron_radius":
                    continue

                if cat_colname not in table.colnames:
                    if f"{cat_colname}_1" in table.colnames:
                        table[cat_colname] = table[f"{cat_colname}_1"]
                        filtered_table[cat_colname] = filtered_table[f"{cat_colname}_1"]
                    else:
                        print(f"{cat_colname} not found in table.")
                        continue
                output_table[column] = table[cat_colname]
                filtered_output_table[column] = filtered_table[cat_colname]
            else:
                output_table[column] = [f"{i:.2f}" for i in table[cat_colname]]
                filtered_output_table[column] = [f"{i:.2f}" for i in filtered_table[cat_colname]]

        for column in extra_columns:
            output_table[extra_columns[column]] = [f"{i:.2f}" for i in table[column]]
            filtered_output_table[extra_columns[column]] = [
                f"{i:.2f}" for i in filtered_table[column]
            ]

        catalog_path = out_path + "catalog.cat"
        # filter_field will be a js variable name, so we need to remove any special characters

        if filter_val != None and filter_field != None and filter_field != "None":
            filter_field = (
                filter_field.replace(" ", "_")
                .replace("-", "_")
                .replace(".", "_")
                .replace("(", "_")
                .replace(")", "_")
                .replace("+", "_")
                .replace("=", "_")
                .replace("/", "_")
            )
            filtered_catalog_path = out_path + f"catalog_{filter_field}.cat"

            if "x" in output_table.colnames and "y" in output_table.colnames:
                if use_wcs and "ra" in output_table.colnames and "dec" in output_table.colnames:
                    output_table.remove_column("x")
                    output_table.remove_column("y")
                    filtered_output_table.remove_column("x")
                    filtered_output_table.remove_column("y")

        # add SED plot columns
        img_array = []
        if plot_path_column != None:
            if plot_path_column in table.colnames:
                sed_plot_paths = table[plot_path_column]
                filtered_sed_plot_path = filtered_table[plot_path_column]
            elif type(plot_path_column) == str:
                sed_plot_paths = [
                    f"{plot_path_column}{i}.png" for i in filtered_table[fitsmap_columns["id"]]
                ]

            for row, path in zip(filtered_output_table, sed_plot_paths):
                try:
                    os.symlink(os.path.dirname(path), out_path + "SED_plots")
                except FileExistsError:
                    pass

                full_path = f"{out_path}/SED_plots/{os.path.basename(path)}"
                path = f"SED_plots/{os.path.basename(path)}"
                if pathlib.Path(f"{out_path}/{path}").is_file():
                    img_array.append(
                        f'<a href="{path}"><img src="{path}" width="723" height="550"></a>'
                    )
                else:
                    img_array.append("Not found.")

            # assume all plots in same folder

            filtered_output_table["SED_plot"] = img_array

        output_table.write(catalog_path, overwrite=True, format="ascii.csv")
        paths.append(catalog_path)

        if filter_val != None and filter_field != None and filter_field != "None":
            filtered_output_table.write(filtered_catalog_path, overwrite=True, format="ascii.csv")
            paths.append(filtered_catalog_path)

    # Add detection image
    if field_info.detection_band != None:
        detection_phot = field_info.detection_band
        paths.append(detection_phot.im_path)

    # Make RGB images

    # Write trilogy.in
    if not os.path.exists(f"{out_path}/{survey}_RGB.png"):
        print("Creating Trilogy RGB...")
        red_bands = rgb_bands["red"]
        green_bands = rgb_bands["green"]
        blue_bands = rgb_bands["blue"]

        with open(out_path + "trilogy.in", "w") as f:
            f.write("B\n")
            for band in blue_bands:
                if band in bands:
                    f.write(f"{field_info.im_paths[band]}[{field_info.im_exts[band]}]\n")
            f.write("\nG\n")
            for band in green_bands:
                if band in bands:
                    f.write(f"{field_info.im_paths[band]}[{field_info.im_exts[band]}]\n")
            f.write("\nR\n")
            for band in red_bands:
                if band in bands:
                    f.write(f"{field_info.im_paths[band]}[{field_info.im_exts[band]}]\n")
            f.write(f"""\nindir  /
    outname  {survey}_RGB
    outdir  {out_path}
    samplesize 2000
    stampsize  2000
    showstamps  0
    satpercent  0.001
    noiselum    0.10
    colorsatfac  1
    correctbias  1
    deletetests  1
    testfirst   0
    sampledx  3000
    sampledy  3000""")
        # Run trilogy
        # subprocess.run(['python', '/nvme/scratch/software/trilogy/trilogy3.py', out_path + 'trilogy.in'])
        Trilogy(out_path + "trilogy.in", images=None).run()
    paths.append(out_path + f"{survey}_RGB.png")

    for band in tqdm(bands, desc="Creating seg maps..."):
        seg_path = field_info.seg_paths[band]
        if seg_path == None:
            continue
        seg_name = os.path.basename(seg_path)
        # Make seg map image
        if not os.path.exists(f'{out_path}/{seg_name.replace(".fits", ".png")}'):
            img = fits.open(seg_path, ignore_missing_simple=True)[0].data
            num_of_colors = len(np.unique(img))
            colors = np.random.choice(list(mcolors.XKCD_COLORS.keys()), num_of_colors)
            colors[0] = "black"
            cmap = mcolors.ListedColormap(colors)

            norm = plt.Normalize(vmin=img.min(), vmax=img.max())
            # map the normalized data to colors
            # image is now RGBA (512x512x4)
            image = cmap(norm(img))

            # save the image
            plt.imsave(f'{out_path}/{seg_name.replace(".fits", ".png")}', image)
        paths.append(f'{out_path}/{seg_name.replace(".fits", ".png")}')

        # Need a cmap which changes color for every 1 increase -

    for other_image in other_image_paths:
        paths.append(other_image)

    display = dict(
        stretch="log",
        max_percent=99.8,
        min_percent=1,
        min_cut=0,
        max_cut=10,
        log_a=10,
    )
    norm = {pathlib.Path(path).name: display for path in paths if path.endswith(".fits")}
    wcs_band_ext = field_info.im_exts[wcs_band]

    cat_wcs_header = fits.open(field_info.im_paths[wcs_band])[wcs_band_ext].header

    convert.files_to_map(
        paths,
        out_dir=out_path,
        title=f'{survey} {filter_field if filter_field != None else ""}',
        cat_wcs_fits_file=cat_wcs_header,
        procs_per_task=3,
        task_procs=2,
        norm_kwargs=norm,
    )


def display_fitsmap(field, out_dir="/nvme/scratch/work/tharvey/fitsmap/", x=800, y=400):
    from IPython.display import IFrame

    # cd to fitsmap directory
    os.chdir(out_dir + field + "/")
    os.system("pkill fitsmap")
    # Run fitsmap serve in another thread and suppress output

    os.system("fitsmap serve &")

    return display(IFrame("http://localhost:8000//", x, y))  # noqa: F821


def gradient_path_effect(
    levels=30, max_width=4, color="black", min_alpha=0.1, max_alpha=0.5, scale="log", line_lw=0
):
    path_effects = []

    # generate alpha range using scale to distribute the levels
    if scale == "log":
        alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha), levels)
    elif scale == "linear":
        alphas = np.linspace(min_alpha, max_alpha, levels)
    elif scale == "sqrt":
        alphas = np.linspace(min_alpha**2, max_alpha**2, levels) ** 0.5

    alphas = alphas[::-1]

    for i in range(levels):
        alpha = alphas[i]
        width = line_lw + max_width * i / levels
        path_effects.append(
            SimpleLineShadow(offset=(0, 0), shadow_color=color, alpha=alpha, linewidth=width)
        )

    return path_effects


def plot_with_shadow(
    ax,
    x,
    y,
    lw=2,
    path_effect_kws={
        "levels": 10,
        "max_width": 15,
        "color": "black",
        "max_alpha": 0.1,
        "min_alpha": 0.01,
        "scale": "log",
    },
    **kwargs,
):
    default_kwargs = {"zorder": 3, "lw": lw}
    default_kwargs.update(kwargs)
    if "shadow_zorder" in default_kwargs:
        default_kwargs["zorder"] = default_kwargs["shadow_zorder"]
        default_kwargs.pop("shadow_zorder")

    label = ""
    if "label" in default_kwargs:
        label = default_kwargs["label"]
        default_kwargs.pop("label")

    default_kwargs["alpha"] = 0
    path_effects = gradient_path_effect(**path_effect_kws, line_lw=lw)
    lines = ax.plot(x, y, **default_kwargs)
    default_kwargs["alpha"] = kwargs.get("alpha", 1)
    for line in lines:
        line.set_path_effects(path_effects)
    lines_normal = [ax.plot(l.get_xdata(), l.get_ydata())[0] for l in lines]
    for l, ln in zip(lines, lines_normal):
        ln.update_from(l)
        ln.set_path_effects([Normal()])

    default_kwargs["zorder"] = kwargs.get("zorder", 4)
    default_kwargs["label"] = label
    lines = ax.plot(x, y, **default_kwargs)


def renorm_psf(psfmodel, filt, fov=4.04, pixscl=0.03):
    """
    Renormalize the PSF model to account for missing flux.

    Parameters:
    psfmodel (array): The PSF model.
    filt (str): The filter for which the PSF model is being generated.
    fov (float): The field of view (FOV) of the PSF in arcseconds.
    pixscl (float): The pixel scale of the PSF model in arcseconds per pixel.

    Returns:
    array: The renormalized PSF model.

    """

    filt = filt.upper()

    # Encircled energy for WFC3 IR within 2" radius, ACS Optical, and UVIS from HST docs
    encircled = {}
    encircled["F225W"] = 0.993
    encircled["F275W"] = 0.984
    encircled["F336W"] = 0.9905
    encircled["F435W"] = 0.979
    encircled["F606W"] = 0.975
    encircled["F775W"] = 0.972
    encircled["F814W"] = 0.972
    encircled["F850LP"] = 0.970
    encircled["F098M"] = 0.974
    encircled["F105W"] = 0.973
    encircled["F125W"] = 0.969
    encircled["F140W"] = 0.967
    encircled["F160W"] = 0.966
    encircled["F090W"] = 0.9837
    encircled["F115W"] = 0.9822
    encircled["F150W"] = 0.9804
    encircled["F200W"] = 0.9767
    encircled["F277W"] = 0.9691
    encircled["F356W"] = 0.9618
    encircled["F410M"] = 0.9568
    encircled["F444W"] = 0.9546

    # These taken from the Encircled_Energy ETC numbers
    encircled["F140M"] = 0.984
    encircled["F162M"] = 0.982
    encircled["F182M"] = 0.980
    encircled["F210M"] = 0.978
    encircled["F250M"] = 0.971
    encircled["F300M"] = 0.967
    encircled["F335M"] = 0.963
    encircled["F360M"] = 0.958
    encircled["F430M"] = 0.956

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(psfmodel)
    Y, X = np.ogrid[:h, :w]
    r = fov / 2.0 / pixscl
    center = [w / 2.0, h / 2.0]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    psfmodel /= np.sum(psfmodel[dist_from_center < r])
    if filt in encircled:
        psfmodel *= encircled[filt]  # to get the missing flux accounted for
    else:
        print(f"WARNING -- I DO NOT HAVE ENCIRCLED ENERGY FOR {filt}! SKIPPING NORM.")

    return psfmodel


# Via Stack Overflow
# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# Via gist https://gist.github.com/vikjam/755930297430091d8d8df70ac89ea9e2


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def _cf_periodicity_dilution_correction_standalone(cf_shape: Tuple[int, int]) -> np.ndarray:
    """
    Calculates the correction factor for DFT-based Correlation Function estimation
    to account for the assumption of periodicity.
    Ported from an internal GalSim function.

    Args:
        cf_shape: Tuple (Ny, Nx) representing the shape of the correlation function array.

    Returns:
        A 2D NumPy array with the correction factors.
    """
    ny, nx = cf_shape

    # Create coordinate arrays for frequency domain
    # dx_coords corresponds to frequencies in the x-direction (columns)
    # dy_coords corresponds to frequencies in the y-direction (rows)
    dx_coords = np.fft.fftfreq(nx) * float(nx)
    dy_coords = np.fft.fftfreq(ny) * float(ny)

    # Create 2D grids of these coordinates
    # deltax will have shape (Ny, Nx) and vary along columns (axis 1)
    # deltay will have shape (Ny, Nx) and vary along rows (axis 0)
    deltax, deltay = np.meshgrid(dx_coords, dy_coords)

    denominator = (nx - np.abs(deltax)) * (ny - np.abs(deltay))

    # Avoid division by zero if denominator entries are zero,
    # though this is unlikely with standard fftfreq outputs for valid shapes.
    # np.finfo(float).eps could be added for numerical stability if needed.
    valid_denominator = np.where(denominator == 0, 1.0, denominator)  # Avoid true div by zero

    correction = (float(nx * ny)) / valid_denominator
    if np.any(denominator == 0):  # Put back zeros where they were to avoid infs if original was 0
        correction[denominator == 0] = 0  # Or handle as error/warning

    return correction


def _generate_noise_from_rootps_standalone(
    rng: np.random.Generator, shape: Tuple[int, int], rootps: np.ndarray
) -> np.ndarray:
    """
    Generates a real-space noise field from its sqrt(PowerSpectrum).
    Ported and adapted from an internal GalSim function.

    Args:
        rng: NumPy random number generator.
        shape: Tuple (Ny, Nx) of the output real-space noise field.
        rootps: The half-complex 2D array (Ny, Nx//2 + 1) representing
                the square root of the Power Spectrum from rfft2.

    Returns:
        A 2D NumPy array representing the generated correlated noise field.
    """
    ny, nx = shape

    # GalSim's GaussianDeviate implies specific scaling for random numbers.
    # E[|gvec_k|^2] = Ny * Nx for each complex component k if parts are N(0, 0.5*Ny*Nx).
    sigma_val_for_gvec_parts = np.sqrt(0.5 * ny * nx)

    gvec_real = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec_imag = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec = gvec_real + 1j * gvec_imag

    # Impose Hermitian symmetry properties and scaling for DC/Nyquist terms.
    # This ensures the iFFT results in a real field with correct variance distribution.
    rt2 = np.sqrt(2.0)

    # DC component (ky=0, kx=0)
    gvec[0, 0] = rt2 * gvec[0, 0].real

    # Nyquist frequency for y-axis (ky=Ny/2) at kx=0 (if Ny is even)
    if ny % 2 == 0:
        gvec[ny // 2, 0] = rt2 * gvec[ny // 2, 0].real

    # Nyquist frequency for x-axis (kx=Nx/2) at ky=0 (if Nx is even)
    if nx % 2 == 0:
        gvec[0, nx // 2] = rt2 * gvec[0, nx // 2].real

    # Corner Nyquist component (ky=Ny/2, kx=Nx/2) (if both Ny and Nx are even)
    if ny % 2 == 0 and nx % 2 == 0:
        gvec[ny // 2, nx // 2] = rt2 * gvec[ny // 2, nx // 2].real

    # Conjugate symmetry for the kx=0 column (y-axis frequencies)
    # gvec[Ny-row, 0] = conj(gvec[row, 0]) for row in 1 .. (Ny/2 - 1) or ((Ny-1)/2)
    # This handles the negative y-frequencies for kx=0.
    # The slice `ny-1 : ny//2 : -1` covers rows from Ny-1 down to (Ny//2 + 1).
    # The slice `1 : (ny+1)//2` covers rows from 1 up to Ny//2.
    if ny > 1:  # This symmetry operation is relevant if Ny > 1 (or Ny > 2 for some ranges)
        gvec[ny - 1 : ny // 2 : -1, 0] = np.conj(gvec[1 : (ny + 1) // 2, 0])

    # Conjugate symmetry for the kx=Nx/2 column (if Nx is even)
    # This handles negative y-frequencies for the Nyquist x-frequency.
    if nx % 2 == 0:
        kx_nyq_idx = nx // 2
        if ny > 1:
            gvec[ny - 1 : ny // 2 : -1, kx_nyq_idx] = np.conj(gvec[1 : (ny + 1) // 2, kx_nyq_idx])

    # Element-wise multiplication in Fourier space
    noise_field_k_space = gvec * rootps

    # Inverse FFT to get the real-space noise field
    # The `s` parameter ensures the output shape matches the desired `shape`.
    noise_real_space = np.fft.irfft2(noise_field_k_space, s=shape)

    return noise_real_space


def model_and_apply_correlated_noise(
    source_image_arr: np.ndarray,
    target_image_arr: np.ndarray,
    subtract_mean: bool = False,
    correct_periodicity: bool = True,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Models correlated noise from a source image and applies it to a target image.

    The noise model, represented by its Correlation Function (CF), is derived
    from the `source_image_arr`. This CF is then used to generate a noise field
    with similar statistical properties, which is subsequently added to the
    `target_image_arr`.

    This function assumes that the pixel scales of the source and target images
    are comparable, or that the CF is intended to be interpreted in pixel units.
    If the images possess different physical pixel scales, appropriate
    pre-processing (e.g., resampling) might be necessary before using this function
    for physically accurate noise transfer.

    Args:
        source_image_arr: A 2D NumPy array. This image is used to model the
                          correlated noise characteristics.
        target_image_arr: A 2D NumPy array. Correlated noise will be added to this image.
        subtract_mean: If True, the mean of the `source_image_arr` is effectively
                       removed before Power Spectrum estimation by setting the
                       DC component (PS[0,0]) of the Power Spectrum to zero.
        correct_periodicity: If True, a correction factor is applied to the
                             estimated Correlation Function. This factor aims to
                             compensate for the inherent assumption of periodicity
                             in Discrete Fourier Transforms.
        rng_seed: An optional integer seed for the random number generator.
                  Providing a seed ensures reproducible noise generation.

    Returns:
        A new 2D NumPy array, which is the `target_image_arr` with the
        synthesized correlated noise added to it.
    """
    if source_image_arr.ndim != 2 or target_image_arr.ndim != 2:
        raise ValueError("Input images must be 2D numpy arrays.")

    rng = np.random.default_rng(rng_seed)
    source_shape = source_image_arr.shape

    # --- Part 1: Model Noise (Correlation Function) from source_image_arr ---
    # Calculate Fourier Transform of the source image
    ft_array = np.fft.rfft2(source_image_arr)

    # Power Spectrum (PS) of the source image
    ps_array_source = np.abs(ft_array) ** 2

    # Normalize PS: Ensures iFFT(PS) results in a CF where CF[0,0] is the variance.
    ps_array_source /= np.prod(source_shape)

    if subtract_mean:
        ps_array_source[0, 0] = 0.0  # Zero out DC component of PS

    # Estimate Correlation Function (CF) from the normalized PS
    # cf_array_prelim is unrolled (origin at [0,0] for DFT purposes)
    cf_array_prelim = np.fft.irfft2(ps_array_source, s=source_shape)

    if correct_periodicity:
        correction = _cf_periodicity_dilution_correction_standalone(source_shape)
        cf_array_prelim *= correction

    # cf_array_prelim now holds the noise model (unrolled CF).
    # cf_array_prelim[0,0] approximates the variance of the source noise.

    # --- Part 2: Prepare CF for target_image_arr dimensions ---
    target_shape = target_image_arr.shape

    # "Draw" cf_array_prelim onto an array of target_shape.
    # This involves rolling the source CF to center its peak, then
    # cropping or padding it to match target dimensions (centers aligned),
    # and finally unrolling it for FFT.

    # Roll source CF so that the peak (variance, originally at [0,0]) is at the center
    cf_source_rolled = np.roll(
        cf_array_prelim, shift=(source_shape[0] // 2, source_shape[1] // 2), axis=(0, 1)
    )

    # Create a rolled CF for the target shape by copying the central part of cf_source_rolled
    cf_target_rolled = np.zeros(target_shape, dtype=float)

    src_cy, src_cx = source_shape[0] // 2, source_shape[1] // 2
    trg_cy, trg_cx = target_shape[0] // 2, target_shape[1] // 2

    # Define copy regions to place the center of source_cf onto the center of target_cf
    # Source region to copy from:
    y_start_src = max(0, src_cy - trg_cy)
    y_end_src = min(source_shape[0], src_cy + (target_shape[0] - trg_cy))
    x_start_src = max(0, src_cx - trg_cx)
    x_end_src = min(source_shape[1], src_cx + (target_shape[1] - trg_cx))

    # Target region to copy to:
    y_start_trg = max(0, trg_cy - src_cy)
    y_end_trg = min(target_shape[0], trg_cy + (source_shape[0] - src_cy))
    x_start_trg = max(0, trg_cx - src_cx)
    x_end_trg = min(target_shape[1], trg_cx + (source_shape[1] - src_cx))

    # Ensure the lengths of the regions to be copied match
    dy = min(y_end_src - y_start_src, y_end_trg - y_start_trg)
    dx = min(x_end_src - x_start_src, x_end_trg - x_start_trg)

    if dy > 0 and dx > 0:
        cf_target_rolled[y_start_trg : y_start_trg + dy, x_start_trg : x_start_trg + dx] = (
            cf_source_rolled[y_start_src : y_start_src + dy, x_start_src : x_start_src + dx]
        )

    # Unroll cf_target_rolled to place the origin at [0,0] for DFT
    # This array represents the CF sampled on the target grid.
    cf_on_target_grid_unrolled = np.roll(
        cf_target_rolled, shift=(-(target_shape[0] // 2), -(target_shape[1] // 2)), axis=(0, 1)
    )

    # Calculate Power Spectrum for this target-shaped CF.
    # This ps_target_fft is unnormalized (direct output of rfft2).
    ps_target_fft = np.fft.rfft2(cf_on_target_grid_unrolled)
    rootps_target = np.sqrt(np.abs(ps_target_fft))  # Magnitudes for sqrt

    # --- Part 3: Generate noise and apply it to the target image ---
    generated_noise = _generate_noise_from_rootps_standalone(rng, target_shape, rootps_target)

    # Add the generated noise to the original target image
    output_image_arr = target_image_arr + generated_noise
    return output_image_arr


def calculate_radial_correlation(
    image_arr: np.ndarray, subtract_mean: bool = True, max_pixel_distance: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the radially averaged auto-correlation of pixel values in an image
    as a function of pixel distance.

    The auto-correlation is normalized by the variance of the (mean-subtracted)
    image, so the correlation at zero distance is 1.0 (assuming non-zero variance).

    Args:
        image_arr: 2D NumPy array representing the image.
        subtract_mean: If True (default), subtracts the overall mean of the image
                       before calculating the correlation. This is generally
                       recommended for interpreting the result as a standard
                       correlation coefficient.
        max_pixel_distance: The maximum integer pixel distance (radius) for which
                            to calculate the average correlation. If None, it
                            defaults to a value based on the image dimensions
                            (typically half the smallest dimension, or the full
                            extent for 1D-like images).

    Returns:
        A tuple (distances, correlations):
        - distances: 1D NumPy array of integer pixel distances [0, 1, 2, ...].
        - correlations: 1D NumPy array of the corresponding radially averaged
                        correlation coefficients. The correlation at distance 0
                        is 1.0 if the image has non-zero variance. Values for
                        distances where no pixels are found (e.g., beyond image
                        bounds for a given max_pixel_distance) will be NaN.
    """
    if image_arr.ndim != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    ny, nx = image_arr.shape

    if subtract_mean:
        # Work with a float version for precision if subtracting mean
        processed_image = image_arr.astype(float) - np.mean(image_arr)
    else:
        # If not subtracting mean, ensure it's float for FFT, operate on copy
        processed_image = image_arr.astype(float, copy=True)

    # Compute 2D Auto-Correlation Function using FFTs
    # FT(I) = F (Fourier Transform of the processed image)
    # Power Spectrum PS = F * conj(F)
    # The Inverse FFT of PS gives the unnormalized auto-covariance function.
    # The element (0,0) of this IFFT(PS) is sum_pixels(processed_image**2).

    ft_array = np.fft.rfft2(processed_image)
    power_spectrum = np.abs(ft_array) ** 2  # Element-wise squared magnitude

    # Inverse FFT of the power spectrum gives the unnormalized auto-covariance
    auto_covariance_unrolled = np.fft.irfft2(power_spectrum, s=processed_image.shape)

    # The value at lag (0,0) of the unnormalized auto-covariance
    # is sum(processed_image**2).
    variance_numerator_term = auto_covariance_unrolled[0, 0]

    # Handle images with effectively zero variance (e.g., flat images)
    # Small epsilon relative to image size to define "effectively zero"
    epsilon = 1e-12 * np.prod(processed_image.shape)
    if variance_numerator_term < epsilon:
        center_y_temp, center_x_temp = ny // 2, nx // 2
        if max_pixel_distance is None:
            # For 1D-like arrays (e.g., 1xN or Nx1), use the available extent
            if ny == 1 or nx == 1:
                max_r = max(center_y_temp, center_x_temp)
            else:  # For 2D arrays, default to half the minimum dimension
                max_r = min(center_y_temp, center_x_temp)
        else:
            max_r = max_pixel_distance

        max_r = max(0, int(np.floor(max_r)))  # Ensure non-negative integer

        distances = np.arange(max_r + 1)
        correlations = np.full_like(distances, np.nan, dtype=float)
        if len(correlations) > 0:
            # Correlation of a variable with itself is 1, even if variance is 0.
            correlations[0] = 1.0
        return distances, correlations

    # Normalize by the value at (0,0) lag to get the auto-correlation coefficient function.
    # rho(lag) = Cov(lag) / Var
    # Cov(lag) is effectively (auto_covariance_unrolled[lag] / N_pixels)
    # Var is effectively (auto_covariance_unrolled[0,0] / N_pixels)
    # So, rho(lag) = auto_covariance_unrolled[lag] / auto_covariance_unrolled[0,0]
    correlation_2d_unrolled_normalized = auto_covariance_unrolled / variance_numerator_term

    # Roll the 2D auto-correlation function so that the (0,0) lag is at the center of the array
    correlation_2d_rolled = np.roll(
        correlation_2d_unrolled_normalized,
        shift=(ny // 2, nx // 2),  # Shift to bring (0,0) lag to center
        axis=(0, 1),
    )
    # After rolling, correlation_2d_rolled[ny//2, nx//2] corresponds to the (0,0) lag, value is 1.0.

    # Prepare for radial averaging
    center_y, center_x = ny // 2, nx // 2

    # Create a grid of distances from the center of the rolled 2D correlation function
    # y_indices will be a column vector, x_indices a row vector due to np.ogrid
    y_indices, x_indices = np.ogrid[-center_y : ny - center_y, -center_x : nx - center_x]
    distance_from_center_grid = np.sqrt(x_indices**2 + y_indices**2)

    # Determine the maximum radius for computation
    if max_pixel_distance is None:
        if ny == 1 or nx == 1:  # For 1D-like arrays
            max_r_computed = max(center_y, center_x)
        else:  # For 2D arrays
            max_r_computed = min(center_y, center_x)
    else:
        max_r_computed = int(np.floor(max_pixel_distance))

    max_r_computed = max(0, max_r_computed)  # Ensure it's not negative

    distances_to_report = np.arange(max_r_computed + 1)
    avg_correlations_final = np.full(max_r_computed + 1, np.nan, dtype=float)

    # Perform radial binning
    for d_bin in distances_to_report:  # Iterate through integer distances 0, 1, 2, ...
        # Define a mask for pixels approximately at distance d_bin
        if d_bin == 0:
            # For distance 0, select pixels very close to the center
            mask = distance_from_center_grid < 0.5
        else:
            # For other distances, select pixels in an annulus
            mask = (distance_from_center_grid >= (float(d_bin) - 0.5)) & (
                distance_from_center_grid < (float(d_bin) + 0.5)
            )

        if np.any(mask):
            avg_correlations_final[d_bin] = np.mean(correlation_2d_rolled[mask])
        # If no pixels fall into this radial bin (e.g., d_bin is too large),
        # the value in avg_correlations_final remains NaN.

    # Ensure correlation at distance 0 is exactly 1.0 if variance was non-zero.
    # The binning for d_bin=0 should capture correlation_2d_rolled[center_y, center_x].
    if variance_numerator_term >= epsilon and len(avg_correlations_final) > 0:
        avg_correlations_final[0] = 1.0

    return distances_to_report, avg_correlations_final
