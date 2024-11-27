import curses
import os
import sys
import threading
import time

import matplotlib as mpl
import numpy as np
from astropy import units as u
from photutils.aperture import (
    EllipticalAperture,
    aperture_photometry,
    CircularAperture,
)
from astropy.wcs import WCS
from astropy.io import fits
import glob
from matplotlib.patches import Arrow, FancyArrow
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.interpolate import interp1d


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
):
    """Scale fluxes to total flux using PSF
    Scale mag_aper to mag_auto (ideally measured in LW stack), and then derive PSF correction by placing a elliptical aperture around the PSF.
    """
    a = kron_radius * a
    b = kron_radius * b
    area_of_aper = np.pi * (aper_diam.to(u.arcsec) / (2 * pix_scale)) ** 2
    area_of_ellipse = np.pi * a * b
    scale_factor = area_of_aper / area_of_ellipse
    if flux_type == "mag":
        flux_auto = 10 ** ((zero_point - mag_auto) / 2.5)
        flux_aper = 10 ** ((zero_point - mag_aper) / 2.5)
    else:
        flux_auto = mag_auto
        flux_aper = mag_aper
    if (scale_factor > 1) and flux_auto > flux_aper:
        factor = flux_auto / flux_aper
        clip = False
    else:
        factor = 1
        clip = True

    if clip:
        # Make Elliptical Aperture be the circle
        a = aper_diam.to(u.arcsec) / (2 * pix_scale)
        b = aper_diam.to(u.arcsec) / (2 * pix_scale)
        theta = 0

    print(f"Corrected aperture flux by {factor} for kron ellipse flux.")
    # Scale for PSF
    assert type(psf) is np.ndarray, "PSF must be a numpy array"
    assert (
        np.sum(psf) < 1
    ), "PSF should not be normalised, some flux is outside the footprint."
    # center = (psf.shape[0] - 1) / 2
    from photutils.centroids import centroid_com

    center = centroid_com(psf)
    #
    # from photutils import CircularAperture, EllipticalAperture, aperture_photometry
    # circular_aperture = CircularAperture(center, center, r=aper_diam/(2*pixel_scale))
    # circular_aperture_phot = aperture_photometry(psf, circular_aperture)

    # get path of this file
    file_path = os.path.abspath(__file__)
    psf_path = (
        os.path.dirname(os.path.dirname(os.path.dirname(file_path))) + "/psfs"
    )
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
        elliptical_aperture_phot = aperture_photometry(
            psf, elliptical_aperture
        )
        encircled_energy = elliptical_aperture_phot["aperture_sum"][0]

    factor_total = factor / encircled_energy

    return factor_total, factor


def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [
        {"code": code, "templates": templates, "lowz_zmax": lowz_zmax}
        for code, templates, lowz_zmaxs in zip(
            SED_code_arr, templates_arr, lowz_zmax_arr
        )
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

                info_line.append(
                    f"Total RAM usage: {psutil.virtual_memory().percent:.2f}%"
                )
                pid = os.getpid()
                python_process = psutil.Process(pid)
                memoryUse = (
                    python_process.memory_info()[0] / 2.0**30
                )  # memory use in GB...I think
                info_line.append(f"Script RAM usage: {memoryUse:.2f} GB")
                # script CPU usage
                info_line.append(
                    f"Script CPU usage: {python_process.cpu_percent(interval = 0.1)}%"
                )
                # Which core is being used
                info_line.append(f"Core = {python_process.cpu_num()}")
                core_num = python_process.cpu_num()

                index = np.argwhere(
                    [
                        i[0] == f"Core {core_num}"
                        for i in psutil.sensors_temperatures()["coretemp"]
                    ]
                )
                if len(index) > 0:
                    # flatten index
                    index = index.flatten()

                    temp = psutil.sensors_temperatures()["coretemp"][index[0]][
                        1
                    ]
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
            progress_bar = (
                "["
                + "#" * progress_width
                + " " * (width - 2 - progress_width)
                + "]"
            )
            self.screen.addstr(
                height - 1, 0, progress_bar[: width - 1], curses.color_pair(3)
            )

        self.screen.refresh()

    def _handle_keyboard_interrupt(self):
        height, width = self.screen.getmaxyx()
        self.screen.clear()
        self.screen.addstr(
            height // 2, width // 2 - 11, "Keyboard Interrupt", curses.A_BOLD
        )
        self.screen.addstr(
            height // 2 + 1, width // 2 - 9, "Exiting...", curses.A_BOLD
        )
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
    if (
        "JUPYTER_RUNTIME_DIR" in os.environ
        or "IPYTHON_KERNEL_PATH" in os.environ
    ):
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
        oauth2_file = (
            "/nvme/scratch/work/tharvey/scripts/testing/client_secret.json"
        )
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


def calculate_ab_zeropoint(known_jy, unknown_counts):
    # Convert μJy to AB magnitude
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
        im_pixel_scale="HEADER",
        image_zp="HEADER",
        image_unit="HEADER",
        im_hdu_ext=0,
        wht_hdu_ext=0,
        err_hdu_ext=0,
        seg_hdu_ext=0,
    ):
        self.band_name = band_name
        self.instrument = instrument

        if wht_path == "im_folder":
            wht_path = image_path

        if err_path == "im_folder":
            err_path = image_path

        if seg_path == "im_folder":
            seg_path = image_path

        # Perform same checks for err, wht and seg

        if wht_path is not None and os.path.isdir(wht_path):
            wht_path_finder = glob.glob(
                os.path.join(wht_path, f"*{band_name.upper()}*wht*.fits")
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(wht_path, f"*{band_name.lower()}*wht*.fits")
                )
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
                glob.glob(
                    os.path.join(
                        wht_path, f"*{band_name.upper()}*weight*.fits"
                    )
                )
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(
                        wht_path, f"*{band_name.lower()}*weight*.fits"
                    )
                )
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
                glob.glob(
                    os.path.join(wht_path, f"*{band_name.upper()}*WHT*.fits")
                )
            )
            wht_path_finder.extend(
                glob.glob(
                    os.path.join(wht_path, f"*{band_name.lower()}*WHT*.fits")
                )
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
                    f"No files found for band {band_name}. Please provide full path."
                )
            else:
                wht_path = wht_path[0]
                print(
                    f"Auto detected weight path {wht_path} for band {band_name}."
                )

        if err_path is not None and os.path.isdir(err_path):
            err_path_finder = glob.glob(
                os.path.join(err_path, f"*{band_name.upper()}*err*.fits")
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(err_path, f"*{band_name.lower()}*err*.fits")
                )
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
                glob.glob(
                    os.path.join(err_path, f"*{band_name.upper()}*error*.fits")
                )
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(err_path, f"*{band_name.lower()}*error*.fits")
                )
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
                glob.glob(
                    os.path.join(err_path, f"*{band_name.upper()}*ERR*.fits")
                )
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(err_path, f"*{band_name.lower()}*ERR*.fits")
                )
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
                glob.glob(
                    os.path.join(err_path, f"*{band_name.upper()}*rms*.fits")
                )
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(err_path, f"*{band_name.lower()}*rms*.fits")
                )
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
                glob.glob(
                    os.path.join(err_path, f"*{band_name.upper()}*RMS*.fits")
                )
            )
            err_path_finder.extend(
                glob.glob(
                    os.path.join(err_path, f"*{band_name.lower()}*RMS*.fits")
                )
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
                    f"No files found for band {band_name}. Please provide full path."
                )
            else:
                err_path = err_path[0]
                print(
                    f"Auto detected error path {err_path} for band {band_name}."
                )

        # Check if imagepath is a folder
        if os.path.isdir(image_path):
            # Use glob to find the image
            image_path_finder = glob.glob(
                os.path.join(image_path, f"*{band_name.upper()}*.fits")
            )
            image_path_finder.extend(
                glob.glob(
                    os.path.join(image_path, f"*{band_name.lower()}*.fits")
                )
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
                ]
                # See if any one of the files has a common ending
                good_endings = []
                for ending in im_endings:
                    test_image_path = [im for im in image_path if ending in im]
                    good_endings.extend(test_image_path)
                if len(good_endings) == 1:
                    image_path = good_endings[0]
                    print(
                        f"Auto detected image path {image_path} for band {band_name}."
                    )
                elif len(good_endings) > 1:
                    raise ValueError(
                        f"Multiple files found for band {band_name} even when guessing ending. Please provide full path."
                    )
                else:
                    raise ValueError(
                        f"Multiple files found for band {band_name}. Please provide full path."
                    )

            elif len(image_path) == 0:
                raise ValueError(
                    f"No files found for band {band_name}. Please provide full path."
                )
            else:
                image_path = image_path[0]

        self.image_path = image_path
        self.wht_path = wht_path
        self.err_path = err_path
        self.seg_path = seg_path

        self.im_pixel_scale = im_pixel_scale
        self.image_zp = image_zp
        self.image_unit = image_unit
        self.im_hdu_ext = im_hdu_ext
        self.wht_hdu_ext = wht_hdu_ext
        self.err_hdu_ext = err_hdu_ext
        self.seg_hdu_ext = seg_hdu_ext

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
            print(
                f"Auto detected instrument {self.instrument} for band {self.band_name}."
            )

        if (
            self.image_path == self.wht_path == self.err_path
            or (self.wht_path == "im")
            or (self.err_path == "im")
        ):
            print(f"Detected single HDUList for {self.band_name}.")
            print(
                "Assuming JWST style HDUList with PrimaryHDU (0), SCI [1],  WHT [2], and ERR [3]."
            )
            if self.im_hdu_ext == 0:
                self.im_hdu_ext = "SCI"
            if self.wht_hdu_ext == 0:
                self.wht_hdu_ext = "WHT"
            if self.err_hdu_ext == 0:
                self.err_hdu_ext = "ERR"

            self.err_path = self.image_path
            self.wht_path = self.image_path

        # Open image to get header

        im_header = fits.open(self.image_path)[self.im_hdu_ext].header
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

        if (self.image_zp not in ["HEADER", None]) and (
            self.image_unit not in ["HEADER", None]
        ):
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
                print(
                    f"Auto detected pixel scale {self.im_pixel_scale:.4f} arcsec/pixel from WCS for band {self.band_name}."
                )


class FieldInfo:
    """
    Just a container for a list of PhotometryBandInfo objects.
    """

    def __init__(self, band_info_list):
        self.band_info_list = band_info_list
        # Get list of bands
        self.band_names = [band.band_name for band in band_info_list]
        # Get list of instruments
        self.instruments = [band.instrument for band in band_info_list]

        self.im_pixel_scales = {
            band.band_name: band.im_pixel_scale for band in band_info_list
        }
        self.im_zps = {
            band.band_name: band.image_zp for band in band_info_list
        }
        self.im_units = {
            band.band_name: band.image_unit for band in band_info_list
        }
        self.im_exts = {
            band.band_name: band.im_hdu_ext for band in band_info_list
        }
        self.wht_exts = {
            band.band_name: band.wht_hdu_ext for band in band_info_list
        }
        self.rms_err_exts = {
            band.band_name: band.err_hdu_ext for band in band_info_list
        }
        self.seg_exts = {
            band.band_name: band.seg_hdu_ext for band in band_info_list
        }

        self.im_paths = {
            band.band_name: band.image_path for band in band_info_list
        }
        self.wht_paths = {
            band.band_name: band.wht_path for band in band_info_list
        }
        self.err_paths = {
            band.band_name: band.err_path for band in band_info_list
        }
        self.seg_paths = {
            band.band_name: band.seg_path for band in band_info_list
        }


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

    origin_coord_pix = wcs.wcs_world2pix(
        origin_coord.ra.degree, origin_coord.dec.degree, 1
    )
    north_coord_pix = wcs.wcs_world2pix(
        north_coord.ra.degree, north_coord.dec.degree, 1
    )
    east_coord_pix = wcs.wcs_world2pix(
        east_coord.ra.degree, east_coord.dec.degree, 1
    )

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

    if return_ang:
        return np.degrees(ang)

    arrow_length = (
        arrow_length.to(u.arcsec).value / pix_scale.to(u.arcsec).value
    )

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
        origin_coord_pix[0]
        + compass_text_scale_factor * (dx_east + dx_offset),
        origin_coord_pix[1]
        + compass_text_scale_factor * (dy_east + dx_offset),
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
            nested_diff = find_dict_differences(
                dict1[key], dict2[key], current_path
            )

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
    cog = np.array([[phot_tab[coln][0] for coln in phot_tab.colnames[3:]]][0])

    return radii, cog
