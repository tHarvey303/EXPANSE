import curses
import os
import sys
import threading
import time

import matplotlib as mpl
import numpy as np
from astropy import units as u
from photutils import EllipticalAperture, aperture_photometry


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
    a,
    b,
    theta,
    kron_radius,
    psf,
    zero_point=28.08,
    aper_diam=0.32 * u.arcsec,
    sourcex_factor=6,
    pix_scale=0.03,
):
    """Scale fluxes to total flux using PSF
    Scale mag_aper to mag_auto (ideally measured in LW stack), and then derive PSF correction by placing a elliptical aperture around the PSF.
    """
    a = sourcex_factor * a
    b = sourcex_factor * b

    area_of_aper = np.pi * (aper_diam / 2) ** 2
    area_of_ellipse = np.pi * a * b
    scale_factor = area_of_aper / area_of_ellipse
    flux_auto = 10 ** ((zero_point - mag_auto) / 2.5)
    flux_aper = 10 ** ((zero_point - mag_aper) / 2.5)

    if scale_factor > 1:
        factor = flux_auto / flux_aper
        factor = np.clip(factor, 1, 1)
    else:
        factor = 1

    flux_aper_corrected = flux_aper * factor

    print(f"Corrected aperture magnitude by {factor} mag.")
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

    if a > psf.shape[0] or b > psf.shape[0]:
        # approximate as a circle
        np.sqrt(a * b) * kron_radius
        # encircled_energy = # enclosed_energy in F444W from band
    else:
        elliptical_aperture = EllipticalAperture(center, a=6 * a, b=6 * b, theta=theta)
        elliptical_aperture_phot = aperture_photometry(psf, elliptical_aperture)
        encircled_energy = elliptical_aperture_phot["aperture_sum"][0]

    flux_aper_total = flux_aper_corrected / encircled_energy
    mag_aper_total = -2.5 * np.log10(flux_aper_total) + zero_point
    return mag_aper_total


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
            progress_bar = (
                "[" + "#" * progress_width + " " * (width - 2 - progress_width) + "]"
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


def send_email(contents, subject='', address='tharvey303@gmail.com'):
    '''
	except Exception as e:
		# Email me if you crash
		ctime  = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
		send_email(contents=f'{e}', subject = f'{sys.argv[0]} crash at {ctime}')
		raise e

    '''
    import yagmail
    yagmail.SMTP('tcharvey303', oauth2_file='/nvme/scratch/work/tharvey/scripts/testing/client_secret.json').send(address, subject, contents)
    print('Sent email.')

