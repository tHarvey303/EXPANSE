import pytest
import os
from EXPANSE.bagpipes import PipesFitNoLoad
import numpy as np
import matplotlib.pyplot as plt


# Get path of this file

current_dir = os.path.dirname(os.path.realpath(__file__))
bands = [
    "F435W",
    "F606W",
    "F775W",
    "F814W",
    "F850LP",
    "F090W",
    "F115W",
    "F150W",
    "F162M",
    "F182M",
    "F200W",
    "F210M",
    "F250M",
    "F277W",
    "F300M",
    "F335M",
    "F356W",
    "F410M",
    "F444W",
]


def test_PipesFitNoLoad():
    fig, ax = plt.subplots()
    for file in os.listdir(f"{current_dir}/posteriors"):
        if file.endswith(".h5"):
            print(f"Testing {file}")
            fit = PipesFitNoLoad(
                galaxy_id="test",
                field="test",
                h5_path=f"{current_dir}/posteriors/{file}",
                bands=bands,
            )
            items = fit._list_items()
            fit._load_item_from_h5(items)
            fit._recalculate_bagpipes_wavelength_array()
            fit.calculate_dof()
            fit._get_redshift()
            fit._recrate_bagpipes_time_grid()
            # Test plotting
            fit.plot_best_fit(ax=ax)
            ax.clear()
            # fit.plot_sed(ax=ax) haven't implemented properly yet.
            fit.plot_best_photometry(ax=ax)
            ax.clear()
            fit.plot_sfh(ax=ax, plottype="absolute")
            ax.clear()
            fit.plot_sfh(ax=ax, plottype="lookback")
            ax.clear()
            fit.plot_pdf(ax=ax, parameter="stellar_mass")
            ax.clear()
            fit.plot_corner_plot()
