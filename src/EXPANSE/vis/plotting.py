import os
import re
import sys

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LogStretch,
    astropy_mpl_style,
)
from astropy.wcs import WCS
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.patheffects import withStroke
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import warnings
import cmasher
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from typing import Dict, Any, Tuple, List, Optional
import matplotlib as mpl

warnings.filterwarnings("ignore")

from ..utils import compass

sys.path.append("/nvme/scratch/software/trilogy/")
try:
    from trilogy3 import Trilogy
except ImportError:
    pass

mpl.rcParams["figure.dpi"] = 300
plt.style.use(astropy_mpl_style)
# Enable LaTeX
mpl.rcParams["text.usetex"] = False
# change font to palatino

# mpl.rcParams['font.family'] =  'Palatino'
# mpl.rcParams['font.serif'] =
# mpl.rcParams['font.size'] = 14

mpl.use("pdf")


def find_renderer(fig):
    if hasattr(fig.canvas, "get_renderer"):
        # Some backends, such as TkAgg, have the get_renderer method, which
        # makes this easy.
        renderer = fig.canvas.get_renderer()
    elif hasattr(fig, "_get_renderer"):
        # In other backends, we can use the _get_renderer() method of the figure.
        renderer = fig._get_renderer()
    else:
        # Other backends do not have the get_renderer method, so we have a work
        # around to find the renderer.  Print the figure to a temporary file
        # object, and then grab the renderer that was used.
        # (I stole this trick from the matplotlib backend_bases.py
        # print_figure() method.)
        import io

        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return renderer


def optimize_panel_spacing(total_length, panels):
    """
    Optimize spacing of panels in one dimension.

    Args:
        total_length: Total available length (width or height)
        panels: List of (position, size) tuples

    Returns:
        list: New positions for panels, maintaining relative order
    """
    if not panels:
        return []

    # Sort by position
    sorted_panels = sorted(enumerate(panels), key=lambda x: x[1][0])
    original_indices = [x[0] for x in sorted_panels]
    sizes = [panel[1][1] for panel in sorted_panels]

    total_size = sum(sizes)
    n_gaps = len(sizes) - 1

    if n_gaps > 0:
        spacing = (total_length - total_size) / n_gaps
        start = 0
    else:
        spacing = 0
        start = (total_length - total_size) / 2

    new_positions = [0] * len(panels)
    current_pos = start

    for i, size in enumerate(sizes):
        new_positions[original_indices[i]] = current_pos
        current_pos += size + spacing

    return new_positions


def show_inset_region(ax, region, color="white", linewidth=0.5):
    """Create a rectangle showing the inset region"""
    width = region["xlim"][1] - region["xlim"][0]
    height = region["ylim"][1] - region["ylim"][0]
    rect = Rectangle(
        (region["xlim"][0], region["ylim"][0]),
        width,
        height,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        aa=True,
        zorder=1000,
    )
    ax.add_patch(rect)
    return rect


def create_connectors(ax, ax_inset, side, color="white", linewidth=0.5):
    if side == "top":
        # Connect bottom corners of inset to top corners of region
        corners = [
            (
                "axes fraction",
                "axes fraction",
                0,
                0,
                "data",
                "data",
                ax_inset.get_xlim()[0],
                ax_inset.get_ylim()[1],
            ),  # bottom left
            (
                "axes fraction",
                "axes fraction",
                1,
                0,
                "data",
                "data",
                ax_inset.get_xlim()[1],
                ax_inset.get_ylim()[1],
            ),  # bottom right
        ]
    elif side == "bottom":
        # Connect top corners of inset to bottom corners of region
        corners = [
            (
                "axes fraction",
                "axes fraction",
                0,
                1,
                "data",
                "data",
                ax_inset.get_xlim()[0],
                ax_inset.get_ylim()[0],
            ),  # top left
            (
                "axes fraction",
                "axes fraction",
                1,
                1,
                "data",
                "data",
                ax_inset.get_xlim()[1],
                ax_inset.get_ylim()[0],
            ),  # top right
        ]
    elif side == "left":
        # Connect right corners of inset to left corners of region
        corners = [
            (
                "axes fraction",
                "axes fraction",
                1,
                0,
                "data",
                "data",
                ax_inset.get_xlim()[1],
                ax_inset.get_ylim()[0],
            ),  # bottom right
            (
                "axes fraction",
                "axes fraction",
                1,
                1,
                "data",
                "data",
                ax_inset.get_xlim()[1],
                ax_inset.get_ylim()[1],
            ),  # top right
        ]
    elif side == "right":
        # Connect left corners of inset to right corners of region
        corners = [
            (
                "axes fraction",
                "axes fraction",
                0,
                0,
                "data",
                "data",
                ax_inset.get_xlim()[0],
                ax_inset.get_ylim()[0],
            ),  # bottom left
            (
                "axes fraction",
                "axes fraction",
                0,
                1,
                "data",
                "data",
                ax_inset.get_xlim()[0],
                ax_inset.get_ylim()[1],
            ),  # top left
        ]

    connectors = []
    for coordsA, coordsB, xA, yA, coordsC, coordsD, xB, yB in corners:
        con = ConnectionPatch(
            xyA=(xA, yA),
            xyB=(xB, yB),
            coordsA=coordsA,
            coordsB=coordsC,
            axesA=ax_inset,
            axesB=ax,
            color=color,
            linewidth=linewidth,
            zorder=1000,
            aa=True,
        )
        ax.add_artist(con)
        connectors.append(con)

    return connectors


def create_plot_with_insets(
    image_data,
    wcs,
    inset_regions,
    figure_size=(15, 15),
    dpi=200,
    ax_pad=0.01,
    ax_pad_bottom=0.05,
    allowed_sides=["right", "left", "top", "bottom"],
    cmap="gray",
    axis_type="wcs",
    normalize_inset_spacing=True,
    figure_xticks=True,
    figure_yticks=True,
    figure_xlabel=True,
    figure_ylabel=True,
    renorm_insets=True,
    facecolor="black",
    axis_color="white",
    text_color="white",  # Calculate loc1, loc2 for mark_inset given each side - should be corners closest to that side
    loc_sides={"left": (2, 3), "right": (1, 4), "top": (3, 4), "bottom": (1, 2)},
    add_compass=False,
    compass_params={},
    fix_single_color_blocks=True,
):
    """
    Create an astronomical image plot with multiple inset plots positioned intelligently.

    Parameters:
    -----------
    image_data : numpy.ndarray
        The full image data array
    wcs : astropy.wcs.WCS
        World Coordinate System object for the image
    inset_regions : list of dict
        List of dictionaries containing inset specifications. Each dict should have:
        - 'xlim': tuple of (min, max) x coordinates
        - 'ylim': tuple of (min, max) y coordinates
    figure_size : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for the figure
    ax_pad : float
        Padding between main image and insets
    ax_pad_bottom : float
        Padding between main image and bottom insets
    allowed_sides : list
        List of sides where insets are allowed. Can be 'right', 'left', 'top', 'bottom'
    cmap : str
        Colormap for the image
    axis_type : str
        Type of axis, can be 'wcs' or 'pixel'
    normalize_inset_spacing : bool
        Whether to normalize the spacing between insets
    figure_xticks : bool
        Whether to show x ticks on the main image
    figure_yticks : bool
        Whether to show y ticks on the main image
    figure_xlabel : bool
        Whether to show x label on the main image
    figure_ylabel : bool
        Whether to show y label on the main image
    renorm_insets : bool
        Whether to renormalize the insets
    facecolor : str
        Background color for the figure
    axis_color : str
        Color for the axis ticks
    text_color : str
        Color for the axis labels
    loc_sides : dict
        Dictionary of loc1, loc2 for mark_inset given each side - should be corners closest to that side
    add_compass : bool
        Whether to add a compass to the plot
    compass_params : dict
        Dictionary of parameters for the compass
    fix_single_color_blocks : bool
        Whether to fix single color blocks in the RGB image

    Returns:
    --------
    tuple
        (fig, ax, list of inset axes)
    """

    def calculate_inset_position(
        region,
        main_shape,
        data_shape,
        fig,
        main_ax_bbox,
        inset_height=0.30,
        ax_pad=0.01,
        ax_pad_bottom=0.05,
        allowed_sides=["right", "left", "top", "bottom"],
        current_inset_axes=[],
    ):
        """
        Calculate the best position for an inset based on its location in the main image
        and existing inset positions.
        """
        # Calculate center and edges of the region
        x_center = (region["xlim"][0] + region["xlim"][1]) / 2
        y_center = (region["ylim"][0] + region["ylim"][1]) / 2

        # Calculate distances to edges of main image
        dist_to_left = x_center
        dist_to_right = main_shape[1] - x_center
        dist_to_bottom = y_center
        dist_to_top = main_shape[0] - y_center

        # Calculate size of square inset given the figure aspect ratio

        inset_width = inset_height * (main_shape[0] / main_shape[1])
        # Then scale to the data aspect ratio
        inset_width = inset_width * (data_shape[1] / data_shape[0])

        # print('Inset fig width:', inset_width)
        # print('Inset fig height:', inset_height)

        # Find the closest edge
        distances = {
            "right": dist_to_right,
            "left": dist_to_left,
            "top": dist_to_top,
            "bottom": dist_to_bottom,
        }

        # Filter out sides that are not allowed
        distances = {k: v for k, v in distances.items() if k in allowed_sides}

        # Sort edges by distance
        sorted_edges = sorted(distances.items(), key=lambda x: x[1])

        # print('Coordinates:', x_center, y_center, 'has closest edge:', side)

        positions = {
            "right": (1 + ax_pad, y_center / main_shape[0] - inset_height / 2),
            "left": (-inset_width - ax_pad, y_center / main_shape[0] - inset_height / 2),
            "top": (x_center / main_shape[1] - inset_width / 2, 1 + ax_pad),
            "bottom": (x_center / main_shape[1] - inset_width / 2, -inset_height - ax_pad_bottom),
        }
        side = sorted_edges[0][0]
        ax_pos = positions[side]
        # Check if the inset will overlap with any existing inset or the main image or the axis ticks or labels
        ax = fig.axes[0]
        # Get the position of the inset in the figure coordinates
        inset_pos = ax.transAxes.transform(ax_pos)
        inset_width_fig = (
            ax.transAxes.transform((ax_pos[0] + inset_width, ax_pos[1]))[0] - inset_pos[0]
        )
        inset_height_fig = (
            ax.transAxes.transform((ax_pos[0], ax_pos[1] + inset_height))[1] - inset_pos[1]
        )

        # print('Inset position:', inset_pos)
        # print('Inset width:', inset_width_fig)
        # print('Inset height:', inset_height_fig)

        # Check if the inset will overlap with any existing inset or the main image or the axis ticks or labels
        # Get bounding boxes from fig
        # First ax is main ax, the rest are inset axes
        other_ax_bbox = [ax.get_tightbbox(find_renderer(fig)) for ax in current_inset_axes]

        inset_bbox = mpl.transforms.Bbox.from_bounds(
            inset_pos[0], inset_pos[1], inset_width_fig, inset_height_fig
        )

        position_shift_step = {
            "right": (0.01, 0),
            "left": (-0.01, 0),
            "top": (0, 0.01),
            "bottom": (0, -0.01),
        }

        ax_overlap_shift_step = {
            "right": (0, 0.01),
            "left": (0, 0.01),
            "top": (0.01, 0),
            "bottom": (0.01, 0),
        }

        if main_ax_bbox.overlaps(inset_bbox):
            # Calculate new position - horizontal shift for left or right, vertical shift for top or bottom
            # Iteratively shift
            # print('Overlap with main axis, shifting.')
            # print('Initial position:', ax_pos)
            counter = 0
            while main_ax_bbox.overlaps(inset_bbox):
                # print(f'Overlap with main axis, {counter}')
                ax_pos = (
                    ax_pos[0] + position_shift_step[side][0],
                    ax_pos[1] + position_shift_step[side][1],
                )
                inset_pos = ax.transAxes.transform(ax_pos)
                inset_bbox = mpl.transforms.Bbox.from_bounds(
                    inset_pos[0], inset_pos[1], inset_width_fig, inset_height_fig
                )

                counter += 1
                if counter > 100:
                    print("Counter exceeded 100, breaking...")
                    break

            # print('Overlap fixed, new position:', ax_pos)
        # Check if the inset will overlap with any existing inset
        for bb_ax in tqdm(other_ax_bbox, desc="Checking overlap with existing insets"):
            if bb_ax.overlaps(inset_bbox):
                # print('Overlap with existing inset')
                overlap_area = inset_bbox.intersection(bb_ax, inset_bbox)
                # no .area method - calculate area manually
                overlap_area = (overlap_area.x1 - overlap_area.x0) * (
                    overlap_area.y1 - overlap_area.y0
                )
                # print('Overlap area:', overlap_area)
                # Calculate vector to shift the inset - vertical shift for left or right, horizontal shift for top or bottom
                # Try both directions first, see which decreases overlap area more
                test_positions = [
                    (
                        ax_pos[0] + ax_overlap_shift_step[side][0],
                        ax_pos[1] + ax_overlap_shift_step[side][1],
                    ),
                    (
                        ax_pos[0] - ax_overlap_shift_step[side][0],
                        ax_pos[1] - ax_overlap_shift_step[side][1],
                    ),
                ]

                test_positions = [ax.transAxes.transform(pos) for pos in test_positions]
                inset_test_bbox_1 = mpl.transforms.Bbox.from_bounds(
                    test_positions[0][0], test_positions[0][1], inset_width_fig, inset_height_fig
                )
                inset_test_bbox_2 = mpl.transforms.Bbox.from_bounds(
                    test_positions[1][0], test_positions[1][1], inset_width_fig, inset_height_fig
                )
                inset_test_bbox = [inset_test_bbox_1, inset_test_bbox_2]
                sign_use = None
                for test_bbox, sign in zip(inset_test_bbox, [1, -1]):
                    if bb_ax.overlaps(test_bbox):
                        test_overlap_area = test_bbox.intersection(bb_ax, test_bbox)
                        test_overlap_area = (test_overlap_area.x1 - test_overlap_area.x0) * (
                            test_overlap_area.y1 - test_overlap_area.y0
                        )

                        if test_overlap_area < overlap_area:
                            sign_use = sign
                            break
                    else:
                        sign_use = sign
                        break
                if sign_use is None:
                    print("No suitable shift found, skipping...")
                    print(inset_test_bbox)
                    print(bb_ax)
                    continue

                counter = 0

                while bb_ax.overlaps(inset_bbox):
                    # Calculate fraction of overlap
                    # Calculate new position - Vertical shift for left or right, horizontal shift for top or bottom. Try both directions.
                    new_ax_pos = (
                        ax_pos[0] + sign_use * ax_overlap_shift_step[side][0] * counter,
                        ax_pos[1] + sign_use * ax_overlap_shift_step[side][1] * counter,
                    )
                    inset_ax_pos = ax.transAxes.transform(new_ax_pos)
                    new_inset_bbox = mpl.transforms.Bbox.from_bounds(
                        inset_ax_pos[0], inset_ax_pos[1], inset_width_fig, inset_height_fig
                    )

                    inset_bbox = new_inset_bbox
                    counter += 1

                extra_shift = 1
                # Add an extra shift to make sure inset is not touching the other inset
                ax_pos = (
                    ax_pos[0] + sign_use * ax_overlap_shift_step[side][0] * (counter + extra_shift),
                    ax_pos[1] + sign_use * ax_overlap_shift_step[side][1] * (counter + extra_shift),
                )
                # print('Overlap fixed, new position:', ax_pos)

                # ax_pos = new_ax_pos

        return ax_pos, (inset_width, inset_height), side

    def create_inset_ax(pos, size, region, inset_data):
        # Create inset
        ax_inset = ax.inset_axes(
            [pos[0], pos[1], size[0], size[1]], xlim=region["xlim"], ylim=region["ylim"]
        )

        if is_rgb:
            norm_inset = None
        else:
            # Set up normalization for inset
            if renorm_insets:
                norm_inset = ImageNormalize(
                    inset_data,
                    interval=AsymmetricPercentileInterval(0.1, 99.9),
                    stretch=LogStretch(),
                    clip=False,
                    invalid=np.nan,
                )
            else:
                norm_inset = norm

        # Display inset
        ax_inset.imshow(image_data, cmap=cmap, origin="lower", norm=norm_inset)
        ax_inset.grid(False)

        # Style inset
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        return ax_inset

    # Create main figure
    fig, ax = plt.subplots(
        subplot_kw={"projection": wcs} if axis_type == "wcs" else {},
        figsize=figure_size,
        dpi=dpi,
        facecolor=facecolor,
        edgecolor=facecolor,
    )

    # Handle NaN values

    # Set up normalization for main plot
    # Check if already a RGB (shape 3)
    if image_data.ndim == 3:
        print("RGB image detected, skipping normalization...")
        is_rgb = True
        norm = None
        cmap = None
        # There are image regions which are all red or all blue or all green, due to lack of overlapping data
        # Copy the other channels to these regions
        if fix_single_color_blocks:
            # This will only work for non-scaled RGB images.
            # Find all red, green, and blue regions
            red_regions = image_data[:, :, 0] == 0
            green_regions = image_data[:, :, 1] == 0
            blue_regions = image_data[:, :, 2] == 0
            only_red = np.all([red_regions, ~green_regions, ~blue_regions], axis=0)
            only_green = np.all([~red_regions, green_regions, ~blue_regions], axis=0)
            only_blue = np.all([~red_regions, ~green_regions, blue_regions], axis=0)
            # Copy the other channels to these regions
            image_data[only_red, 0] = image_data[only_red, 1]
            image_data[only_red, 2] = image_data[only_red, 1]
            image_data[only_green, 0] = image_data[only_green, 1]
            image_data[only_green, 2] = image_data[only_green, 1]
            image_data[only_blue, 0] = image_data[only_blue, 1]
            image_data[only_blue, 2] = image_data[only_blue, 1]

    else:
        is_rgb = False
        image_data[image_data == 0] = np.nan
        norm = ImageNormalize(
            image_data,
            interval=AsymmetricPercentileInterval(0.1, 99.9),
            stretch=LogStretch(),
            clip=False,
            invalid=np.nan,
        )

    # Display main image
    ax.imshow(image_data, cmap=cmap, origin="lower", norm=norm)
    ax.grid(False)
    ax.set_facecolor(facecolor)

    # Style main plot
    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    x_text = "R.A. (J2000)" if axis_type == "wcs" else "x"
    y_text = "Dec (J2000)" if axis_type == "wcs" else "y"
    if not figure_xticks:
        ax.set_xticks([])
    if not figure_yticks:
        ax.set_yticks([])
    if figure_xlabel:
        ax.set_xlabel(
            x_text,
            color=text_color,
            fontsize=18,
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )
    if figure_ylabel:
        ax.set_ylabel(
            y_text,
            color=text_color,
            fontsize=18,
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    # Create custom colormap for insets
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color="black")

    # Keep track of inset positions and axes
    existing_positions = []
    inset_side_sizes = {}
    inset_side_positions = {}
    inset_axes_sides = {}
    inset_datas = {}
    regions_side = {}
    inset_axes = []

    main_ax_bbox = ax.get_tightbbox(find_renderer(fig))

    # Create insets
    for region in tqdm(inset_regions, desc="Positioning insets"):
        # Get inset data
        inset_data = image_data[
            region["ylim"][0] : region["ylim"][1], region["xlim"][0] : region["xlim"][1]
        ]

        if np.isnan(inset_data).all():
            print(f"Inset at {region} is empty, skipping...")
            continue

        # Calculate position for this inset
        pos, size, side = calculate_inset_position(
            region,
            image_data.shape,
            inset_data.shape,
            fig,
            main_ax_bbox,
            ax_pad_bottom=ax_pad_bottom,
            ax_pad=ax_pad,
            allowed_sides=allowed_sides,
            current_inset_axes=inset_axes,
        )

        existing_positions.append(pos)

        if side not in inset_side_sizes.keys():
            inset_side_positions[side] = []
            inset_side_sizes[side] = []
            inset_axes_sides[side] = []
            inset_datas[side] = []
            regions_side[side] = []

        inset_side_positions[side].append(pos)
        inset_side_sizes[side].append(size)
        inset_datas[side].append(inset_data)
        regions_side[side].append(region)

        # Create inset axis
        ax_inset = create_inset_ax(pos, size, region, inset_data)

        # Add connection lines
        if not normalize_inset_spacing:
            show_inset_region(ax, region, color="white", linewidth=0.5)
            connectors = create_connectors(ax, ax_inset, side, color=axis_color, linewidth=0.5)

        inset_axes.append(ax_inset)
        inset_axes_sides[side].append(ax_inset)

    print("Inset positions:", inset_side_positions)
    if normalize_inset_spacing:
        for side in inset_side_positions.keys():
            if side in ["right", "left"]:
                input_arr = [
                    (position[1], size[1])
                    for position, size in zip(inset_side_positions[side], inset_side_sizes[side])
                ]
            elif side in ["top", "bottom"]:
                input_arr = [
                    (position[0], size[0])
                    for position, size in zip(inset_side_positions[side], inset_side_sizes[side])
                ]

            new_position = optimize_panel_spacing(1, input_arr)
            for i, (position, size) in enumerate(
                zip(inset_side_positions[side], inset_side_sizes[side])
            ):
                if side in ["right", "left"]:
                    inset_side_positions[side][i] = (position[0], new_position[i], size[0], size[1])
                elif side in ["top", "bottom"]:
                    inset_side_positions[side][i] = (new_position[i], position[1], size[0], size[1])

            for ax_inset, position, region, inset_data in zip(
                inset_axes_sides[side],
                inset_side_positions[side],
                regions_side[side],
                inset_datas[side],
            ):
                # Copy axis and delete the old one
                # Delete ax_inset
                ax_inset.remove()

                ax_inset = create_inset_ax(position[:2], position[2:], region, inset_data)
                # ax.indicate_inset_zoom(ax_inset, edgecolor='white')
                # mark_inset(ax, ax_inset, loc1=loc_sides[side][0], loc2=loc_sides[side][1], fc="none", ec=axis_color, lw=0.5)
                show_inset_region(ax, region, color="white", linewidth=0.5)
                connectors = create_connectors(ax, ax_inset, side, color=axis_color, linewidth=0.5)

    print("New inset positions:", inset_side_positions)
    # ax.grid(False)
    # draw the main bbox
    if add_compass:
        if "arrow_length" not in compass_params.keys():
            compass_params["arrow_length"] = 1 * u.arcminute
        if "arrow_width" not in compass_params.keys():
            compass_params["arrow_width"] = 200

        if "compass_loc" not in compass_params.keys():
            compass_params["compass_loc"] = "bottom right"

        # Possible locations in figure coordinates

        locations = {
            "top right": (0.95, 0.95),
            "top left": (0.05, 0.95),
            "bottom right": (0.95, 0.05),
            "bottom left": (0.05, 0.05),
        }

        # Compute ra and dec of chosen location using WCS

        # firstly figure to axes coordinates
        fig_coords = locations[compass_params["compass_loc"]]
        if axis_type == "wcs":
            ra, dec = ax.transAxes.inverted().transform(fig_coords)

        else:
            x, y = ax.transAxes.inverted().transform(fig_coords)
            print(x, y, wcs)
            ra, dec = wcs.all_pix2world(x, y, 0)
            print(ra, dec)

        compass(
            ra,
            dec,
            wcs,
            ax,
            arrow_length=compass_params["arrow_length"],
            x_ax="ra",
            ang_text=False,
            arrow_width=compass_params["arrow_width"],
            arrow_color=text_color,
            text_color=text_color,
            fontsize="large",
            return_ang=False,
            pix_scale=0.03 * u.arcsec,
            compass_text_scale_factor=1.15,
        )

    return fig, ax, inset_axes


def load_regions_from_reg(reg_path, wcs=None):
    import regions

    reg = regions.Regions.read(reg_path)
    regions_list = []

    for r in reg:
        # Check if class is inherited from SkyRegion
        if issubclass(type(r), regions.SkyRegion):
            r = r.to_pixel(wcs)

        x_center = r.center.x
        y_center = r.center.y
        width = r.width
        height = r.height
        x_low = np.round(x_center - width / 2).astype(int)
        x_high = np.round(x_center + width / 2).astype(int)
        y_low = np.round(y_center - height / 2).astype(int)
        y_high = np.round(y_center + height / 2).astype(int)

        regions_list.append({"xlim": (x_low, x_high), "ylim": (y_low, y_high)})

    return regions_list


def create_galaxy_pie_visualization(
    data_dict,
    cmaps=None,
    vlims=None,
    labels=None,
    galaxy_name="Galaxy",
    figsize=(12, 12),
    add_dividers=True,
    divider_width=2,
    divider_color="white",
    center_circle_radius=0.05,
    rotation_offset=0,
):
    """
    Create a pie-slice visualization of galaxy properties.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys as property names and values as 2D arrays
    cmaps : dict or None
        Dictionary mapping property names to colormaps
    vlims : dict or None
        Dictionary mapping property names to (vmin, vmax) tuples
    labels : dict or None
        Dictionary mapping property names to display labels
    galaxy_name : str
        Name of the galaxy for the title
    figsize : tuple
        Figure size
    add_dividers : bool
        Whether to add white divider lines between slices
    divider_width : float
        Width of divider lines
    divider_color : str
        Color of divider lines
    center_circle_radius : float
        Radius of central circle (fraction of image radius)
    rotation_offset : float
        Rotation offset in radians for animation

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Get number of properties
    n_props = len(data_dict)
    prop_names = list(data_dict.keys())

    # Set default colormaps if not provided
    if cmaps is None:
        default_cmaps = {
            "stellar_mass": "viridis",
            "dust_attenuation": "YlOrRd",
            "sfr": "plasma",
            "metallicity": "coolwarm",
            "age": "copper",
            "velocity": "RdBu_r",
        }
        cmaps = {}
        for prop in prop_names:
            # Try to match property name with defaults
            for key in default_cmaps:
                if key in prop.lower().replace(" ", "_"):
                    cmaps[prop] = default_cmaps[key]
                    break
            else:
                # Fallback colormap
                cmaps[prop] = plt.cm.list_cmap_names()[
                    prop_names.index(prop) % len(plt.cm.list_cmap_names())
                ]

    # Set default labels if not provided
    if labels is None:
        labels = {prop: prop.replace("_", " ").title() for prop in prop_names}

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get image dimensions (assume all arrays have same shape)
    first_data = list(data_dict.values())[0]
    ny, nx = first_data.shape

    # Create coordinate grids
    x = np.arange(nx) - nx / 2
    y = np.arange(ny) - ny / 2
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # Normalize R for center circle
    R_max = np.max(R)
    R_norm = R / R_max

    # Calculate angle ranges for each slice with rotation offset
    angle_step = 2 * np.pi / n_props
    angle_ranges = [
        (
            (i * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi,
            ((i + 1) * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi,
        )
        for i in range(n_props)
    ]

    # Plot each property in its slice
    for i, (prop, data) in enumerate(data_dict.items()):
        # Create mask for this slice
        angle_min, angle_max = angle_ranges[i]

        # Handle angle wrapping
        if angle_min > angle_max:
            # Slice wraps around -π/π boundary
            mask = ((Theta >= angle_min) | (Theta <= angle_max)) & (R_norm > center_circle_radius)
        else:
            mask = (Theta >= angle_min) & (Theta <= angle_max) & (R_norm > center_circle_radius)

        # Apply mask to data
        masked_data = np.ma.array(data, mask=~mask)

        # Get colormap and limits
        cmap = cmaps.get(prop, "viridis")
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if vlims and prop in vlims:
            vmin, vmax = vlims[prop]
        else:
            vmin, vmax = np.nanpercentile(data[mask], [2, 98]) if np.any(mask) else (None, None)

        # Plot the masked data
        im = ax.imshow(
            masked_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[-nx / 2, nx / 2, -ny / 2, ny / 2],
            origin="lower",
            interpolation="nearest",
        )

        # Add text label for this slice
        angle_mid = (angle_min + angle_max) / 2
        if angle_min > angle_max:  # Handle wrapping
            angle_mid = (angle_min + angle_max + 2 * np.pi) / 2
            if angle_mid > np.pi:
                angle_mid -= 2 * np.pi

        label_r = R_max * 0.85  # Position label at 85% of radius
        label_x = label_r * np.cos(angle_mid)
        label_y = label_r * np.sin(angle_mid)

        # Determine text rotation
        rotation = np.degrees(angle_mid)
        if rotation > 90 or rotation < -90:
            rotation += 180
            ha = "right"
        else:
            ha = "left"

        ax.text(
            label_x,
            label_y,
            labels.get(prop, prop),
            rotation=rotation,
            ha=ha,
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
        )

    # Add divider lines if requested
    if add_dividers:
        for i in range(n_props):
            angle = (i * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi
            x_end = R_max * np.cos(angle)
            y_end = R_max * np.sin(angle)
            x_start = center_circle_radius * R_max * np.cos(angle)
            y_start = center_circle_radius * R_max * np.sin(angle)
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=divider_color,
                linewidth=divider_width,
                zorder=10,
            )

    # Add central circle
    if center_circle_radius > 0:
        circle = plt.Circle((0, 0), center_circle_radius * R_max, color="black", zorder=11)
        ax.add_patch(circle)

    # Set axis properties
    ax.set_xlim(-nx / 2, nx / 2)
    ax.set_ylim(-ny / 2, ny / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    ax.set_title(
        f"{galaxy_name} - Multi-Property Visualization", fontsize=16, fontweight="bold", pad=20
    )

    plt.tight_layout()

    return fig, ax


def create_galaxy_animation(
    data_dict: Dict[str, np.ndarray],
    galaxy: Any,  # Pass your actual galaxy object here
    rgb: Optional[np.ndarray] = None,
    cmaps: Optional[Dict[str, Any]] = None,
    vlims: Optional[Dict[str, Tuple[float, float]]] = None,
    labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    static_rgb_frames: int = 20,
    transition_frames: int = 60,
    rotation_frames: int = 120,
    fps: int = 30,
    save_path: Optional[str] = None,
    writer: str = "pillow",
    show_progress: bool = True,
    ref_band: str = "F200W",
) -> FuncAnimation:
    """
    Creates a galaxy animation with sleek styling and a smooth transition.

    Features:
    - Modern, minimalist aesthetic with a clean sans-serif font.
    - Smooth clockwise wipe transition that leaves dividers behind.
    - Clockwise rotation of the final map.
    - Correctly positioned labels and scalebars.
    - The RGB image remains as a persistent background.
    """
    # --- Modern Styling Setup ---
    plt.style.use("dark_background")
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'sans-serif']

    ui_color = "#F0F0F0"  # A softer off-white for UI elements

    if vlims is None:
        vlims = {
            prop: (np.nanpercentile(data, 2), np.nanpercentile(data, 98))
            for prop, data in data_dict.items()
        }

    total_frames = static_rgb_frames + transition_frames + rotation_frames
    fig, ax = plt.subplots(figsize=figsize)

    first_data = next(iter(data_dict.values()))
    ny, nx = first_data.shape
    x, y = np.arange(nx) - nx / 2, np.arange(ny) - ny / 2
    X, Y = np.meshgrid(x, y)
    R, Theta = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    R_max = np.max(R) if R.size > 0 else 1.0

    prop_names, n_props = list(data_dict.keys()), len(data_dict)
    angle_step = 2 * np.pi / n_props
    rotation_angles = np.linspace(0, -2 * np.pi, rotation_frames, endpoint=False)

    def update(frame: int):
        if show_progress and frame % 10 == 0:
            print(f"Rendering frame {frame}/{total_frames}")

        for artist in ax.artists + ax.patches + ax.lines + ax.texts:
            artist.remove()
        for image in ax.images:
            image.remove()
        for cbar_ax in fig.axes[1:]:
            cbar_ax.remove()
        ax.axis("off")

        is_transition = static_rgb_frames <= frame < static_rgb_frames + transition_frames
        is_rotation = frame >= static_rgb_frames + transition_frames

        drawn_artists = []

        if rgb is not None:
            drawn_artists.append(
                ax.imshow(rgb, origin="lower", extent=[-nx / 2, nx / 2, -ny / 2, ny / 2], zorder=0)
            )

        try:
            scalebar1 = AnchoredSizeBar(
                ax.transData,
                1 / galaxy.im_pixel_scales[ref_band].value,
                '1"',
                "lower right",
                pad=0.5,
                color=ui_color,
                frameon=False,
                size_vertical=0.8,
                fontproperties=FontProperties(size=18),
                label_top=True,
            )
            drawn_artists.append(ax.add_artist(scalebar1))
            re = 40.1
            d_A = cosmo.angular_diameter_distance(galaxy.redshift)
            pix_scal = u.pixel_scale(galaxy.im_pixel_scales[ref_band].value * u.arcsec / u.pixel)
            re_as = (re * u.pixel).to(u.arcsec, pix_scal)
            re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())
            scalebar2 = AnchoredSizeBar(
                ax.transData,
                re,
                f"{re_kpc:.0f}",
                "lower left",
                pad=0.5,
                color=ui_color,
                frameon=False,
                size_vertical=0.8,
                fontproperties=FontProperties(size=24),
                label_top=True,
            )
            drawn_artists.append(ax.add_artist(scalebar2))
        except Exception as e:
            print(f"Warning: Could not draw scalebars. Error: {e}")

        if is_transition or is_rotation:
            rotation_offset = (
                rotation_angles[frame - (static_rgb_frames + transition_frames)]
                if is_rotation
                else 0.0
            )

            wipe_angle, start_angle, end_angle = 0, 0, 0
            if is_transition:
                progress = (frame - static_rgb_frames) / transition_frames
                wipe_angle, start_angle = progress * 2 * np.pi, np.pi / 2
                end_angle = start_angle - wipe_angle

            for i, prop in enumerate(prop_names):
                data, cmap = data_dict[prop], cmaps.get(prop, "viridis") if cmaps else "viridis"
                vmin, vmax = vlims[prop]

                angle_min = (i * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi
                angle_max = ((i + 1) * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi

                slice_mask = (
                    (Theta >= angle_min) | (Theta <= angle_max)
                    if angle_min > angle_max
                    else (Theta >= angle_min) & (Theta <= angle_max)
                )

                if is_transition:
                    reveal_mask = (
                        (Theta >= (end_angle + 2 * np.pi)) | (Theta <= start_angle)
                        if end_angle < -np.pi
                        else (Theta >= end_angle) & (Theta <= start_angle)
                    )
                    final_mask = slice_mask & reveal_mask
                else:
                    final_mask = slice_mask

                d = ax.imshow(
                    np.ma.array(data, mask=~final_mask),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[-nx / 2, nx / 2, -ny / 2, ny / 2],
                    origin="lower",
                    zorder=1,
                )

                drawn_artists.append(d)

                x1, y1 = np.cos(angle_min), np.sin(angle_min)
                x2, y2 = np.cos(angle_max), np.sin(angle_max)
                angle_mid = np.arctan2(y1 + y2, x1 + x2)

                show_label = True
                if is_transition:
                    mid_norm = (angle_mid + 2 * np.pi) % (2 * np.pi)
                    start_norm = (start_angle + 2 * np.pi) % (2 * np.pi)
                    end_norm = (end_angle + 2 * np.pi) % (2 * np.pi)
                    show_label = (
                        (end_norm <= mid_norm <= start_norm)
                        if start_norm >= end_norm
                        else ((mid_norm >= end_norm) or (mid_norm <= start_norm))
                    )

                if show_label:
                    label_r = R_max * 0.65
                    label_x, label_y = label_r * np.cos(angle_mid), label_r * np.sin(angle_mid)
                    text = ax.text(
                        label_x,
                        label_y,
                        labels.get(prop, prop) if labels else prop,
                        ha="center",
                        va="center",
                        fontsize=22,
                        color=ui_color,
                        zorder=20,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
                    )

                    drawn_artists.append(text)
                    """cbar_width, cbar_height = 0.12, 0.012
                    inv = ax.transData.inverted()
                    text_bb = text.get_window_extent(renderer=find_renderer(fig))
                    text_data_bb = inv.transform(text_bb)
                    cbar_x_center = (text_data_bb[0, 0] + text_data_bb[1, 0]) / 2
                    cbar_y_center = text_data_bb[0, 1] - (text_data_bb[0,1]-text_data_bb[1,1]) - 5
                    
                    cax = fig.add_axes([cbar_x_center, cbar_y_center, cbar_width, cbar_height], 
                                     transform=ax.transData, anchor='N')
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
                    cbar.set_ticks([])
                    cbar.outline.set_edgecolor(ui_color)
                    cbar.outline.set_linewidth(0.5)
                    """
            if is_rotation:
                for i in range(n_props):
                    angle = (i * angle_step - np.pi + rotation_offset) % (2 * np.pi) - np.pi
                    x_end, y_end = 0.70 * R_max * np.cos(angle), 0.70 * R_max * np.sin(angle)
                    (d,) = ax.plot([0, x_end], [0, y_end], color=ui_color, linewidth=1.5, zorder=10)
                    drawn_artists.append(d)
            elif is_transition:
                # Always draw the moving sweep line
                line_angle = start_angle - wipe_angle
                x_end_sweep, y_end_sweep = (
                    0.70 * R_max * np.cos(line_angle),
                    0.70 * R_max * np.sin(line_angle),
                )
                (d,) = ax.plot(
                    [0, x_end_sweep], [0, y_end_sweep], color=ui_color, linewidth=1.5, zorder=10
                )
                drawn_artists.append(d)

                # Also draw static dividers that the sweep line has passed
                for i in range(n_props):
                    divider_angle = (i * angle_step - np.pi) % (2 * np.pi) - np.pi

                    div_norm = (divider_angle + 2 * np.pi) % (2 * np.pi)
                    start_norm = (start_angle + 2 * np.pi) % (2 * np.pi)
                    end_norm = (end_angle + 2 * np.pi) % (2 * np.pi)

                    is_passed = (
                        (end_norm <= div_norm <= start_norm)
                        if start_norm >= end_norm
                        else ((div_norm >= end_norm) or (div_norm <= start_norm))
                    )

                    if is_passed:
                        x_end, y_end = (
                            0.70 * R_max * np.cos(divider_angle),
                            0.70 * R_max * np.sin(divider_angle),
                        )
                        (d,) = ax.plot(
                            [0, x_end],
                            [0, y_end],
                            color=ui_color,
                            linewidth=1.5,
                            zorder=10,
                            alpha=0.8,
                        )
                        drawn_artists.append(d)

        ax.set_xlim(-nx / 2, nx / 2)
        ax.set_ylim(-ny / 2, ny / 2)
        ax.set_aspect("equal")
        return drawn_artists

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        writer_obj = (
            PillowWriter(fps=fps)
            if writer == "pillow" or save_path.endswith(".gif")
            else FFMpegWriter(fps=fps, bitrate=5600)
        )
        anim.save(save_path, writer=writer_obj, dpi=300, savefig_kwargs={"bbox_inches": "tight"})
        print("Animation saved successfully.")

    plt.rcdefaults()
    return anim


if __name__ == "__main__":
    make_trilogy_rgb = False
    if make_trilogy_rgb:
        folder = "/raid/scratch/data/jwst/GLIMPSE/NIRCam/mosaic_1210_wispnathan_test/30mas"

        files = os.listdir(folder)
        bands = {}
        for file in files:
            result = re.search(r"[fF]\d{3,4}[mMwW]", file)
            # print(result)
            if result is not None:
                result = result.group(0)
                bands[result] = f"{folder}/{file}"

        blue_bands = [
            "F115W",
            "F150W",
        ]  # , 'F115W', 'F150W']
        green_bands = [
            "F200W",
            "F277W",
            "F356W",
        ]  # , 'F277W'] #['F162M', 'F182M','F200W', 'F210M', 'F250M', 'F277W']
        red_bands = [
            "F410M",
            "F444W",
            "F480M",
        ]  # ['F310M', 'F350M', 'F356W', 'F410M', 'F430M', 'F444W']

        file_dir = os.getcwd()
        print(file_dir)
        overwrite = True
        if not os.path.exists(f"{file_dir}/temp_RGB.png") or overwrite:
            with open(f"{file_dir}/trilogy.in", "w") as f:
                f.write("B\n")
                for band in blue_bands:
                    if band in bands.keys():
                        f.write(f"{bands[band]}[1]\n")
                f.write("\nG\n")
                for band in green_bands:
                    if band in bands.keys():
                        f.write(f"{bands[band]}[1]\n")
                f.write("\nR\n")
                for band in red_bands:
                    if band in bands.keys():
                        f.write(f"{bands[band]}[1]\n")
                f.write(f"""\nindir  /
                        outname  temp_RGB
                        outdir  {file_dir}
                        samplesize 20000
                        stampsize  2000
                        showstamps  0
                        satpercent  0.0001
                        noiselum    0.20
                        noisesig    50
                        noisesig0   0
                        combine      sum
                        colorsatfac  1.0
                        deletetests  1
                        correctbias 1
                        testfirst   0
                        sampledx  0
                        sampledy  0""")
            Trilogy(file_dir + "/trilogy.in", images=None).run()

    rgb = "/nvme/scratch/software/trilogy/JOF.png"
    reg_path = "/nvme/scratch/work/tharvey/catalogs/JOF_plot.reg"
    band = "/raid/scratch/data/jwst/JOF/NIRCam/mosaic_1084_wispnathan/jw04210-o001_t002_nircam_clear-f444w_i2dnobg.fits"

    data = np.flipud(plt.imread(rgb))
    wcs = WCS(fits.open(band)[1].header)
    fits_data = fits.open(band)[1].data
    reg = load_regions_from_reg(reg_path, wcs=wcs)

    fig, ax, inset_ax = create_plot_with_insets(
        data,
        wcs,
        reg,
        ax_pad_bottom=0,
        allowed_sides=["top", "bottom"],
        axis_type="pixel",
        add_compass=False,
    )

    fig.savefig("JOF_RGB_inset.pdf", dpi=300, bbox_inches="tight")
