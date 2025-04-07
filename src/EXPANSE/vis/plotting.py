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
