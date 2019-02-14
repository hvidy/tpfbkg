import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

import lightkurve 
from lightkurve import TessTargetPixelFile, TessLightCurveFile, TessLightCurve
from lightkurve.utils import plot_image
from lightkurve import MPLSTYLE

def plot_bkg(tpf,ax=None, frame=0, cadenceno=None, aperture_mask=None,
         show_colorbar=True, mask_color='pink', style='lightkurve', **kwargs):
    """Plot the pixel data for a single frame (i.e. at a single time).
    The time can be specified by frame index number (`frame=0` will show the
    first frame) or absolute cadence number (`cadenceno`).
    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib axes object to plot into. If no axes is provided,
        a new one will be generated.
    frame : int
        Frame number. The default is 0, i.e. the first frame.
    cadenceno : int, optional
        Alternatively, a cadence number can be provided.
        This argument has priority over frame number.
    bkg : bool
        If True, background will be added to the pixel values.
    aperture_mask : ndarray or str
        Highlight pixels selected by aperture_mask.
    show_colorbar : bool
        Whether or not to show the colorbar
    mask_color : str
        Color to show the aperture mask
    style : str
        Path or URL to a matplotlib style file, or name of one of
        matplotlib's built-in stylesheets (e.g. 'ggplot').
        Lightkurve's custom stylesheet is used by default.
    kwargs : dict
        Keywords arguments passed to `lightkurve.utils.plot_image`.
    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    if style == 'lightkurve' or style is None:
        style = MPLSTYLE
    if cadenceno is not None:
        try:
            frame = np.argwhere(cadenceno == tpf.cadenceno)[0][0]
        except IndexError:
            raise ValueError("cadenceno {} is out of bounds, "
                             "must be in the range {}-{}.".format(
                                 cadenceno, tpf.cadenceno[0], tpf.cadenceno[-1]))
    try:
        if np.any(np.isfinite(tpf.flux_bkg[frame])):
            pflux = tpf.flux_bkg[frame]
    except IndexError:
        raise ValueError("frame {} is out of bounds, must be in the range "
                         "0-{}.".format(frame, tpf.shape[0]))
    with plt.style.context(style):
        img_title = 'Target ID: {}'.format(tpf.targetid)
        img_extent = (tpf.column, tpf.column + tpf.shape[2],
                      tpf.row, tpf.row + tpf.shape[1])
        ax = plot_image(pflux, ax=ax, title=img_title, extent=img_extent,
                        show_colorbar=show_colorbar, **kwargs)
        ax.grid(False)
    if aperture_mask is not None:
        aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        for i in range(tpf.shape[1]):
            for j in range(tpf.shape[2]):
                if aperture_mask[i, j]:
                    ax.add_patch(patches.Rectangle((j+tpf.column, i+tpf.row),
                                                   1, 1, color=mask_color, fill=True,
                                                   alpha=.6))
    return ax

def plot_ffi_bkg(hdf,ax=None, frame=0, cadenceno=None, cut=None,
         show_colorbar=True, style='lightkurve', **kwargs):
    """Plot the pixel data for a single frame (i.e. at a single time).
    The time can be specified by frame index number (`frame=0` will show the
    first frame) or absolute cadence number (`cadenceno`).
    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib axes object to plot into. If no axes is provided,
        a new one will be generated.
    frame : int
        Frame number. The default is 0, i.e. the first frame.
    cadenceno : int, optional
        Alternatively, a cadence number can be provided.
        This argument has priority over frame number.
    bkg : bool
        If True, background will be added to the pixel values.
    aperture_mask : ndarray or str
        Highlight pixels selected by aperture_mask.
    show_colorbar : bool
        Whether or not to show the colorbar
    mask_color : str
        Color to show the aperture mask
    style : str
        Path or URL to a matplotlib style file, or name of one of
        matplotlib's built-in stylesheets (e.g. 'ggplot').
        Lightkurve's custom stylesheet is used by default.
    kwargs : dict
        Keywords arguments passed to `lightkurve.utils.plot_image`.
    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    if style == 'lightkurve' or style is None:
        style = MPLSTYLE
    if cadenceno is not None:
        try:
            frame = np.argwhere(cadenceno == hdf["cadenceno"][()])[0][0]
        except IndexError:
            raise ValueError("cadenceno {} is out of bounds, "
                             "must be in the range {}-{}.".format(
                                 cadenceno, hdf['cadenceno'][0], hdf['cadenceno'][-1]))
    frame = "{:04d}".format(frame)
    try:
        if cut is not None:
            if np.any(np.isfinite(hdf["backgrounds"][frame][()])):
                try:
                    pflux = hdf["backgrounds"][frame][()][cut[0]:cut[1],cut[2]:cut[3]]
                except IndexError:
                    raise ValueError("cut {} is out of bounds, must be in the"
                                     "range 0-{}".format(cut,hdf['backgrounds'][frame][()].shape))
        else:
            if np.any(np.isfinite(hdf["backgrounds"][frame][()])):
                pflux = hdf["backgrounds"][frame][()]
    except IndexError:
        raise ValueError("frame {} is out of bounds, must be in the range "
                         "0-{}.".format(frame, len(hdf["backgrounds"])))
    with plt.style.context(style):
        img_title = 'FFI background'
        if cut is not None:
            img_extent = cut
        else:
            img_extent = (0, hdf['backgrounds'][frame][()].shape[1],
                          0, hdf['backgrounds'][frame][()].shape[0])
        ax = plot_image(pflux, ax=ax, title=img_title, extent=img_extent,
                        show_colorbar=show_colorbar, **kwargs)
        ax.grid(False)
    return ax

def plot_new(tpf, newbkg, ax=None, frame=0, cadenceno=None, bkg=False, aperture_mask=None,
     show_colorbar=True, mask_color='pink', style='lightkurve', **kwargs):
    """Plot the pixel data for a single frame (i.e. at a single time).
    The time can be specified by frame index number (`frame=0` will show the
    first frame) or absolute cadence number (`cadenceno`).
    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib axes object to plot into. If no axes is provided,
        a new one will be generated.
    frame : int
        Frame number. The default is 0, i.e. the first frame.
    cadenceno : int, optional
        Alternatively, a cadence number can be provided.
        This argument has priority over frame number.
    bkg : bool
        If True, background will be added to the pixel values.
    aperture_mask : ndarray or str
        Highlight pixels selected by aperture_mask.
    show_colorbar : bool
        Whether or not to show the colorbar
    mask_color : str
        Color to show the aperture mask
    style : str
        Path or URL to a matplotlib style file, or name of one of
        matplotlib's built-in stylesheets (e.g. 'ggplot').
        Lightkurve's custom stylesheet is used by default.
    kwargs : dict
        Keywords arguments passed to `lightkurve.utils.plot_image`.
    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    if style == 'lightkurve' or style is None:
        style = MPLSTYLE
    if cadenceno is not None:
        try:
            frame = np.argwhere(cadenceno == tpf.cadenceno)[0][0]
        except IndexError:
            raise ValueError("cadenceno {} is out of bounds, "
                             "must be in the range {}-{}.".format(
                                 cadenceno, tpf.cadenceno[0], tpf.cadenceno[-1]))
    try:
        if bkg and np.any(np.isfinite(tpf.flux_bkg[frame])):
            pflux = tpf.flux[frame] + tpf.flux_bkg[frame]
        else:
            pflux = tpf.flux[frame] + tpf.flux_bkg[frame] - newbkg[frame]
    except IndexError:
        raise ValueError("frame {} is out of bounds, must be in the range "
                         "0-{}.".format(frame, tpf.shape[0]))
    with plt.style.context(style):
        img_title = 'Target ID: {}'.format(tpf.targetid)
        img_extent = (tpf.column, tpf.column + tpf.shape[2],
                      tpf.row, tpf.row + tpf.shape[1])
        ax = plot_image(pflux, ax=ax, title=img_title, extent=img_extent,
                        show_colorbar=show_colorbar, **kwargs)
        ax.grid(False)
    if aperture_mask is not None:
        aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        for i in range(tpf.shape[1]):
            for j in range(tpf.shape[2]):
                if aperture_mask[i, j]:
                    ax.add_patch(patches.Rectangle((j+tpf.column, i+tpf.row),
                                                   1, 1, color=mask_color, fill=True,
                                                   alpha=.6))
    return ax