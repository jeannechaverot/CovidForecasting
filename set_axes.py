import matplotlib.pyplot as plt

def set_ax(ax,
           fs=16, 
           fs_ticks=None, 
           fs_title=None,
           fs_legend=None,
           xlabel=None,
           ylabel=None,
           title=None,
           xticks=None,
           yticks=None,
           xticklabels=None,
           yticklabels=None,
           legend=False,
           axis=None,
           colorbar=False,
           grid=True,
           grid_alpha=0.8,
           tick_direction='in'):
    '''
    Method that takes an instance of matplotlib.Axes and
    sets the required parameters. 
    --------------
    Parameters:
        ax: instance of matplotlib.Axes
            the object on which the rest of the
            parameters will be set
        fs: int, optional
            fontsize used for xlabel and ylabel.
            Defaults to 16
        fs_ticks: int, optional
            fontsize used for the ticks of the plot.
            If None, (fs-2) is used
        fs_title: int, optional
            fontsize used for the title.
            If None, (fs + 1) is used
        fs_legend: int, optional
            fontsize used for the legend, if present.
            If None, fs_ticks is used.
        xlabel: str, optional
            the xlabel of the plot. 
            If None, no xlabel will be set.
        ylabel: str, optional
            the ylabel of the plot.
            If None, no ylabel will be set.
        title: str, optional
            the title of the plot.
            If None, no title will be set.
        xticks: list, optional
            the position of the xticks.
            If None, the default ticks will be used.
        yticks: list, optional
            the position of the yticks.
            If None, the default ticks will be used.
        xtickslabels: list of str, optional
            the labels used in correspondence to xticks.
            If None, the default labels will be used.
        ytickslabels: list of str, optional
            the labels used in correspondence to yticks.
            If None, the default labels will be used.
        legend: bool, optional
            if True, a legend will be set
        axis: [xmin, xmax, ymin, ymax], optional
            the scale shown in the plot.
            If None, the default one is used.
        colorbar: boolean, optional
            if True, a colorbar will be included and its
            label font size will be set equal to ticks_size.
            Defaults to False
        grid: boolean, optional
            whether to set the grid on or off.
            Defaults to True
        grid_alpha: float, optional
            considered when grid = True. 
            Defaults to 0.8
        tick_direction: str, optional
            direction with respect to the plot where
            the ticks appear. Options are 'in', 'out' and 'inout'.
            Defaults to 'in'

    ----------------
    '''

    if fs_ticks is None:
        fs_ticks = fs
    if fs_title is None:
        fs_title = fs + 2
    if fs_legend is None:
        fs_legend = fs_ticks
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fs)
    if title is not None:
        ax.set_title(title,fontsize=fs_title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is None:        
        ax.xaxis.set_tick_params(labelsize=fs_ticks, which='both')
    else:
        ax.set_xticklabels(xtickslabels,fontdict = {'fontsize':fs_ticks})
    if yticklabels is None:
        ax.yaxis.set_tick_params(labelsize=fs_ticks, which='both')
    else:
        ax.set_yticklabels(ytickslabels,fontdict = {'fontsize':fs_ticks})
    if legend:
        ax.legend(fontsize=fs_legend)
    if axis is not None:
        ax.axis(axis)
    if colorbar:
        clb = plt.colorbar()
        clb.ax.tick_params(labelsize=fs_ticks)
    ax.grid(b=grid)
    ax.tick_params(which='both', direction=tick_direction, bottom=True,
                   top=True, left=True, right=True)
    if grid:
        try:
            ax.tick_params(grid_alpha=grid_alpha)
        except:
            pass
    return ax

