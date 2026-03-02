import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

def plot_with_colors(ax, x, y_values, color_values, cmap='jet'):
    fig = ax.figure
    
    vmin = color_values.min()
    vmax = color_values.max()
    
    cmap_obj = plt.get_cmap(cmap)
    
    for c, y in zip(color_values, y_values):
        color = (c - vmin)/(vmax - vmin)
        ax.plot(x, y, color=cmap_obj(color))
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax)
    
def _create_colors(data, vmin, vmax, cmap='RdBu_r'):
    norm_obj = mpl.colors.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)
    
    res = {k:cmap_obj(norm_obj(v)) for (k,v) in data.items() if np.isfinite(v)}
    
    return res

def draw_four_views(axes, data, parcel_names, surfaces, n_parcels=400, cmap='viridis', norm=None, title=None, norm_values=None, 
                   title_kwargs=None, cbar_loc='left', cbar_ax_kwargs=None, significant_parcels=None, border_color='white'):
    
    if cbar_ax_kwargs is None:
        cbar_ax_kwargs = dict()
        
    if title_kwargs is None:
        title_kwargs = dict()
    
    if not('fontsize' in title_kwargs):
        title_kwargs['fontsize'] = 7

    slices = [slice(0,n_parcels//2), slice(n_parcels//2,n_parcels)]
    
    if (norm_values is None):
        vmin, vmax =  np.nanpercentile(data, (1,99))
    else:
        vmin, vmax = norm_values
    
    plot_data = {n:v for (n,v) in zip(parcel_names, data)}

    if any([type(v) is tuple for v in data]):
        plot_data_colors = plot_data
    else:
        plot_data_colors = _create_colors(plot_data, vmin, vmax, cmap=cmap)
                   
                   
    for hemi_idx, (bs, parcel_slice) in enumerate(zip(surfaces, slices)):
        # bs.set_data(plot_data_colors, significant_parcels)
        bs.set_data(plot_data_colors)

        zoom = 1.6
        
        for view_idx in range(2):
            camera_pos = (-1,0,0) if (hemi_idx == view_idx) else (1,0,0)
        
            bs.plot(show=False, camera_position=camera_pos, zoom=zoom, cmap=cmap, lightning='three lights', 
                    vmin=vmin, vmax=vmax, norm=norm, border_color=border_color)

            img = bs.plotter.screenshot(return_img=True)
            axes[view_idx, hemi_idx].imshow(img)
            axes[view_idx, hemi_idx].set_axis_off()
    
    if norm is None:
        cbar_norm = mpl.colors.Normalize(vmin, vmax)
    elif norm == 'log_discrete':
        bounds = np.geomspace(vmin, vmax, 10)
        cbar_norm = mpl.colors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)
    elif norm == 'log':
        cbar_norm = mpl.colors.LogNorm(vmin, vmax)
    
    if cbar_loc == 'left':
        if not('bbox_to_anchor' in cbar_ax_kwargs):
            cbar_ax_kwargs['bbox_to_anchor'] = (-0.15,0.35,0.7,1.35)
        
        if not('height' in cbar_ax_kwargs):
            cbar_ax_kwargs['height'] = '60%'

        cbar_ax = inset_axes(axes[1,0], width="10%", loc=2, bbox_transform=axes[1,0].transAxes, **cbar_ax_kwargs) 
        orientation = 'vertical'
    elif cbar_loc == 'bottom':
        if not('bbox_to_anchor' in cbar_ax_kwargs):
            cbar_ax_kwargs['bbox_to_anchor'] = (0.3,-0.2,1.3,0.8)
            
        cbar_ax = inset_axes(axes[1,0], width="40%", height="10%", loc=8, bbox_transform=axes[1,0].transAxes, **cbar_ax_kwargs) 
        orientation = 'horizontal'
    
    
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=plt.get_cmap(cmap), orientation=orientation, norm=cbar_norm)
    cbar_ax.yaxis.set_ticks_position('left')
    
    if  norm == 'log_discrete':
        cbar.set_ticks(cbar.get_ticks()[::3])
    elif norm == 'log':
        cbar.set_ticks(np.geomspace(vmin, vmax, 4))
       
    cbar.outline.set_linewidth(0.25)
    axes[0,0].set_title(title,  x=1.15, **title_kwargs)