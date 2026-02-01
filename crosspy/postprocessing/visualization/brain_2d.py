import numpy as np
import pandas as pd 

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import nibabel as nib

from PIL import Image

from datashader.bundling import hammer_bundle

from copy import deepcopy



def group_by_membership(membership: list, community_offset=0):
    """
        Create map community->node from list of communities membership
    :param membership: Community membership. Must be a list of community indices
    :param community_offset: offset for community index (held the same representation between different libraries)
    :return: None
    """
    community_to_node = defaultdict(list)

    for node, community in enumerate(membership):
        community_to_node[community - community_offset].append(node)

    return community_to_node

###############
# those two functions were copied from pycortex library ( https://github.com/gallantlab/pycortex/blob/main/cortex/freesurfer.py )
# because the library does not support windows but we use Windows server a lot 
###############
def _remove_disconnected_polys(polys):
    """Remove polygons that are not in the main connected component.
    
    This function creates a sparse graph based on edges in the input.
    Then it computes the connected components, and returns only the polygons
    that are in the largest component.
    
    This filtering is useful to remove disconnected vertices resulting from a
    poor surface cut.
    """
    n_points = np.max(polys) + 1
    import scipy.sparse as sp

    # create the sparse graph
    row = np.concatenate([
        polys[:, 0], polys[:, 1], polys[:, 0],
        polys[:, 2], polys[:, 1], polys[:, 2]
    ])
    col = np.concatenate([
        polys[:, 1], polys[:, 0], polys[:, 2],
        polys[:, 0], polys[:, 2], polys[:, 1]
    ])
    data = np.ones(len(col), dtype=bool)
    graph = sp.coo_matrix((data, (row, col)), shape=(n_points, n_points),
                          dtype=bool)
    
    # compute connected components
    n_components, labels = sp.csgraph.connected_components(graph)
    unique_labels, counts = np.unique(labels, return_counts=True)
    non_trivial_components = unique_labels[np.where(counts > 1)[0]]
    main_component = unique_labels[np.argmax(counts)]
    extra_components = non_trivial_components[non_trivial_components != main_component]

    # filter all components not in the largest component
    disconnected_pts = np.where(np.isin(labels, extra_components))[0]
    disconnected_polys_mask = np.isin(polys[:, 0], disconnected_pts)
    return polys[~disconnected_polys_mask]


def _move_disconnect_points_to_zero(pts, polys):
    """Change coordinates of points not in polygons to zero.
    
    This cleaning step is useful after _remove_disconnected_polys, to
    avoid using this points in boundaries computations (through pts.max(axis=0)
    here and there).
    """
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.unique(polys)] = True
    pts[~mask] = 0
    return pts

#############

def plot_distance_clusters(dist: np.array, partition: list, threshold=None):
    """
        Given distance matrix and partition of this matrix to clusters, plot it with channels grouped by communities
    :param dist: distance matrix. Should be square and symmetric.
    :param partition: list of community labels
    :param threshold: mask thresholding parameter
    :return: None
    """
    image = dist.copy()

    if not(threshold is None):
        image[image < threshold] = 0
        image[image >= threshold] = 1

    idx = sum(group_by_membership(partition).values(), [])
    clusters = np.array(partition)
    clusters = clusters[idx]

    fig = plt.figure(figsize=(10, 10))

    msize = 30

    ax1 = plt.subplot2grid((msize, msize), (0, 1), colspan=msize-1)
    ax2 = plt.subplot2grid((msize, msize), (1, 0), rowspan=msize-1)
    ax3 = plt.subplot2grid((msize, msize), (1, 1), rowspan=msize-1, colspan=msize-1)

    ax1.imshow(clusters.reshape((1, -1)), cmap='tab20', interpolation='nearest', aspect='auto')
    ax2.imshow(clusters.reshape((-1, 1)), cmap='tab20', interpolation='nearest', aspect='auto')
    im_ax = ax3.imshow(image[idx][:, idx], cmap='jet')

    cax = fig.add_axes([1.05, 0.025, 0.04, 0.925])

    plt.colorbar(im_ax, cax=cax, orientation='vertical')

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()


def create_gif(dynamic: list, out_fname: str, reorder=None, **kwargs):
    """
        Given list of distance matrix for time windows, create .gif animation from it
    :param dynamic: list of distance matrices
    :param out_fname: name of the .gif animation
    :param reorder: how to reorder channels. should be either None (keep order as it is) or list of indices equal to N.
                    Where N is number of channels.
    :param kwargs: parameters passed to Image.save of PIL.
    :return: None
    """
    if reorder is None:
        reorder = np.arange(dynamic[0].shape[0])

    def _create_image(mat):
        return Image.fromarray(np.array(mat[reorder][:, reorder]*255, dtype=np.uint8))

    cm = plt.get_cmap('jet')
    imgs = [_create_image(mat) for mat in cm(dynamic)]
    imgs[0].save(out_fname, save_all=True, append_images=imgs[1:], loop=0, **kwargs)


def get_rotation_matrix(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    return R

def rotate_coords(coords, degree):
    return np.dot(coords, get_rotation_matrix(degree))

def parse_surf(filename):
    import struct
    """
    """
    with open(filename, 'rb') as fp:
        #skip magic
        fp.seek(3)
        _ = fp.readline()
        fp.readline()
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.frombuffer(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.frombuffer(fp.read(4*3*faces), dtype='i4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def get_surf_data(surf_fname, patch_fname):
    from cortex.freesurfer import get_paths, parse_patch

    pts, polys = parse_surf(surf_fname)

    patch = parse_patch(patch_fname)
    verts = patch[patch['vert'] > 0]['vert'] - 1
    edges = -patch[patch['vert'] < 0]['vert'] - 1

    idx = np.zeros((len(pts),), dtype=bool)
    idx[verts] = True
    idx[edges] = True
    valid = idx[polys.ravel()].reshape(-1, 3).all(1)
    polys = polys[valid]
    idx = np.zeros((len(pts),))
    idx[verts] = 1
    idx[edges] = -1

    for i, x in enumerate(['x', 'y', 'z']):
        pts[verts, i] = patch[patch['vert'] > 0][x]
        pts[edges, i] = patch[patch['vert'] < 0][x]
        
    return pts, polys, idx

def get_triangle_labels(labels, triangles, edge_value=-1):
    res = np.zeros(shape=len(triangles), dtype=labels.dtype)
    
    for idx, coords in enumerate(triangles):
        triangle_labels = labels[coords]
        if len(set(triangle_labels)) == 1:
            res[idx] = triangle_labels[0]
        else:
            res[idx] = edge_value
    
    return res

def create_hb_df(surf, connectome, threshold):
    parc_coords = np.concatenate([surf.annotations['lh']['coords'][1:], surf.annotations['rh']['coords'][1:]])
    
    coords_df = pd.DataFrame(parc_coords, columns=['x', 'y'])
    coords_df['name'] = surf.annotations['lh']['parcel_names'][1:] + surf.annotations['rh']['parcel_names'][1:]
    
    edges_df = {'source': [], 'target': [], 'weight': []}
    for i, j in zip(*np.triu_indices_from(connectome, 1)):
        value = connectome[i,j]
        
        if value >= threshold:
            edges_df['source'].append(i)
            edges_df['target'].append(j)
            edges_df['weight'].append(value)
        
    edges_df = pd.DataFrame(edges_df)
    
    return coords_df, edges_df


class FlatSurface:
    def __init__(self, subject_path, parcellation='Schaefer2018_100Parcels_17Networks'):
        self.subject_path = subject_path
        self.parcellation = parcellation
        
        self._load_hemis()
        self._load_annotations()
        
    def _load_hemis(self):
        self.surfaces = dict()
        
        for hemi in ['lh', 'rh']:
            wm_path = os.path.join(self.subject_path, 'surf', f'{hemi}.smoothwm')
            patch_path = os.path.join(self.subject_path, 'surf', f'{hemi}.cortex.patch.flat')

            hemi_coords, hemi_faces, _ = get_surf_data(wm_path, patch_path)
            
            hemi_coords = hemi_coords[:, [1, 0, 2]]
            # Flip Y axis upside down
            hemi_coords[:, 1] = -hemi_coords[:, 1]
            
            hemi_faces = _remove_disconnected_polys(hemi_faces)
            hemi_coords = _move_disconnect_points_to_zero(hemi_coords, hemi_faces)[:, :2]
            
            degree = 90 if hemi == 'lh'  else -90
            hemi_coords = rotate_coords(hemi_coords, degree)

            if hemi == 'rh':
                hemi_coords[:, 0] += self.surfaces['lh']['coords'][:,0].max() - hemi_coords[:,0].min() + 10
            
            self.surfaces[hemi] = {'coords': hemi_coords, 'faces': hemi_faces}
            
    def _load_annotations(self):
        self.annotations = dict()
        self.parcel_coords = dict()
        
        self.n_parcels = 0
        
        for hemi in ['lh', 'rh']:
            annot_path = os.path.join(self.subject_path, 'label', f'{hemi}.{self.parcellation}_order.annot')
            
            labels_orig, _, annot_ch_names = nib.freesurfer.io.read_annot(annot_path)   
            annot_ch_names = [n.decode() for n in annot_ch_names]
            
            labels_faces = get_triangle_labels(labels_orig, self.surfaces[hemi]['faces'])
            
            parcel_coords = np.zeros((len(set(labels_orig)), 2))
            for label in set(labels_orig):
                label_indices = (labels_orig == label)
                parcel_coords[label] = np.median(self.surfaces[hemi]['coords'][label_indices], axis=0)
            
            self.annotations[hemi] = {'vertex_labels': labels_orig, 'face_labels': labels_faces,
                                      'parcel_names': annot_ch_names, 'coords': parcel_coords}
            
            self.n_parcels += len(annot_ch_names)
            
        self.connectome = None
        self.data = dict()
                        
    def add_connectome(self, connectome):
        if connectome.shape[0] != self.n_parcels - 2: # -2 unknown
            raise RuntimeError(f'Amount of connectome parcels ({connectome.shape[0]}) is not equal to data ({self.n_parcels})')
        
        if connectome.shape[0] != connectome.shape[1]:
            raise RuntimeError(f'Connectome is not square! Shape: {connectome.shape}')
        
        self.connectome = connectome.copy()
        
    def add_data(self, data):
        # data should be a dict with mapping parcel_name -> value
        self.data = deepcopy(data)
    
    def plot(self, data_cmap='viridis', connectome_cmap='jet', connectome_threshold=0.0, ax=None, 
             use_norm=True, draw_colorbar=False, alpha=1):
        if (ax is None):
            fig, ax_plot = plt.subplots(figsize=(4*3, 3*3))
        else:
            ax_plot = ax
        
        self._plot_data(cmap=data_cmap, ax=ax_plot, use_norm=use_norm, draw_colorbar=draw_colorbar, alpha=alpha)
        
        if not(self.connectome is None):
            self._plot_connectome(cmap=connectome_cmap, ax=ax_plot, threshold=connectome_threshold)
        
        ax_plot.set_axis_off()
    
    def _plot_connectome(self, cmap, ax, threshold):
        fig = ax.get_figure()
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap

        ds_nodes, ds_edges = create_hb_df(self, self.connectome, threshold=threshold)
        hb = hammer_bundle(ds_nodes, ds_edges, initial_bandwidth=0.125, decay=0.125)
        
        segments = np.array_split(hb.values, np.where(np.isnan(hb.values[:,0]))[0])
        
        norm = mpl.colors.Normalize(ds_edges['weight'].min(), ds_edges['weight'].max())
        
        for s in segments[:-1]:
            width = s[1,2]
            ax.plot(s[:,0], s[:,1], color=cmap_obj(norm(width)), lw=width*5)
            
        pos =  ax.get_position()
        cbar_ax = fig.add_axes([pos.x1, pos.y0 + pos.height*3/4, 0.01, pos.height*1/4])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_obj, orientation='vertical', norm=norm)

        
    def _plot_data(self, cmap, ax, draw_colorbar=False, use_norm=True, alpha=1.0):
        fig = ax.get_figure()
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        min_value = np.nanmin(list(self.data.values()))
        max_value = np.nanmax(list(self.data.values()))
        norm = mpl.colors.Normalize(min_value, max_value)
        
        for hemi in ['lh', 'rh']:
            colors_to_use = [(0, 0, 0)]
            
            face_data = np.zeros_like(self.annotations[hemi]['face_labels'], dtype=float)
            face_mask = (self.annotations[hemi]['face_labels'] != -1)
            face_data[~face_mask] = len(colors_to_use)

            for parcel_idx, parcel_name in enumerate(self.annotations[hemi]['parcel_names']):
                parcel_value = self.data.get(parcel_name)

                if parcel_value is None:
                    continue
                
                if use_norm:
                    parcel_value = norm(parcel_value)

                colors_to_use.append(cmap_obj(parcel_value))
                
                face_indices = (self.annotations[hemi]['face_labels'] == parcel_idx)
                face_data[face_indices] = len(colors_to_use)
                face_mask[face_indices] = False                
            
            plot_cmap = mpl.colors.ListedColormap(colors_to_use)

            ax.tripcolor(*self.surfaces[hemi]['coords'].T, self.surfaces[hemi]['faces'], 
                 facecolors=face_data, mask=face_mask, cmap=plot_cmap, alpha=alpha)  
        
        if len(self.data) > 0 and draw_colorbar:
            self._plot_colorbar(ax, min_value, max_value, cmap)
            
    def _plot_colorbar(self, ax, vmin, vmax, cmap):
        fig = ax.get_figure()
        norm = mpl.colors.Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap) if (type(cmap) is str) else cmap
        
        pos =  ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 + pos.height*3/4, 0.01, pos.height*1/4])
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_obj, orientation='vertical', norm=norm)

        return cbar_ax
