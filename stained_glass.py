import cv2

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

import seaborn as sns

from scipy import ndimage, misc

sns.set(color_codes=True)

from image_utils import harris_response, response_to_points, non_maximal_suppression

from scipy.spatial import Delaunay

from skimage import feature
from skimage.filters.rank import autolevel_percentile
from skimage.filters.rank import autolevel
from skimage.morphology import disk

from tqdm import tqdm

def unique_rows(a):
    """
    Select the unique rows of a numpy array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def gen_random_border_points(max_n, min_dist, rng):
    """
    Select random points on the border that are more or less evenly spaced, 
    including the corners

    Args:
        max_n (int): The end of the interval
        min_dist (int): minimal separation between points
        rng (numpy.random.RandomState) : numpy random number generator

    Returns:
        border (numpy.ndarray) : Array of border points
    """
    candids = np.arange(min_dist, max_n - min_dist)
    border = [0, max_n]
    
    while True:
        try:
            ind = rng.choice(candids)
        except ValueError:
            border = np.asarray(border)
            border = np.sort(border)
            return border
        
        border.append(ind)
        candids = candids[np.logical_or(candids < ind - min_dist,
                                        candids > ind + min_dist)]


def fill_triangle(tr, art_img, new_img):
    """
    Fill a specified triangular region with the average color in that region

    Args:
        tr (numpy.ndarray): Triangle coordinates, (3,2)
        art_img (numpy.ndarray): The original RGB image, (width, height, 3)
        new_img (numpy.ndarray): Destination to save colored triangle
    """
    r = cv2.boundingRect(np.float32([tr]))
    t_rec = []
    for i in xrange(0, 3):
        t_rec.append(((tr[i][0] - r[0]),(tr[i][1] - r[1])))
    
    
    mask = np.zeros((r[3], r[2]), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rec), (1.0, 1.0, 1.0), 16, 0);
    art_img_patch = art_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]

    kx, ky = np.where(mask == 1)
    vals = np.mean(art_img_patch[kx,ky,:], axis=0).astype('int')
    
    #print art_img_patch.shape
    new_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]][kx,ky,:] = vals[None,None,:] #* (mask[:,:,None])
    #art_img_patch * (mask[:,:,None] ) + 

def gen_all_filters(art):
    """
    Generate all the filtered versions of the input image, for use in other functions

    Args:
        art (numpy.ndarray): The original RGB image, (width, height, 3)

    Returns:
        im_smooth (numpy.ndarray): Smoothed grayscale image (width, height)
        smooth_grad_mag (numpy.ndarray): Gradient magnitude of smoothed, grayscale image (width, height)
        loc_autolevel (numpy.ndarray): Rank-filtered grayscale image (width, height)
        sharp_grad_mag (numpy.ndarray): Gradient magnitude of rank-filtered, grayscale image (width, height)
    """
    #Convert to grayscale for edge estimation
    hsv_art = mpl.colors.rgb_to_hsv(art).astype(np.float32)

    #The smooth image
    im_smooth = np.zeros(hsv_art[:,:,0].shape)
    ndimage.filters.gaussian_filter(hsv_art[:,:,2], (10,10), (0,0), im_smooth)

    #Smooth grad image, with thresholding
    im_x = np.zeros(im_smooth.shape)
    sigma = 2
    ndimage.filters.gaussian_filter(im_smooth, (sigma,sigma), (0,1), im_x)

    im_y = np.zeros(im_smooth.shape)
    ndimage.filters.gaussian_filter(im_smooth, (sigma,sigma), (1,0), im_y)
    smooth_grad_mag = np.sqrt(im_y**2 + im_x**2)
    
    #Sharp im
    selem = disk(20)
    loc_autolevel = autolevel(hsv_art[:,:,2] / 256.0, selem=selem)

    #Sharp im grad
    im_x = np.zeros(loc_autolevel.shape)
    sigma = 2
    ndimage.filters.gaussian_filter(loc_autolevel, (sigma,sigma), (0,1), im_x)

    im_y = np.zeros(loc_autolevel.shape)
    ndimage.filters.gaussian_filter(loc_autolevel, (sigma,sigma), (1,0), im_y)
    sharp_grad_mag = np.sqrt(im_y**2 + im_x**2)

    return im_smooth, smooth_grad_mag, loc_autolevel, sharp_grad_mag

        
def edge_points_width_bounding_box(smooth_im, sharp_im, inds, min_dist):
    """
    Obtain critical points on the image and boundary

    Args:
        smooth_im (numpy.ndarray): Smoothed, grayscale version of the image (width, height)
        sharp_im (numpy.ndarray): Higher-contrast, grayscale version of the image (width, height)
        inds (numpy.ndarray): (x,y) coordinates of a given flat index (n_samples, 2), int
        min_dist (int) : minimum distance between points

    Returns:
        points (numpy.ndarray): The (x,y) coordinates of the critical points, (n_points, 2)
        p_edges (numpy.ndarray): Fuzzy boundary map around critical points in image (width, height)
    """
    edges = feature.canny(smooth_im, sigma=1)
    edges2 = feature.canny(sharp_im, sigma=5)
    p_edges = edges2 - edges

    edge_points = np.nonzero(p_edges.flatten())[0]
    edge_points = np.asarray([inds[pt] for pt in edge_points]).tolist()
    values = np.asarray([sharp_im[p[0],p[1]] for p in edge_points])
    index = np.argsort(values)

    # # select the best points taking min_distance into account
    filtered_coords = non_maximal_suppression(index, edge_points, 
        smooth_im.shape, min_dist).tolist()

    rng = np.random.RandomState(223245)
    x_range = gen_random_border_points(smooth_im.shape[0] - 1, min_dist, rng)
    y_range = gen_random_border_points(smooth_im.shape[1] - 1, min_dist, rng)

    lft_row = [[0, yi] for yi in y_range]
    filtered_coords.extend(lft_row)

    rht_row = [[x_range[-1], yi] for yi in y_range]
    filtered_coords.extend(rht_row)

    top_row = [[xi, 0] for xi in x_range]
    filtered_coords.extend(top_row)

    bot_row = [[xi, y_range[-1]] for xi in x_range]
    filtered_coords.extend(bot_row)

    points = np.array(filtered_coords)
    points = unique_rows(points)
    points = points[:,::-1]

    return points, p_edges


def stained_glass_transform(art, points):
    """
    Use Delaunay tesselation to create a crystalline average of an image

    Args:
        art (numpy.ndarray): The original RGB image, (width, height, 3)
        points (numpy.ndarray): The (x,y) coordinates of the critical points, (n_points, 2)

    Returns
        new_image (numpy.ndarray): The transformed RGB image (width, height, 3)
    """
    tri = Delaunay(points)

    new_image = np.zeros(art.shape)

    for i, region in tqdm(enumerate(tri.simplices), desc='regions'):
        vertices = points[region] 
        fill_triangle(vertices, 256 - art, new_image)

    return new_image

def add_contours(art, inds, sharp_grad, smooth_grad):
    """
    Increase stained-glass effect by adding a frame to the image

    Args:
        art (numpy.ndarray): The original RGB image, (width, height, 3)
        inds (numpy.ndarray): (x,y) coordinates of a given flat index (n_samples, 2), int
        smooth_grad (numpy.ndarray): Gradient magnitude of smoothed grayscale image (width, height)
        sharp_grad (numpy.ndarray): Gradient magnitude of rank-filtered grayscale image (width, height)

    Returns:
        smooth_edge_points (numpy.ndarray): Location of the thin frame (n_pts, 2)
        cast_edge_points (numpy.ndarray): Location of the thick frame (n_pts, 2)
    """

    #Smooth edge contours
    smooth_grad_thresh = smooth_grad.copy()
    smooth_grad_thresh[smooth_grad_thresh < 1.2] = 0
    smooth_edge_points = np.nonzero(smooth_grad_thresh.flatten())[0]
    smooth_edge_points = inds[smooth_edge_points]

    #Delta thresh
    Ps_thresh = (sharp_grad - smooth_grad)
    Ps_thresh[Ps_thresh < 20] = 0

    cast_edge_points = np.nonzero(Ps_thresh.flatten())[0]
    cast_edge_points = inds[cast_edge_points]

    return smooth_edge_points, cast_edge_points

def gen_image(art, stained_glass, cast_edge_points, smooth_edge_points, fname):
    """
    Generate and save the stained glass transform

    Args:
        art (numpy.ndarray): The original RGB image, (width, height, 3)
        stained_glass (numpy.ndarray): The Delaunay transformed RGB image (width, height, 3)
        cast_edge_points (numpy.ndarray): Location of the thick frame (n_pts, 2)
        smooth_edge_points (numpy.ndarray): Location of the thin frame (n_pts, 2)
        fname (str) : Directory to save image

    """
    fig = plt.figure(figsize=(30,30))
    axs = fig.add_subplot(111)

    axs.imshow(art, alpha=0.2)
    axs.imshow(stained_glass)

    axs.plot(smooth_edge_points[:,0], smooth_edge_points[:,1], 'ko', ms=3)

    axs.plot(cast_edge_points[:,0], cast_edge_points[:,1], 'ko', ms=2)

    axs.set_xlim((0, stained_glass.shape[1]))
    axs.set_ylim((stained_glass.shape[0],0))

    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def main(args):
    art = misc.imread(args.art)

    x = gen_all_filters(art)
    smooth_im, smooth_grad, sharp_im, sharp_grad = x

    x_range = np.arange(art.shape[0])
    y_range = np.arange(art.shape[1])
    xi, yi = np.meshgrid(x_range,y_range)
    inds = np.array([xi.ravel(), yi.ravel()]).T

    points, p_edges = edge_points_width_bounding_box(smooth_im, sharp_im, inds, args.min_dist)
    stained_glass = stained_glass_transform(art, points)
    stained_glass[p_edges] = 0

    x_range = np.arange(art.shape[1])
    y_range = np.arange(art.shape[0])
    xi, yi = np.meshgrid(x_range,y_range)
    inds = np.array([xi.ravel(), yi.ravel()]).T
    smooth_edge_points, cast_edge_points = add_contours(art, inds, sharp_grad, smooth_grad)


    gen_image(art, stained_glass, cast_edge_points, smooth_edge_points, args.output)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a stained glass transform of an image")
    parser.add_argument('--art', dest='art', type=str, help='Input image to transform')
    parser.add_argument('--output', dest='output', type=str, help='Path to save file')
    parser.add_argument('--min_dist', dest='min_dist', type=int, help='Min distance for non maximal suppression. Higher values mean higher crystallinity of the transform.')
    parser.set_defaults(min_dist=35, output='stained_glass_transform.png')

    args = parser.parse_args()

    main(args)


