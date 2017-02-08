import numpy as np

from scipy.ndimage import filters

def harris_response(im, sigma=3):
    """
    Find corners using the determinant of the autocorrelation function

    Args:
        im (numpy.ndarray) : grayscale im (width, height)

    Kwargs:
        sigma (float) : width of gaussian filter used for derivative estimation

    Returns:
        (numpy.ndarray) : Harris response matrix (width, height)
    """

    #compute derivatives
    im_x = np.zeros(im.shape)
    im_x = filters.gaussian_filter(im, (sigma,sigma), (0,1), im_x)
    
    im_y = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), im_y)
    
    #Harris matrix
    W_xx = filters.gaussian_filter(im_x * im_x,sigma)
    W_xy = filters.gaussian_filter(im_x * im_y,sigma)
    W_yy = filters.gaussian_filter(im_y * im_y,sigma)
    
    Wdet = W_xx * W_yy - W_xy ** 2
    Wtr = W_xx + W_yy
    
    return Wdet / Wtr

def non_maximal_suppression(index, points, shape, min_dist=10):
    """
    Filter points based on the strength of their activation and their relative distances

    Args:
        index (numpy.ndarray): (x,y) coordinates of a given flat index (n_samples, 2), int
        points (numpy.ndarray): (x,y) coordinates of the candidate points (n_points, 2), int
        shape (tuple): shape of the array in which the points reside

    Kwargs:
        min_dist (int) : minimum distance between points

    Returns:
        filtered_coords (numpy.ndarray) : (x,y) coordinates of the selected points (n_points, 2), int
    """

    # store allowed point locations in array
    allowed = np.zeros(shape)
    allowed[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        x,y = points[i]
        if allowed[x,y]:
            filtered_coords.append(points[i])
            allowed[(x - min_dist):(x + min_dist),
                    (y - min_dist):(y + min_dist)] = 0
            
    return np.asarray(filtered_coords)

def response_to_points(harris, min_dist=10, threshold=0.1):
    """
    Select harris response points via non maximal suppression

    Args:
        harris (numpy.ndarray): the harris response matrix (width, height)

    Kwags:
        min_dist (int) : minimum distance between points
        threshold (float) : fraction of max, below which corner candidates are rejected (0-1)

    Returns:
        (numpy.ndarray) : Corner points, (n_points, 2)
    """
    corner_thresh = np.max(harris) * threshold
    
    # Obtain candidate points and their harris values
    points = np.asarray(np.where(harris > corner_thresh)).T
    values = np.asarray([harris[p[0],p[1]] for p in points])
    index = np.argsort(values)
    
    return non_maximal_suppression(index, points, shape, min_dist)







