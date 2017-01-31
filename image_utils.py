import numpy as np

from scipy.ndimage import filters

def harrisResponse(im, sigma=3):
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

def nonMaximalSuppression(index, points, shape, min_dist=10):
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

def response2points(harris, min_dist=10, threshold=0.1):
    corner_thresh = np.max(harris) * threshold
    
    # Obtain candidate points and their harris values
    points = np.asarray(np.where(harris > corner_thresh)).T
    values = np.asarray([harris[p[0],p[1]] for p in points])
    index = np.argsort(values)
    
    return nonMaximalSuppression(index, points, shape, min_dist)







