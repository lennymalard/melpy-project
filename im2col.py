import numpy as np

def get_indices(image_shape, window_shape, stride):
    output_height = int((image_shape[2] - window_shape + stride)//stride)
    output_width = int((image_shape[3] - window_shape + stride)//stride)
    
    level1 = np.repeat(np.arange(window_shape), window_shape)
    level1 = np.tile(level1, image_shape[1])
    
    increment = stride * np.repeat(np.arange(output_height), output_width)
    
    i = level1.reshape(-1, 1) + increment.reshape(1, -1) # indices i

    slide1 = np.tile(np.arange(window_shape), window_shape * image_shape[1])
    increment = stride * np.tile(np.arange(output_width), output_height)
    
    j = slide1.reshape(-1, 1) + increment.reshape(1, -1) # indices j

    k = np.repeat(np.arange(image_shape[1]), window_shape * window_shape).reshape(-1, 1) # indices canaux 
    return k, i, j

def im2col(images, window_shape, stride):
    k, i, j = get_indices(images.shape, window_shape, stride)
    columns = np.concatenate(images[:,k,i,j], axis=-1)  
    return columns

def col2im(columns, image_shape, window_shape, stride):
    images = np.zeros(image_shape)
    k, i, j = get_indices(image_shape, window_shape, stride)
    cols_reshaped =  np.array(np.hsplit(columns, image_shape[0]))
    np.add.at(images, (slice(None), k, i, j), cols_reshaped)
    return images
    