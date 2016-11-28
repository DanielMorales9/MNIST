import numpy as np



#method that compute mean by column for each label
def mean(my_data):
    training = np.ones(len(my_data[0]), bool)
    training[0] = False
    label_prev = -1
    count= 1
    pixel = np.array([np.zeros(len(my_data[0])-1, int)])
    prev = None
    for d in my_data:
        if d[0] != label_prev:
            label_prev = d[0]
            if prev is not None:
                prev = prev/float(count)
                pixel = np.append(pixel, [prev], axis=0)
            prev = d[training]
            count = 1
        else:
            prev = np.add(prev, d[training])
            count += 1
    prev = prev/float(count)
    pixel = np.append(pixel, [prev], axis=0)
    select = np.ones(len(pixel), bool)
    select[0] = False
    pixel = pixel[select,:]

    return pixel

#method that computes the median value
def median(my_data):
    training = np.ones(len(my_data[0]), bool)
    training[0] = False
    pixel = np.array([np.zeros(len(my_data[0])-1, int)])
    prev = np.array([my_data[0][training]])
    label_prev = 0
    i=1
    while i < len(my_data)-1:
        if label_prev != my_data[i][0]:
            label_prev = my_data[i][0]
            #compute median
            pixel= np.append(pixel, [np.median(prev, axis=0)], axis=0)
            prev = np.array([my_data[i][training]])
        else:
            prev = np.append(prev, [my_data[i][training]], axis=0)
        i+=1
    pixel = np.append(pixel, [np.median(prev, axis=0)], axis=0)
    select = np.ones(len(pixel), bool)
    select[0] = False
    pixel = pixel[select,:]
    return pixel

#method that computes the median value
def std(my_data):
    training = np.ones(len(my_data[0]), bool)
    training[0] = False
    pixel = np.array([np.zeros(len(my_data[0])-1, int)])
    prev = np.array([my_data[0][training]])
    label_prev = 0
    i=1
    while i < len(my_data)-1:
        if label_prev != my_data[i][0]:
            label_prev = my_data[i][0]
            #compute median
            pixel= np.append(pixel, [np.std(prev, axis=0)], axis=0)
            prev = np.array([my_data[i][training]])
        else:
            prev = np.append(prev, [my_data[i][training]], axis=0)
        i+=1
    pixel = np.append(pixel, [np.median(prev, axis=0)], axis=0)
    select = np.ones(len(pixel), bool)
    select[0] = False
    pixel = pixel[select,:]
    return pixel
