def sortByFirstColumn(my_data):
    #sort data by column
    a = []
    for d in my_data[0,:]:
        a.append("int64")
    dtype = ','.join(a)
    #this is what implements the sorting by columns
    my_data.view(dtype=dtype).sort(order=['f0'], axis=0)
