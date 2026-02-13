def unbiased_variance(label_list):
    n = len(label_list)
    if n < 2:
        return 0

    mean = sum(label_list)/n
    tot = 0
    for val in label_list:
        tot += (mean - val)**2

    return tot/(n-1)
