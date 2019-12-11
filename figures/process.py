with open('loss_random_lay2.txt') as in_file:
    running_av = 0
    count = 0.0
    for line in in_file.readlines():
        loss_val = float(line.split()[-1])
        running_av += loss_val
        count += 1.0
        if count == 16.0:
            print(running_av/count)
            count = 0
            running_av = 0.0
