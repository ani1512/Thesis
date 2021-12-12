import os
import random

path = os.path.join(os.getcwd(), 'CUB_200_2011')

with open(os.path.join(path, 'train_val_test_split.txt'), 'w') as nf:
    with open(os.path.join(path, 'train_test_split.txt'), 'r') as f:
        for line in f:
            image_id, flag = line.split()
            if flag == '0':
                prob = random.randint(0, 1)
                if prob:
                    print("in")
                    flag = '2'
                    line = image_id + " " + flag + "\n"
            nf.write(line)
