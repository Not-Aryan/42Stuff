import numpy as np

np.random.seed(100)
a = np.random.randint(0, 5, 10)
# print('Array: ', a)
# print(a[np.unique(a)])

b = np.array(a, dtype=bool)
i = 0
for x in a:
    # print(i)
    if (np.where(a==x)[0][0] == i):
        b[i] = False
    else:
        # print(x)
        # print(np.where(a==x)[0][0])
        # print(i)
        # print("----")
        b[i] = True;
    i = i+1

print(b)