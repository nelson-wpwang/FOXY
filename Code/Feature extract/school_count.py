import csv
import os
import glob

smallfea_lst = []
elem = 0
middle = 0
high = 0

with open('Lightningtest.csv', 'r+') as n:
    small_fea = csv.reader(n)
    for item in small_fea:
        if int(item[0]) >= 72144676:
            smallfea_lst.append(item)
            print(item)

for i in range(10001):
    if smallfea_lst[i][15] == '':
        elem += 1
    if smallfea_lst[i][16] == '':
        middle += 1
    if smallfea_lst[i][17] == '':
        high += 1


print("results:")
print("elem: ", elem, "middle: ", middle, "high: ", high)
