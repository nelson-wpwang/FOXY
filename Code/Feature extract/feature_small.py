import csv

fea = open("jiadian.txt", 'r')
read_input = fea.readline()
inf = read_input[:46]
id_num = inf[38:]
oven = 0
microwave = 0
refrigerator = 0
sink = 0
tv_monitor = 0

fea_lst = []
out_lst = []
while inf != '':
    fea_lst.append(inf)
    read_input = fea.readline()
    inf = read_input[:46]

for item in fea_lst:
    print("in the loop")
    if item[0] == '/':
        if item[38:] == id_num:
            continue
        else:
            out_lst.append([id_num, oven, microwave, refrigerator, sink, tv_monitor])
            oven = 0
            microwave = 0
            refrigerator = 0
            sink = 0
            tv_monitor = 0
            id_num = item[38:]
            continue
    else:
            if item[0] == "o":
                oven = 1
            elif item[0] == 'm':
                microwave = 1
            elif item[0] == 'r':
                refrigerator = 1
            elif item[0] == 's':
                sink = 1
            elif item[0] == 't':
                tv_monitor = 1
            else:
                continue

with open('small_feature.csv', 'w') as n:
    writer = csv.writer(n)
    for item in out_lst:
        writer.writerow(item)
        print (item)
