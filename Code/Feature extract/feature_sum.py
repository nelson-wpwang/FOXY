import csv
import os
import glob

textile_lst = []
smallfea_lst = []

with open('csvinput.csv', 'w') as f:
    writer = csv.writer(f)
    path = "/Users/NelsonWang/BU/CS542/Project/FOXY/Code/Feature extract"
    with open('furniture_textile.csv', 'r') as c:
        reader = csv.reader(c)
        for item in reader:
            textile_lst.append(item)
            #print(item)
    with open('small_feature.csv', 'r') as n:
        small_fea = csv.reader(n)
        for item in small_fea:
            smallfea_lst.append(item)

    for path, dirs, files in os.walk(path):
        #print(type(dirs))
        #sort_dir = dirs.sort()
        for filename in sorted(dirs):

            chair = 0
            sofa = 0
            table = 0
            bed = 0
            leather_chair = 0
            stone_chair = 0
            wood_chair = 0
            leather_sofa = 0
            wood_sofa = 0
            stone_table = 0
            wood_table = 0
            oven = 0
            microwave = 0
            refrigerator = 0
            sink = 0
            tv_monitor = 0
            contents = os.listdir(os.path.join(path,filename))
            for item in contents:
                if item[0] == 'c':
                    chair += 1
                elif item[0] == 's':
                    sofa += 1
                elif item[0] == 't':
                    table += 1
                elif item[0] == 'b':
                    bed += 1
            for stuff in textile_lst:
                if stuff[0] == filename:
                    if stuff[1][0] == 'c':
                        if stuff[2] == "wood":
                            wood_chair += 1
                        elif stuff[2] == "leather":
                            leather_chair += 1
                        elif stuff[2] == "stone":
                            stone_chair += 1
                    elif stuff[1][0] == 's':
                        if stuff[2] == "wood":
                            wood_sofa += 1
                        elif stuff[2] == "leather":
                            leather_sofa += 1
                    elif stuff[1][0] == 't':
                        if stuff[2] == "wood":
                            wood_table += 1
                        elif stuff[2] == "stone":
                            stone_table += 1
            write_flag = 0
            for stuff in smallfea_lst:
                if stuff[0] == filename:
                    writer.writerow([filename, chair, leather_chair, stone_chair, wood_chair, sofa, leather_sofa, wood_sofa, table, stone_table, wood_table, bed, stuff[1], stuff[2], stuff[3], stuff[4], stuff[5]])
                    write_flag = 1
            if write_flag == 0:
                writer.writerow([filename, chair, leather_chair, stone_chair, wood_chair, sofa, leather_sofa, wood_sofa, table, stone_table, wood_table, bed, 0,0,0,0,0])
            #print (fea.read(46))

            #print(filename, chair, leather_chair, stone_chair, wood_chair, sofa, leather_sofa, wood_sofa, table, stone_table, wood_table, bed, oven, microwave, refrigerator, sink, tv_monitor)
