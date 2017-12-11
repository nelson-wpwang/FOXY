import csv
import os

path = "/Users/NelsonWang/BU/CS542/Project/FOXY/Code/Feature extract/"

bottle_list = []
container_list = []
# Read all data from the csv file.
with open('csvinput.csv', 'r+') as b:
    bottles = csv.reader(b)
    for row in bottles:
        bottle_list.append(row)
    #bottle_list.append(bottles)
for item in bottle_list:
    print(item)
with open('LightningExport_test.csv', "r+") as l:
    container = csv.reader(l)
    #print(container)

    for stuff in container:
        #.encode('utf-8').strip()
        #print(stuff)
        container_list.append(stuff)

for item in container_list:
    match_flag = 0
    for things in bottle_list:
        if item[0] == things[0]:
            print("matched: ", item[0])
            item.extend((things[1:]))
            print("to be inserted: ", things[1:])
            match_flag = 1
    if match_flag == 0:
        item.extend((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))


for item in container_list:
    print(item)

with open('output.csv', 'w') as f:
    writer = csv.writer(f)
    for item in container_list:
        writer.writerow(item)
# data to override in the format {line_num_to_override:data_to_write}.
    #line_to_override = {1:['e', 'c', 'd'] }

# Write data to the csv file and replace the lines in the line_to_override dict.
    #writer = csv.writer(b)
    #for line, row in enumerate(bottle_list):
    #     data = line_to_override.get(line, row)
    #     print(data)
    #     writer.writerow(data)
