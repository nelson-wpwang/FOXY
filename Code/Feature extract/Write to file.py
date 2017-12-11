import os
import re
import string

write_file = open("list&features.txt", "w")
write_file.write("    Name: sink, sofa, chair, dinningtable, oven, refrigrator, microwave, tvmonitor, bed")
read_file = open("list.txt", "r")
lists_str = read_file.read()

print(lists_str)
#lists_str.replace("'", "")
#lists_str = lists_str.replace(, '')
#g1 = [i.replace('"', '') for i in lists_str]
list_names = lists_str.split(", ")
#list_names.strip('"')
#g1 = [x.strip("'") for x in lists_str.split(', ')]
#print (g1)
print(list_names)
for item in list_names:
    write_file.write(item)
    write_file.write(": ")
    write_file.write("0,0,0,0,0,0,0,0,0")
    write_file.write("\n")

write_file.close()
read_file.close()
