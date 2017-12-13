import os 

cwd = os.getcwd()
with open("train.txt", "w") as outF:
    name = ['leather','stone','wood']

    for j in range(3):
        k = os.path.join(cwd+'/imageData_aug','%s'%name[j])
        print(k)
        ls = os.listdir(k)
        for img in ls:
            print('%s/%s %d'%(k,img,j))
            outF.writelines('%s/%s %d\r'%(k,img,j))    
