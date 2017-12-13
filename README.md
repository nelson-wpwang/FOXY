# FOXY

This is a machine learning project by CS542 at Boston University. In this project, we build a system that can provide an estimation of the house sales price merely by several house images. 

1.Download the dependencies of caffe and install some softwares include python, opencv, matlab, cuda, cudnn, downloads one in (Atlas,Blas, Mkl), you can follow the instructions in http://caffe.berkeleyvision.org/installation.html 


2.When finied the software detection, input "git clonoe https://github.com/pjreddie/darknet"," git clone https://github.com/nelson-wpwang/FOXY"to download the model and photo data, then make the file by inputting "cd darknet", "make". 

3.After that, test the modle by inputting "./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg", it will show the result the detecting the iamge dog.jpg. If you want to test my data, just overlap the code in Object_detection.py file to the darknet.py file in the darknet.py folder and change the path of photo in Object_detection.py and then run it

4.To run the prediction part, just enter the prediction file and run the 542Projectgbr file. note the csv file and the python must be at the same directory.




