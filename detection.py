import cv2
import numpy as numpy
import time
class detection:
    def_init__(self,videopath,configpath,modelpath,classespath):
        self.videopath=videopath
        self.configpath=configpath
        self.modelpath=modelpath
        self.classespath=classespath

        self.net=cv2.dnn_detectionModel(self.modelpath,self.configpath)
        self.net.setIinputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
        self.readclasses()

    def readClasses(self):
        with open(self.classespath, 'r') as f:
            self.classesList=f.read().splitlines()
        self.classesList.insert(0,'__Background__')
        print(self.classesList)
    def onVideo(self):
        cap=cv2.VideoCapture(self.videopath)
        if(cap.isOpened()==false):
            print("Error opening file...")
            return
        (success,image)=cap.read()
        while success:
           classLabelIDs,confidences,bboxs= self.net.detect(image,confThreshold=0.5)
           bboxs=list(bboxs)
           



