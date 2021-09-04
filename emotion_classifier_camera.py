#-*- coding: utf-8 -*-

import cv2
import sys
import gc
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
model_path='C:/Users/GU/Desktop/model'
img_size=48

#load json and create model arch
emo_labels = ['non','confused']
num_class = len(emo_labels)
#加载模型结构
json_file=open(model_path+'/model_json.json')    #加载模型结构文件
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)      #结构文件转化为模型
#加载权重
model.load_weights(model_path+'/model.h5')#h5文件保存模型的权重数据

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
              
    #框住人脸的矩形边框颜色       
    color = (0, 0, 2555)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt.xml"
    
    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频
        
        #图像灰化，降低计算复杂度（当然，用于训练的数据集也是灰的）
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

        #利用分类器识别出哪个区域为人脸，返回检测到的人脸序列（矩形框四个参数）
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.1,
                                    minNeighbors = 1, minSize = (120, 120))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect   #人脸矩形框的左上角坐标、宽、高
                images=[]
                rs_sum=np.array([0.0]*num_class)    #([0, 0, 0, 0, 0, 0, 0])
                #截取脸部图像提交给模型识别这是谁
                image = frame_gray[y: y + h, x: x + w ]     #注意这里x、y的先后顺序
                image=cv2.resize(image,(img_size,img_size)) #将人脸缩放成网络所对应的输入图片大小
                image=image*(1./255)                        #归一化
                images.append(image)
                images.append(cv2.flip(image,1))    #水平翻转
                images.append(cv2.resize(image[2:45,:],(img_size,img_size)))  #裁切
                for img in images:
                    image=img.reshape(1,img_size,img_size,1)
                    #预测出每个类别的概率值
                    list_of_list = model.predict_proba(image,batch_size=32,verbose=1)
                    result = [prob for lst in list_of_list for prob in lst]
                    rs_sum+=np.array(result)
                print(rs_sum)
                rs_sum[0] = rs_sum[0] * 0.016
                print(rs_sum)
                label=np.argmax(rs_sum)
                emo = emo_labels[label]
                print ('Emotion : ',emo)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                #文字提示是谁
                cv2.putText(frame,'%s' % emo,(x + 30, y + 30), font, 1, (255,0,255),4)
        cv2.imshow("识别朕的表情！", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(30)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
