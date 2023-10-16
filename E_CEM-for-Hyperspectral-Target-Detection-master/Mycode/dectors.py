
import numpy as np
from utils import Detector

class Detectors(Detector):   # 检测器的类

    def __init__(self):
        Detector.__init__(self)

    def cem(self):
        # Basic implementation of the Constrained Energy Minimization (CEM) detector
        # 约束能量最小化(CEM)探测器的基本实现
        # Farrand, William H., and Joseph C. Harsanyi. 
        # "Mapping the distribution of mine tailings 
        # in the Coeur d'Alene River Valley, Idaho, through the use of a constrained energy minimization 
        # technique." 
        # 通过使用有限能源最小化技术，绘制爱达荷州科达伦河谷的尾矿分布图
        # Remote Sensing of Environment 59, no. 1 (1997): 64-76.
        # 环境遥感
        size = self.img.shape   # 得到图像矩阵的大小
        R = np.dot(self.img, self.img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv(R), self.tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, self.img).T  # y=w'* X;
        return result

    def ace(self):
        # Basic implementation of the Adaptive Coherence/Cosine Estimator (ACE)
        # 自适应相干/余弦估计器（ACE）的基本实现
        # Manolakis, Dimitris, David Marden, and Gary A. Shaw. "Hyperspectral image processing for 
        # automatic target detection applications." Lincoln laboratory journal 14, no. 1 (2003): 79-116.
        size = self.img.shape
        img_mean = np.mean(self.img, axis=1)[:, np.newaxis]
        img0 = self.data.img-img_mean.dot(np.ones((1, size[1])))
        R = img0.dot(img0.T)/size[1]
        y0 = (self.tgt-img_mean).T.dot(np.linalg.inv(R)).dot(img0)**2
        y1 = (self.tgt-img_mean).T.dot(np.linalg.inv(R)).dot(self.tgt-img_mean)
        y2 = (img0.T.dot(np.linalg.inv(R))*(img0.T)).sum(axis=1)[:, np.newaxis]
        result = y0/(y1*y2).T
        return result.T


    def detect(self, img_data):
        self.load_data(img_data)
        # return {'CEM': self.cem, 'ACE': self.ace, 'MF': self.mf, 'SID': self.sid, 'SAM': self.sam}  只有SAM和SID两种数据的情况下
        #return {'CEM': self.cem}
        return {'CEM': self.cem,'ACE': self.ace}