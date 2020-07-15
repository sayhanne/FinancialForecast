import pandas as pd
import numpy as np

data_dest="proje\\data_for_model.csv"
class HannesTool:
    def get_mean(self, Y):
        tot = 0
        for i in range(0, len(Y)):
            tot += Y[i]
        return tot / len(Y)

    def getTSS(self, Y):
        mean = self.get_mean(Y)
        tss = 0
        for i in range(0, len(Y)):
            tss += (Y[i] - mean) ** 2
        #print("tss:",tss)
        return tss

    def getRSS(self, fx, Y):
        rss = 0
        for i in range(0, len(Y)):
            rss += (Y[i] - fx[i]) ** 2
        #print("rss:",rss)
        return rss

    def getR2(self, fx, Y):
        return 1 - self.getRSS(fx, Y) / self.getTSS(Y)

    def to_X_Y(self, data):
        arr=[]
        for i in range(0,len(data)-1):
            arr.append([data[i],data[i+1]])
        return arr
    
    def get_err(self,type,W):
        test_x=np.array(pd.read_csv(data_dest)["te_now"].dropna())
        test_y=np.array(pd.read_csv(data_dest)["te_next"].dropna())
        fx=test_x*W[1]+W[0]
        
        if(type=='R2'):
            return self.getR2(fx, test_y)
        elif(type=='Class'):
            return self.get_err_class(fx, test_y)
    
    def get_err_class(self,fx,Y):
        total=0
        for i in range(len(Y)):
            if(Y[i]*fx[i]>0):
                print("s", Y[i], fx[i])
                total+=1
        print("total/len",total, len(Y))
        return total/len(Y)


