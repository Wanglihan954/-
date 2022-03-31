import numpy as np
class LinearRegression():
    def __init__(self,X,Y,learning_rate,epoch):
        self.X=X
        self.Y=Y
        self.mu=X.mean()
        self.sigma=X.std()
        self.X=(self.X-self.mu)/self.sigma#进行标准化
        self.theta=np.ones(X[0].size)
        self.batch_size=float(X[:,0].size)#计算batch_size
        self.ecoph=epoch
        self.lr=learning_rate
        self.count=0
    def f(self,X):
        return np.dot(X,self.theta)
    def E(self,X,Y):
        return  0.5*(np.power((self.f(X)-Y),2))/self.batch_size
    def fit(self):
        for k in  range(self.ecoph):
            error=self.E(self.X,self.Y)
            n=self.lr*(self.f(self.X)-self.Y)/self.batch_size
            tmp=self.theta-self.lr*np.dot(self.X.T,(self.f(self.X)-self.Y))/self.batch_size
            self.theta=tmp
            current_error=self.E(self.X,self.Y)
            error=current_error
            self.count+=1
    def predict(self,X):
        pred_y=self.f((X-self.mu)/self.sigma)
        return pred_y
#     下列代码为测试用
if __name__=='__main__':
    x=np.arange()
    y=2*np.ones([5])
    print(x.shape)
    print(y.shape)
    lr=LinearRegression(x,y,1e-3)
    lr.fit()
    print(lr.predict(x))