import numpy as np
import matplotlib.pyplot as plt 

sr=4000
sec=2
data=np.linspace(0,sec,sr*sec)

def show(x,y):
	plt.plot(x,y)
	plt.show()

#データ生成
y=0
for i in range(1,19):
	if i%3:
		y+=0.7*np.sin(2*np.pi*i*data)
	else:
		y+=0.3*np.cos(2*np.pi*data*i)

gain=100
threshold=50
#クリッピング
y*=-gain
y[y>threshold]=threshold
y[y<-threshold]=-threshold

#ファズ
#データに対して絶対値をとる
y[y<0]*=-1		
												
plt.plot(data,y)
plt.title("clipping")
plt.xlabel("sec")
plt.ylabel("amplitude")
#plt.xlim(0,np.max(data)+1)
plt.ylim(0,70)
plt.show()
