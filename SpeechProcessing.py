import wave
import numpy as np
import matplotlib.pyplot as plt
import struct as st
import time



def overlaps(data,sr,Fs,overlap):
	ts=len(data)/sr
	fc=Fs/sr
	x=Fs*(1-(overlap/100))
	N=int((ts-(fc*(overlap/100)))/(fc*(1-(overlap/100))))
	a=[]
	for i in range(N):
		ps=int(x*i)
		a.append(data[ps:ps+Fs])
		final_time=(ps+Fs)/sr
	return a,final_time,N

#hanning処理
def hanning(data,fs,N):
	han=np.hanning(fs)
	#振幅補正係数
	acf=1/(np.sum(han)/fs)
	for i in range(N):
		data[i]*=han
	return data,acf

#dB化
def db(x,dBref):
	y=20*np.log10(x/dBref)
	return y

def aweightings(f):
	if f[0]==0:
		f[0]=1e-6
	else:
		pass
	ra=(np.power(12194,2)*np.power(f,4))/((np.power(f,2)+np.power(20.6,2))*np.sqrt((np.power(f,2)+np.power(107.7,2))*(np.power(f,2)+np.power(737.9,2)))*(np.power(f,2)+np.power(12194,2)))
	return 20*np.log10(ra)+2.00

def stft(data,sr,fs,N,acf):
	stft=[]
	fft_axis=np.linspace(0,sr,fs)
	a_scale=aweightings(fft_axis)
	for i in range(N):
		stft.append(db(acf*np.abs(np.fft.fft(data[i])/(fs/2)),2e-5))
	stft=np.array(stft)+a_scale
	fft_mean=np.mean(stft,axis=0)
	return stft,fft_mean,fft_axis
	
def stft_plot(data,sr,overlap=50,fs=4096,save_path="img3.jpg"):
	
	ov,f,N=overlaps(data,sr,fs,overlap)
	v=1
	han,acf=hanning(ov,fs,N)

	if v==0:
		n=300
		plt.plot(np.linspace(0,44100,2048)[:n],np.abs(np.fft.fft(han[30])[:n]))
		plt.show()
	elif v==1:
		stft1,mean,axis=stft(han,sr,fs,N,acf)
		fig,ax=plt.subplots(2,1,figsize=(12,12))
		ax[0].plot(np.linspace(0,len(data)/sr,len(data)),data)
		ax[0].set_xlabel("time")
		ax[0].set_ylabel("amplitude")
	
		img=ax[1].imshow(stft1.T,vmin=-20,vmax=50,extent=[0,f,0,sr],aspect='auto',cmap='gnuplot2')
		col=fig.colorbar(img)
		ax[1].set_yticks(np.arange(0,20000,200))
		col.set_label("[dB]")
		ax[1].set_ylabel("Frequency[Hz]")
		ax[1].set_xlabel("Time[s]")
		ax[1].set_ylim(0,1300)
		plt.show()
	
		plt.savefig(save_path)
	else:
		stft2,mean,axis=stft(han,sr,fs,N,acf)
		plt.imshow(stft2.T,vmin=-20,vmax=50,extent=[0,f,0,sr],aspect='auto',cmap='gnuplot2')
		col=plt.colorbar()
		plt.yticks(np.arange(0,20000,500))
		col.set_label("[dB]")
		plt.ylabel("Frequency[Hz]")
		plt.xlabel("Time[s]")
		plt.ylim(0,1300)
		plt.show()

def write2(data,path,sr,channel=2,sample_width=2,level=1):
	w=wave.open(path,'w')
	w.setnchannels(channel)
	w.setsampwidth(sample_width)
	w.setframerate(sr)
	if sample_width==2:
		data=clip(data*32768.0,level,sample_width).astype("int16")
	elif sample_width==4:
		data=clip(data*2147483648.0,level,sample_width).astype("int32")
	data=st.pack("h"*len(data),*data)
	w.writeframes(data)
	w.close()
	
def fir_bpf(fe1,fe2,j,w=None):
	b=np.zeros(j+1)
	offset=j//2
	for m in range((-offset),(offset+1)):
		
		b[(offset)+m]=2*fe2*np.sinc(2*np.pi*fe2*m)-2*fe1*np.sinc(2*np.pi*fe1*m)
	
	if w==None:
		w=np.hanning(j+1)
	
	for m in range(0,j+1):
		b[m]*=w[m]
	return b	

def fir_bef(fe1,fe2,j,w=None):
	b=np.zeros(j+1)
	offset=j//2
	for m in range((-offset),(offset+1)):
		
		b[(offset)+m]=np.sinc(np.pi*m)-2*fe2*np.sinc(2*np.pi*fe2*m)+2*fe1*np.sinc(2*np.pi*fe1*m)
	
	if w==None:
		w=np.hanning(j+1)
	
	for m in range(0,j+1):
		b[m]*=w[m]
	return b
			
def fir_lpf(fe,j,w=None):
	b=np.zeros(j+1)
	offset=j//2
	for m in range((-offset),(offset+1)):
		
		b[(offset)+m]=2*fe*np.sinc(2*np.pi*fe*m)
	
	if w==None:
		w=np.hamming(j+1)
	
	for m in range(0,j+1):
		b[m]*=w[m]
	return b


def fir_hpf(fe,j,w=None):
	b=np.zeros(j+1)
	offset=j//2
	for m in range((-offset),(offset+1)):
		
		b[(offset)+m]=np.sinc(np.pi*m)-2*fe*np.sinc(2*np.pi*fe*m)
	
	if w==None:
		w=np.hamming(j+1)
	
	for m in range(0,j+1):
		b[m]*=w[m]
	return b

def fir(way,y,fe1,sr,w=None,fe2=None,all=True):
	start=time.time()
	if all==True:
		yd=np.zeros(y.shape[0])
		fer1=fe1/sr
		if fe2!=None:
			fer2=fe2/sr
			
		delta=fe1/sr
		
		j=int((3.1/delta)+0.5)-1
		if j%2==1:
			j+=1
		
		if way=="lpf":
			b=fir_lpf(fer1,j)
		elif way=="hpf":
			b=fir_hpf(fer1,j,w=w)
		elif way=="bpf":
			b=fir_bps(fer1,fer2,j,w=w)
		elif way=="bef":
			b=fir_bef(fer1,fer2,j,w=w)
		else:
			raise Exception("way is not found")
		
		for n in range(y.shape[0]):
			for m in range(j):
				if n-m>=0:
					yd[n]+=b[m]*y[n-m]
	else:
		#1secごとに処理
		flag=0
		yd=np.zeros(y.shape[0])
		for i in range(y.shape[0]//sr):
			data=y[i*sr:(i+1)*sr]
			if flag==0:
				flag=1
				
				fer1=fe1/sr
				if fe2!=None:
					fer2=fe2/sr
			
				delta=fe1/sr
		
				j=int((3.1/delta)+0.5)-1
				if j%2==1:
					j+=1
		
				if way=="lpf":
					b=fir_lpf(fer1,j,w=w)
				elif way=="hpf":
					b=fir_hpf(fer1,j,w=w)
				elif way=="bpf":
					b=fir_bps(fer1,fer2,j,w=w)
				elif way=="bef":
					b=fir_bef(fer1,fer2,j,w=w)
				else:
					raise Exception("way is not found")
				
			for n in range(data.shape[0]):
				for m in range(j):
					if n-m>=0:
						if (i==0)and(n==0):
							yd[0]+=b[m]*data[n-m]
						else:
							yd[(n+1)*(i+1)]+=b[m]*data[n-m]
	print(time.time()-start)		
			
	return yd


#clipping
def clip(data,level,sample_width,threshold=None):
		if threshold==None:
			if sample_width==2:
				threshold=32767.0
			elif sample_width==4:
				threshold=2147483647.0
		if data[data>threshold].shape[0]>1 or data[data<-threshold].shape[0]>1 :
			print("clipping")
			data[data>threshold]=threshold
			data[data<-threshold]=-threshold
			data*=level
			
		return data

def mel2freq(mel):
	return 700*(np.exp(mel/1125)-1)
def freq2mel(freq):
	return 1125*np.log(1+(freq/700))

#移動平均 平滑化	 連続した1つの音に対して有効 0値が含まれる配列に適用しない t=奇数のみ
def average(x,y,t=3):
	c=[]
	b=[]		
	for v in range(x.shape[0]):
		if v>=(t-1)/2 and v<=(x.shape[0]-1)-(t-1)/2:
			c.append(np.mean(y[v-(t-1)/2:v+(t-1)/2+1]))
			b.append(v)
	xm,xs=np.max(b),np.min(b)
	plt.plot(x[xs:xm+1],c)
	plt.show()

def two_average(data):
		f=[]
		for i in range(data.shape[0]):
			if (i+1)<data.shape[0]:
				f.append(np.sum(data[i:i+2])//2)
			else:
				f.append(data[i])
		return f

def melfilterbank(sr,nfft,n_filters,min_freq=0):
	"""
	メルフィルタバンク生成
	"""
	max_freq=sr/2
	max_mel=freq2mel(max_freq)
	min_mel=freq2mel(min_freq)
	df = sr / nfft
	dmel =max_mel /(n_filters+ 1) 
	melcenters = np.arange(1,n_filters + 1) * dmel
	fcenters = mel2freq(melcenters)
	mel_gap=np.linspace(min_mel,max_mel,n_filters+2)
	freq_gap=mel2freq(mel_gap)
	f=np.floor((nfft+1)*freq_gap/sr)
		
	H=[]	
	for n in range(n_filters):
		n+=1
		h=[]
		for k in range(max(f).astype(np.int32)):
			k+=1
			if k<f[n-1]:
				h.append(0)
			elif (f[n-1]<=k) and (k<=f[n]):
				val=(k-f[n-1])/(f[n]-f[n-1])		
				h.append(val)
			elif (f[n]<=k) and (k<=f[n+1]):
				val=(f[n+1]-k)/(f[n+1]-f[n])
				h.append(val)
			else:
				h.append(0)
		H.append(h)
	H=np.array(H)
	#このfreqはフィルターバンクのプロット時に補正するもの
	freq=np.linspace(0,max_freq,max(f))
	return H,freq,fcenters
		
def mfcc(stft,sr,nfft,n_filters,min_freq=0):
	H,_,fc=melfilterbank(sr,nfft,n_filters,min_freq)
	Melspec=[]
	for d in range(stft.shape[0]):
		mel=10*np.log10(np.dot(stft[d][:nfft/2],H.T))
		Melspec.append(mel)		
	return Melspec,fc

def dct(data):
	N=data.shape[0]
	f=np.array([phi(i,N) for i in range(N)])	
	return f.dot(data)
def idct(c):
	N=c.shape[0]
	f=np.array([phi(i,N) for i in range(N)])
	return np.sum(f.T * c ,axis=1)
def phi(k,N):

	if k == 0:
		return np.ones(N)/np.sqrt(N)
	else:
		return np.sqrt(2.0/N)*np.cos((k*np.pi/	(2*N))*(np.arange(N)*2+1))	

class SpeechProcessing:
	def __init__(self):
		self.n_sample=None
		self.sr=None
		self.sample_width=None
		self.channels=None
		self.record_time=None
		self.l_channel=None
		self.r_channel=None
		self.nfft=None
		
	def load(self,path=None,sr=None,all=None,size=None):
		if path==None:
			raise Exception("path is not found")
		else:
			with wave.open(path,"rb") as file:
				if size==None:
					self.n_sample=file.getnframes()
				else:
					self.n_sample=size
						
				if sr==None:
					self.sr=file.getframerate()
					
				else:
					self.sr=sr
					
				self.channels=file.getnchannels()
				self.sample_width=file.getsampwidth()
				self.buf=file.readframes(self.n_sample)
				
			self.record_time=self.n_sample/self.sr
			if self.sample_width==2:
				data=np.frombuffer(self.buf,dtype='int16')
				data=data/32768.0
			else:
				self.sample_width==4
				data=np.frombuffer(self.buf,dtype='int32')
				data=data/2147483648.0
			
			if self.channels==2:
				self.l_channel=data[0::2]
				self.r_channel=data[1::2]
				#データ解析時はデータ数を減らすのを推奨
				#all==None :default
				if all==None:
					data=data[0::2]
		
			del self.buf
	
			return np.array(data),self.sr
		
	def ifft(self,data):
		ift=np.real(np.fft.ifft(data))
		return ift
	def reverse(self,data):
		return data[::-1]	
	def cep(self,fft,cep_coff,vis=False):
		n_time=fft.shape[0]/self.sr
		logD=20*np.log10(fft)
		cep=np.real(np.fft.ifft(logD))
		arr=[]
		ceparr=[]
		cep_coff=np.array(cep_coff)
		
		for i in cep_coff:
			dd=np.array(cep)
			dd[i:len(dd)-i+1]=0
			ceparr.append(dd)
			arr.append(np.fft.fft(dd,len(fft)))
		arr=np.array(arr)
		ceparr=np.array(ceparr)
		
		if vis==True:
			x=np.linspace(0,n_time,n_time*self.sr)
			fig,ax=plt.subplots(2,2)
			ax[0,1].plot(cep[:100])
			ax[0,1].set_xlabel("frequency")
			ax[0,1].set_ylabel("cepstrum")
			#ax[1,1].plot(logD[:logD.shape[0]/2])
			ax[1,1].plot(logD)
			"""
			for i in range(len(arr)):
				ax[1,1].plot(arr[i,:logD.shape[0]/2])
			"""
			for i in range(len(arr)):
				ax[1,1].plot(arr[i])

			
			ax[1,0].plot(ceparr[:100],"b")
			ax[1,0].set_xlabel("quefrency")
			ax[1,0].set_ylabel("cepstrum")
			for i in range(arr.shape[0]):
				ax[1,0].plot(ceparr[i,:100])
			'''	
			ax[0,0].plot(x,data)
			ax[0,0].set_xlabel("time")
			ax[0,0].set_ylabel("amplitude")
			'''
			plt.tight_layout()
			plt.show()
			
		else:
			return ceparr
			
	#ボリューム下げる必要があればレベル指定する。
	def write(self,path,data,level=1):
		
		with wave.open(path,'w') as w:
			w.setnchannels(self.channels)
			w.setsampwidth(self.sample_width)
			w.setframerate(self.sr)
		
			if self.sample_width==2:
				data=self.clip(data*32768.0,level).astype("int16")
			elif self.sample_width==4:
				data=self.clip(data*2147483648.0,level).astype("int32")
			#data=st.pack("h"*len(data),*data)
			w.writeframes(data)
						
		
	def clip(self,data,level=None,threshold=None,ratio=None,complex=None):
		if threshold==None:
			if self.sample_width==2:
				threshold=32767.0
			elif  self.sample_width==4:
				threshold=2147483647.0
		
		if complex!=None:
			gain=1/(threshold+(1-threshold)*ratio)
			print(gain)
			#元の波形を圧縮
			data[data>threshold]=threshold+(data[data>threshold]-threshold)*ratio
			data[data<-threshold]=-threshold+(data[data<-threshold]+threshold)*ratio
			#増幅の余地を残してるため全体が増幅される
			data*=gain
		else:
			if data[data>threshold].shape[0]>1 or data[data<-threshold].shape[0]>1 :
				print("clipping")
				data[data>threshold]=threshold
				data[data<-threshold]=-threshold
				data*=level
			
		return data
		
	def distortion(self,data,threshold,gain=100,level=0.5):
		data=data*gain
		data=self.clip(data,level,threshold)
		return data
	
	#depthはオリジナルの振幅のレンジに合わせてrateを調整して使う
	#order n^2 とてつもなく計算量多いと落ちる
	#matplotlib overflow問題発生したら再起動で解決する
	def ring_modulation(self,data,depth=1,rate=1):
		n=np.arange(data.shape[0])
		y=depth*np.sin(2*np.pi*rate*n)
		mod=data*y
		return mod
		
	def chorus(self,data,depth,rate,time):
		d=self.sr*time #d(sec)
		depth=self.sr*depth
		n=np.arange(data.shape[0]).astype(np.float32)
		tau=d+depth*np.sin(2*np.pi*n*rate/self.sr) #d+-depth(msec)の範囲で揺らす
		t=n-tau
		m=t.astype(np.int32)
		delta=t-m.astype(np.float32)
		a=np.array(np.where(m>0))
		b=np.array(np.where(m+1<data.shape[0]))
		intersect=np.intersect1d(a,b)
		data[intersect]+=delta[intersect]*data[m[intersect]+1]+(1.0-delta[intersect])*data[m[intersect]]
		
		return data
	
	def autpan(self,depth,rate):
		n=np.arange(self.n_sample)
		l_data=(1+depth*np.sin(2*np.pi*rate*n/self.sr))*self.l_channel
		r_data=(1-depth*np.sin(2*np.pi*rate*n/self.sr))*self.r_channel
		c=self.altanative_connect(l_data,r_data)		
		return c
	
	#delayはmsec 大きいほど過去のデータに引っ張られる。
	def vocal_cancel(self,data,delay,data_size=None):
		delay*=self.sr
		
		s1=data[0::2]-data[1::2]
		
		indent=(np.arange(self.n_sample)-delay).astype(np.int32)
		
		j=np.argwhere(indent>=0)
		sl1=s1
		sr1=s1
		"""
		for i,g in enumerate(j):
			sl1[g]+=s1[i]
			sr1[g]-=s1[i]
		"""
		sl1[j]+=s1[j-np.min(j)]
		sr1[j]+=s1[j-np.min(j)]
		c=self.altanative_connect(data,sl1,sr1)
		return c
		
		
	
	def altanative_connect(self,data,l_data,r_data):
		data=np.arange(self.n_sample)
		return np.insert(r_data,data,l_data)
	
	
#適正a 0.7~0.1
#適正ディレイ0.35~0.7	

	def delay(self,data,delay=0.35,rep=2,a=0.5):
		if self.sr==None:
			self.sr=44100
		delay*=self.sr 
		for i in range(data.shape[0]):	
			for n in range(1,rep):
				m=int(i-n*delay)
				if m>=0:
					data[i]+=pow(a,n)*data[m]			
					
		return data
	
	def toremoro(self,data,depth=0.5,rate=5):
		n=np.arange(0,data.shape[0])
		a=1+depth*np.sin(2*np.pi*rate*n/self.sr)
		data*=a
		return data


	def resample(self,data,pitch=1.5,j=24,sigma=None):
		
		def gen(offset):
			m=[np.arange(of-j/2,of+j/2+1) for of in offset]
			print(m)	
			g=np.argwhere(m>=0) 
			f=np.argwhere(m<self.n_sample)
			intersect=np.intersect1d(g,f)		
			data+=data[intersect]*np.sinc(np.pi*(t-intersect))
			yield data
			
			
		if sigma!=None:
			j=round(3.1/sigma)
			if j%2!=0:
				j+1
		
		
		n=np.arange(0,data.shape[0])
		t=pitch*n
		offset=t.astype(np.int32)
		
		for data in gen(offset):
			print(data)
							
		
		#m=[np.arange(of-j/2,of+j/2+1) for of in offset]
		#m=map(np.arange(offset-j/2,offset+j/2+1),offset)		
		return data	
	
	def complex(self,y,threshold=None,ratio=None):
		
		if threshold==None and ratio==None:
			if self.sample_width==2:
				threshold=32767.0
			elif  self.sample_width==4:
				threshold=2147483647.0
			ratio=0.00001
		
		if threshold==None or ratio==None:
			raise Exception("thresholdとratioが未設定です")
		data=self.clip(y,threshold=threshold,ratio=ratio,complex=True)
		return data 
	
	"""
	def stft(self,y,N=None,t=None):
		
		#fftデータポイント間隔N
		
		
		if N==None and t==None:
			raise Exception("input parameter 'N or t'")
		if N==None:
			N=44100*t
		
		#self.nfft=nfft
		S=N/2
		#データ数
		n=int(y.shape[0]//S)
		
		X=[]
		Y=[]
		SN=[]
		sn=0
		FFT=[]
		
		for _ in range(n):
			if y[sn:sn+N].shape[0]==N:
				X.append(np.arange(0,S))
				w=np.hanning(N)
				sy=y[sn:sn+N]
				Y.append(sy)
				fft=np.fft.fft(w*sy)
				abs_fft=np.abs(fft)
				FFT.append(abs_fft)
				SN.append(sn)
				sn+=S
				
			else:
				break
		
		freq=np.linspace(0,self.sr,len(FFT[0]))
		t=np.linspace(0,y.shape[0]/self.sr,len(FFT))
		return np.array(FFT),freq,t,SN
	"""
		
	def fft_plot(self,freq,fft,max=None):
		if max==None:
			plt.plot(freq[:len(freq)//2],fft[:len(fft)//2])
		else:
			
			max_p=np.count_nonzero(np.array(freq)<max)
			
			
		
	def filter_change_modulation(self,y,sr,way,a=50,t=2,m=10000):
		#tは最初のグラフでの周波数領域調節のためのパラメータ
		#aは格フィルターのパラメーター
		
		fft=[]
		c=0
		f=0
		ans=np.zeros_like(y)
		d=y.shape[0]
		#推奨10000
		#1000だとデータ数44100*300で訳404epochまで処理時間1分弱
		
		n=d//m
		han=np.hamming(m)
		if way=="sigmoid":
			#a=100
			x=np.linspace(-1,1,m//2)
			w=1/(1+np.exp(-a*x))
			w=w[::-1]
			filter=np.zeros(m)
			filter[:m//2]=w
			filter[m//2:]=w[::-1]
	
		elif way=="low_path":
			#aを減らすと音が篭る
			#推奨分解能m=1000
			
			#a=250
			
			filter=np.zeros(m)
			filter[:a]=1
			filter[a:]=0
	
		elif way=="high_path":
			#aを増やすと検出領域狭くなる。
			#推奨分解能m=1000
			#a=200
			filter=np.zeros(m)
			filter[:a]=0
			filter[a:]=1
		elif way=="inverse_sigmoid":
			#a=100
			x=np.linspace(-1,1,m//2)
			w=-1/(1+np.exp(-a*x))
			filter=np.zeros(m)
			filter[:m//2]=w
			filter[m//2:]=w[::-1]
		 
		print("-"*20)
		print("epochs:{} ".format(f))
		print(n)
		for i in range(n):			
			
			fft=np.fft.fft(y[m*i:m*(i+1)]*han)
			mid=fft*filter
			ans[i*m:(i+1)*m]=np.real(np.fft.ifft(mid))/han
			
			c+=1
			if c==60:
				c=0
				f+=1
				print("epochs:{} ".format(f))
				#周波数検証
				if f==1:
					fig,ax=plt.subplots(3)
					x=np.linspace(0,44100,m)[:m//t]
					freq=np.fft.fftfreq(filter.shape[0],d=self.sr)
					ax[0].plot(x,fft[:m//t])
					ax[0].plot(x,filter[:m//t],"pink")
					ax[1].plot(x,mid[:m//t])
					
					ax[2].plot(x,filter[:m//t])
					plt.show()
									
				
		return ans
