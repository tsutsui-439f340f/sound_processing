import numpy as np
import matplotlib.pyplot as plt
import tsutsui_lib as tt



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

def hanning(data,fs,N):
	han=np.hanning(fs)
	#振幅補正係数
	acf=1/(np.sum(han)/fs)
	for i in range(N):
		data[i]*=han
	return data,acf

def db(x,dBref):
	y=20*np.log10(x/dBref)
	return y

def aweightings(f):
	if f[0]==0:
		f[0]=1e-6
	else:
		pass
	ra=(np.power(12194,2)*np.power(f,4))/((np.power(f,2)+np.power(20.6,2))*np.sqrt((np.power(f,2)+np.power(107.7,2))*(np.power(f,2)+np.power(737.9,2)))*(np.power(f,2)+np.power(12194,2)))
	a=20*np.log10(ra)+2.00
	return a

def stft(data,sr,fs,N,acf):
	stft=[]
	fft_axis=np.linspace(0,sr,fs)
	a_scale=aweightings(fft_axis)
	for i in range(N):
		stft.append(db(acf*np.abs(np.fft.fft(data[i])/(fs/2)),2e-5))
	stft=np.array(stft)+a_scale
	fft_mean=np.mean(stft,axis=0)
	return stft,fft_mean,fft_axis
	

	

			
data,sr=tt.tsutsui_lib().load("ddd.wav")

ts=0

tp=8

data=data[sr*ts:sr*tp]

overlap=50
fs=4096

save_path="img2.jpg"


ov,f,N=overlaps(data,sr,fs,overlap)


v=1
han,acf=hanning(ov,fs,N)

if v==0:
	n=300
	plt.plot(np.linspace(0,44100,2048)[:n],np.abs(np.fft.fft(han[30])[:n]))
	plt.show()
elif v==1:
	stft,mean,axis=stft(han,sr,fs,N,acf)
	fig,ax=plt.subplots(2,1,figsize=(12,12))
	ax[0].plot(np.linspace(0,len(data)/sr,len(data)),data)
	ax[0].set_xlabel("time")
	ax[0].set_ylabel("amplitude")
	
	img=ax[1].imshow(stft.T,vmin=-20,vmax=50,extent=[0,f,0,sr],aspect='auto',cmap='gnuplot2')
	col=fig.colorbar(img)
	ax[1].set_yticks(np.arange(0,20000,200))
	col.set_label("[dB]")
	ax[1].set_ylabel("Frequency[Hz]")
	ax[1].set_xlabel("Time[s]")
	ax[1].set_ylim(0,1300)
	plt.show()
	
	plt.savefig(save_path)
else:
	stft,mean,axis=stft(han,sr,fs,N,acf)
	plt.imshow(stft.T,vmin=-20,vmax=50,extent=[0,f,0,sr],aspect='auto',cmap='gnuplot2')
	col=plt.colorbar()
	plt.yticks(np.arange(0,20000,500))
	col.set_label("[dB]")
	plt.ylabel("Frequency[Hz]")
	plt.xlabel("Time[s]")
	plt.ylim(0,1300)
	#plt.xlim(0,1)

	plt.show()


