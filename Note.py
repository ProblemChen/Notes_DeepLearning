import torch
from torch import nn

#多层感知机的从零实现
#初始化模型参数
num_inputs,num_outputs,num_hiddens 
W1 =np.random.normal(scale=0.01,size =(num_inputs,num_hiddens))
b1 =np.zeros(num_hiddens)
W1 =np.random.normal(scale=0.01,size=(num_hiddens,num_outputs))
b2 =np.zeros(num_outputs)
params =[W1,b1,W2,b2]

for param in params:
	param.requires_grad_()

#激活函数
def relu(X):
	return np.maxinum(X,0)
#模型
def net(X):
	X=X.reshape(-1,num_inputs)
	H=relu(np.dot(H,W1)+b1)
	return H@W2 +b2

def corr2d(X,K):
	h,w =K.shape
	Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
	return Y   

class Conv2d(nn.Module):
	def __init__(self,kernel_size):
		super().__init__()
		self.weights =nn.Parameter(torch.rand(kernel_size))
		self.bias =nn.Parameter(torch.rand(1))

	def forward(self,X):
		return corr2d(X,self.weights) + self.bias

#学习由卷积核X到Y

conv2d =nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
#nn.Conv2d(in_channels,out_channels,kernel_size,padding,stride)
X=torch.rand(6,8)
Y=torch.rand(6,7)

X=X.reshape(1,1,6,8)
X=Y.reshape(1,1,6,7)

for i in range(10):
	Y_hat =conv2d(X)
	l =(Y_hat-Y)**2
	conv2d.zero_grad()
	l.sum().backward()
	conv2d.weight.data[:]-= 3e-2 * conv2d.weight.grad
	if (i+1) % 2==0:
		print(f'bath {i+1},loss {l.sum():.3f}')



# How to Padding 

def comp_conv2d(conv2d,X):
	X=X.reshape((1,1)+X.shape)
	Y=conv2d(X)
	return Y.shape[2:]

conv2d =nn.Conv2d(1,1,kernel_size=3, padding=1)
X=torch.rand(8,8)
comp_conv2d(conv2d,X)

#多输入通道多输出通道	

from d2l import torch as d2l

def corr2d_multi_in(X,K):
	return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

def corr2d_multi_out(X,K):
	return torch.stack([corr2d_multi_in(X,k) for k in K],0)

#1*1的卷积核实现
def corr2d_multi_in_out_1x1(X,K):
	c_i,h,w =X.shape
	c_o=K.shape[0]
	X=X.reshape((c_i,h*w))
	K=K.reshape((c_o,c_i))
	Y=torch.matmul(X,K)
	return Y.reshape(c_o,h,w)	
#nn.Module中的调用demo
conv2d =nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=2)

#池化层code
def pool2d(X,pool_size,mode='max'):
	p_h,p_w=pool_size
	Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			if mode=='max':
				Y[i,j] =X[i:i+p_h,j:j+p_w].max()
			elif mode=='arg':
				Y[i,j] =X[i:i+p_h,j:j+w].mean()
	return Y 


#LeNet_Code
import torch 
from torch import nn
from d2l import torch as d2l

class Reshape(nn.Module):
	def forward(self,x):
		return x.view(-1,1,28,28)

net =nn.Sequential(
	Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),
	nn.Sigmoid(),
	nn.AvgPool2d(2,stride=2)#池化层不会改变通道数·
	nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
	nn.AvgPool2d(2,stride=2),nn.Flatten(),
	nn.Linear(16*5*5,120),nn.Sigmoid(),
	nn.Linear(120,84),nn.Sigmoid(),
	nn.Linear(84,10)
	)

#检查模型
X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
	X=layer(X)
	print(layer.__class__.__name__,'Out Shape: \t',X.shape)


#Dataset and DataLoaders
#Creating a custom Dataset for my file
import os 
import pandas as pd 
from torchvision.io import read_image

def CustomImageDataset(Dataset):
	def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
		self.img_labels=pd.read_csv(annotations_file)
		self.img_dir =img_dir
		self.transform =transform 
		self.target_transform =target_transform

	def len(self):
		return len(self.img_labels)

	def __getitem__(self,idx):
		img_path =os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
		image  =read_image(img_path)
		label =self.img_labels.iloc[idx,1]
		if self.transform:
			image =self.transform(image)
		if self.target_transform:
			label =self.target_transform(label)
		return image,label 
#Preparing your data for training with DataLoaders
from torch.utils.data import DataLoader
train_dataloader =DataLoader(training_data,bath_size =64,shuffle=True)
test_dataloader  =DataLoader(test_data,bath_size=64,shuffle=True)

#Iterate through the DataLoader
#display image and label
train_features,train_labels =next(iter(train_dataloader))


bath_size =256
train_iter,test_iter =d2l.load_data_fashion_mnist(bath_size=bath_size)

def evaluate_accuracy_gpu(net,data_iter,device=None):
	if isinstance(net,torch.nn.Module):
		net.eval()
		if not device :
			device =next(iter(net.Parameters())).device
	Matric =d2l.Accumulator(2)
	for X,y in data_iter:
		if isinstance(X,list):
			X=[x.to(device) for x in X]
		else:
			X=X.to(device)
		y=y.to(device)
		metric.add(d2l.accuracy(net(X),y),y.numel())
	return metric[0]/metric[1]

# def train_ch6(net, train_iter,test_iter,num_epochs,lr,device):
# 	"""Train this module with GPU"""
# 	def init_weights(m):
# 		if type(m) ==nn.Linear or type(m) ==nn.Conv2d:
# 			nn.init_xavier_uniform_(m.weight)
# 	net.apply(init_weights)
# 	print('training on',device)
# 	net.to(device)
# 	optimizer =torch.optim.SGD(net.Parameters(),lr=lr)
# 	loss =nn.CrossEntropyLoss()
# 	

#实现Drop_OutLayer 经常作用于隐藏全连接层的输出上
#一个好的模型需要对输入的数据扰动鲁棒，
#使用有噪音的数据等价于Tikhonov正则,丢弃法：在层之间加入噪音
#无差别的加入噪音：E(X')=X
def dropout_Layer(X,drop_out):
	assert 0<=drop_out<=1
	if drop_out ==0:
		return X 
	if drop_out ==1:
		return torch.zeros_like(X)
	mask =(torch.randn(X.shape)>drop_out).float()
	return mask*X/(1,0-drop_out)


#实现Vgg块
import torch 
from torch import nn 
from d2l import torch as d2l

def vgg_block(num_convs,in_channels,out_channels):
	layers=[]
	for i in range(num_convs):
		layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
		layers.append(nn.ReLU())
		in_channels =out_channels
	layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
 	return nn.Sequential(*layers)

conv_arch =((1,64),(1,128),(2,256),(2,512),(2,512))
#conv_arch=((num_convs,in_channels)
def vgg(conv_arch):
	conv_blks=[]
	in_channels=1
	for (num_convs,out_channels) in conv_arch:
		conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
		in_channels = out_channels

	returk nn.Sequential(*conv_blks,nn.Flatten(),
			nn.Linear(out_channels*7*7,4096),nn.ReLU(),
			nn.Dropout(0.5),nn.Linear(4096,4096),nn.ReLU(),
			nn.Dropout(0.5),nn.Linear(4096,10))

net =vgg(conv_arch)
X=torch.randn(1,1,224,224)
for blk in net:
	X=blk(X)
	print(blk.__class__.__name__, 'output_shape: \t',X.shape)

#NiN网络
import torch 
from torch import nn 
def nin_block(in_channels,out_channels,kernel_size,strides,padding):
	return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size，strides,padding)
		nn.ReLU(),nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
		nn.Conv2d(out_chnnels,out_channels,kernel_size=1),nn.ReLU()
		)

net =nn.Sequential(
	nin_block(1,96,kernel_size=11,strides=4,padding=0),
	nn.MaxPool2d(3,stride=2),
	nin_block(96,256,kernel_size=5,strides=1,padding=2),
	nn.MaxPool2d(3,stride=2),
	nin_block(256,384,kernel_size=3,strides=1,padding=1),
	nn.MaxPool2d(3,stride=2),nn.Dropout(0.5),
	nin_block(348,10,kernel_size=3,strides=1,padding=1),
	nn.AdaptiveAvgPool2d((1,1)),
	nn.Flatten())
#检查每个块的输出形状
X=torch.rand((1,1,224,224))
for layer in net:
	X=layer(X)
	print(layer.__class__.__name__,'Output_shape:\t',X.shape)

#训练模型	
from d2l import torch as d2l
lr,num_epochs,batch_size=0.1,10,128
train_iter,test_iter =d2l.load_data_fashion_mnist(bath_size,resize=224)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())



#GoogleNet
class Inception(nn.Module):
	def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
		super().__init__(**kwargs)
		self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
		self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
		self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=1)
		self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
		self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)

	def forward(self,x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		return torch.cat((p1,p2,p3,p4),dim=1)

#批量归一化：固定小批量的均值和方差，然后学习出适合的偏移和缩放，可以加快收敛速度，但一般不会改变模型的精度
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
	if not torch.is_grad_enabled():
		X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
	else:
		assert len(X.shape) in (2,4)
		if len(X.shape)==2:
			mean=X.mean(dim=0)
			var=((X-mean)**2).mean(dim=0)
		else:
			mean =X.mean(dim=(0,2,3),keepdim=True)
			var =((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
		X_hat =(X-mean)/torch.sqrt(var+eps)
		moving_mean =momentum*moving_mean+(1.0-momentum)*mean 
		moving_var =momentum*moving_var+(1.0-momentum)*var
	Y=gamma*X_hat+beta
	return Y,moving_mean.data,moving_var.data

class BatchNorm(nn.Module):
	def __init__(self,num_features,num_dims):
		super().__init__()
		if num_dims ==2:
			shape =(1,num_features)
		else:
			shape =(1,num_features,1,1)
		self.gamma= nn.Parameter(torch.ones(shape))
		self.beta = nn.Parameter(torch.zeros(shape))
		self.moving_mean =torch.zeros(shape)
		self.moving_var =torch.ones(shape)

	def forward(self,X):
		if self.moving_mean.device!= X.device:
			self.moving_mean = self.moving_mean.to(X.device)
			self.moving_var =self.moving_var.to(X.device)
		Y,self.moving_mean,self.moving_var =batch_norm(
				X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
		return Y

net =nn.Sequential(
	nn.Conv2d(1,6,kernel_size=5),BatchNorm(6,num_dims=4),
	nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
	nn.Conv2d(6,16,kernel_size=5),BatchNorm(16,num_dims=4),
	nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
	nn.Flatten(),nn.Linear(16*4*4,120),
	BatchNorm(120,num_dims=2),nn.Sigmoid(),
	nn.Linear(120,84),BatchNorm(84,num_dims=2),
	nn.Sigmoid(),nn.Linear(84,10))

lr,num_epochs,batch_size=1.0,10,256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr)

#调包实现code:
net =nn.Sequential(
	nn.Conv2d(1,6,kernel_size=5),nn.BathNorm2d(6),
	nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
	nn.Conv2d(6,16,kernel_size=5),nn.BathNorm2d(16),
	nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
	nn.Flatten(),nn.Linear(256,120),nn.BatchNorm1d(120),
	nn.Sigmoid(),nn.Linear(120,84),nn.BatchNorm1d(84),
	nn.Sigmoid(),nn.Linear(84,10))

d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr)

#Resnet实现
import torch
from torch import nn 
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
	def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
		super().__init__()
		self.conv1 =nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
		self.conv2 =nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
		if use_1x1conv:
			self.conv3 =nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)	
		else:
			self.conv3 =None 
		self.bn1 =nn.BatchNorm2d(num_channels)
		self.bn2 =nn.BatchNorm2d(num_channels)
		self.relu =nn.ReLU(inplace=True)

	def forward(self,X):
		Y=F.relu(self.bn1(self.conv1(X)))
		Y=self.bn2(self.conv2(Y))
		if self.conv3:
			X=self.conv3(X)
		Y+=X
		return F.relu(Y)

#检验输入输出形状是否一致
blk =Residual(3,3)
X=torch.rand(4,3,6,6)
Y=blk(X)
Y.shape 
#增加通道数的同时，减半输出的高宽	
blk =Residual(3,6,use_1x1conv=True,strides=2)
blk(X).shape 

#Resnet:
b1 =nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=2),
					nn.BatchNorm2d(64),nn.ReLU(),
					nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
	blk=[]
	for i in range(num_residuals):
		if i==0 and not first_block:
			blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))
		else:
			blk.append(Residual(num_channels,num_channels))
	return blk 

b2 =nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 =nn.Sequential(*resnet_block(64,128,2))
b4 =nn.Sequential(*resnet_block(128,256,2))
b5 =nn.Sequential(*resnet_block(256,512,2))

net =nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))

X=torch.rand(1,1,224,224)
for layer in net:
	X=layer(X)
	print(layer.__class__.__name__,'Output_shape:\t',X.shape)

#数据增广
import torchvision
import torch 
from torch import nn 
from d2l import torch as d2l 

d2l.set_figsize()
img =d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

def apply(img ,aug,num_rows=2,num_cols=4,scale=1.5):
	Y=[aug(img) for _ in range(num_rows*num_cols)]
	d2l.show_images(Y,num_rows,num_cols,scale=scale)

#左右翻转图片
apply(img,torchvision.transforms.RandomHorizontalFlip())
#上下翻转图片
apply(img,torchvision.transforms.RandomVerticalFlip())
#随机剪裁
shape_aug=torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))
apply(img,shape_aug)
#随机改变图的亮度
apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))
#随机更改图像的色调
apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
#结合多种的图像增广方法
augs=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),color_aug,shape_aug])
apply(img,augs)

#使用图像增广进行训练
all_images=torchvision.datasets.CIFAR10(train=True,root="../data",download=True)
d2l.show_images([all_images[i][0] for i in range(32)],4,8,scale=0.8)

train_augs =trochvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.ToTensor()])
test_augs =torchvision.transforms,Compose([torchvision.transforms.ToTensor()])

#定义一个辅助函数帮助读取图像和应用于图像增广
def load_cifar10(is_train,augs,batch_size):
	dataset =torchvision.datasets.CIFAR10(root="../data",train=is_train,transforms=augs,download=True)
	dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train,num_workers=4)

#定义一个函数对多个GPU进行训练和评估
def train_batch_ch13(net,X,y,loss,trainer,devices):
	if isinstance(X,list):
		X=[x.to(devices[0]) for x in X] 
	else:
		X.to(devices[0])
	y=y.to(devices[0])
	net.train()
	trainer.zero_grad()
	pred=net(X)
	l=loss(pred,y)
	l.sum().backward()
	trainer.step()
	train_loss_sum=l.sum()
	train_acc_sum =d2l.accuracy(pred,y)
	return train_loss_sum,train_acc_sum
#Fintune_code:
import os 
import torch 
import torchvision
from d2l import torch as d2l
from torch import nn 

#收集热狗数据集
#图像的大小和横纵比各不相同
#数据增广
normalize =torchvision.transforms.Normlize([0.485,0.456,0.406],[0.229,0.224,0.225]) 

train_augs=torchvision.transforms.Compose([
	torchvision.transforms.RandomResizedCrop(224),
	torchvision.transforms.RandomHorizontalFlip()
	torchvision.transforms.ToTensor(),normalize])

#test_augs =.......

#定义和初始化模型
pretrained_net=torchvision.models.resnet18(pretrained=True)
pretrained_net.fc
finetune_net=torchvision.models.resnet18(pretrained=True)
fintune_net.fc=nn.Linear(fintune_net.fc.in_features,2)
nn.init.xavier_uniform_(fintune_net.fc.weight)

#目标检测和边界框
import torch 
from d2l import torch as d2l 
d2l.set_figsize()
img =d2l.plt.imread('../img/cat_dog.jpg')
d2l.plt.imshow(img)

#定义两种表示之间的转换函数
def box_corner_to_center(boxs):
	"""从左上右下到中间，宽度高度"""
	x1,x2,y1,y2 =boxs[:0],boxs[:1],boxs[:2],boxs[:3]
	cx =(x1+x2) /2
	cy =(y1+y2) /2
	w=x2-x1
	h=y2-y1
	boxs=torch.stack((cx,cy,w,h),axis=-1)
	return boxs

#定义图像中猫和狗的边框
dog_bbox=[60.0,45.0,378.0,516.0],[.....]
def bbox_to_rect(bbox,color):
	return d2l.plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)

fig= d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox,'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox),'red')

#目标检测数据集
import os 
import pandas as pd 
import torchvision
from d2l import torch as d2l

#读取香蕉检测数据集
def read_data_bananas(is_train=True):
	data_dir=d2l.download_extract('your datadir name ')
	csv_fname=os.path.join(data_dir,'bananas_train')	
	csv_data=pd.read_csv(csv_fname)
	csv_data=csv_data.set_index('img_name')
	images,targets =[],[]
	for img_name,target in csv_data.iterrows():
		images.append(torchvision.io.read_image(os.path.join(data_dir,'bananas_train')))
		targets.append(list(target))
	return images,targets 

#创建一个data_set实例
class BananasDataset(torch.utils.data.Dataset):
	def __init__(self,is_train):
		self.features,self.labels =read_data_bananas(is_train)

	def __getitem__(self,idx):
		return (self.features[idx].float(),self.labels[idx])
	def __len__(self):
		return len(self.features)
#为训练集和测试集返回两个数据加载器的实例
def load_data_bananas(batch_size):
	train_iter=torch.utils.data.DataLoader(BananasDataset(is_train),batch_size,shuffle=True)

	val_iter =torch.utils.data.DataLoader(BananasDataset(is_train=False),batch_size,shuffle=False)
	return train_iter,val_iter

#读取小批量，并打印其中的图像和标签的形状
batch_size,edge_size=32,256
train_iter,_ =loade_data_bananas(batch_size)
batch =next(iter(train_iter))
batch[0].shape,batch[1].shape  

#序列模型
import torch
from torch import nn 
from d2l import torch as d2l 

T=1000
time =torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time) +torch.normal(0,0.2,(T,))
d2l.plot(time,[x],'time','x',xlim=[1,1000],figsize=(6,3))
d2l.plt.show()
tau =4
features=torch.zeros((T-tau,tau))
for i in range(tau):
	features[:,i]=x[i:T-tau+i]
labels=x[tau:].reshape((-1,1))

batch_size,n_train=16,600
train_iter=d2l.load_array((features[:n_train],labels[:n_train]),batch_size,is_train=True)

#使用一个相当简单的结构：两个全连接层的多层感知机
def init_weights(m):
	if type(m)==nn.Linear:
		nn.init.xavier_uniform_(m.weight)

def get_net():
	net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
	net.apply(init_weights)
	return net 

loss =nn.MSELoss()

#训练模型
def train(net,train_iter,loss,epochs,lr):
	trainer=torch.optim.Adam(net.parameters(),lr)#定义优化器
	for epoch in range(epochs):
		for X,y in train_iter:
			trainer.zero_grad()
			l=loss(net(X),y)
			l.backward()
			trainer.step()
		print(f'epoch {epoch+1},'f'loss: {d2l.evaluate_loss(net,train_iter,loss)}')

net =get_net()
train(net,train_iter,loss,5,0.01)

#模型预测下一个时间步
onestep_preds=net(features)
d2l.plot(
	[time,time[tau:]],
	[x.detach().numpy(),onestep_preds.detach().numpy()],'time','x',
	legend=['data','1-step preds'],xlim=[1,1000],figsize=(6,3))

#进行多步预测
multistep_preds=torch.zeros(T)
multistep_preds[:n_train+tau]=x[:n_train+tau]
for i in range(n_train+tau,T):
	multistep_preds[i]=net(multistep_preds[i-tau,i].reshape((-1,1)))
d2l.plot([time,time[tau:],time[n_train+tau:]],[x.detach().numpy(),onestep_preds.detach().numpy(),multistep_preds[n_train+tau:].detach().numpy()],'time','x',legend=['data','1-step preds','multistep preds'],xlim=[1,1000],figsize=(6,3))	

#文本预处理
import collections
import re 
from d2l import torch as d2l

d2l.DATA_HUB['time_machine']=(d2l.DATA_URL+'time_machine.txt')

def read_time_machine():
	"""load time_machine dataset into a list of text lines"""
	with open(d2l.download('time_machine'),'r') as f:
		lines = f.readlines()
	return [re.sub('[^A-Za-z]+','',line).strip().lower() for line in lines]
#处理的相当暴力
lines =read_time_machine()

def tokenize(lines,token='word'):
	if token=='word':
		return [line.split() for line in lines]
	elif token=='char':
		return [list(line) for line in lines]
	else:
		print('错误：未知令牌类型'+token) 

#语言模型和数据集
import torch
import random
from d2l import torch as d2l
tokens =d2l.tokenize(d2l.read_time_machine())
corpus =[token for line in tokens for token in line]
vocab =d2l.Vocab(corpus)
vocab.token_freqs[:10]
freqs=[freq for token ,freq in vocab.token_freqs]
d2l.plot(freqs,xlabel='token: x',ylabel='frequency:n(x)',xscale='log',yscale='log')

#其他的词元组合
bigram_tokens=[pair for pair in zip(corpus[:-1],corpus[1:])]
bigram_vocab=d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
#三元的组合
trigram_tokens=[triple for triple in zip(corpus[:-2],corpus[1:-1],corpus[2:])]
trigram_vocab=d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]

def seq_data_iter_random(corpus,batch_size,num_steps):
	"""使用随机抽样生成一个小批量的子序列"""
	corpus=corpus[random.ranint(0,num_steps-1):]
	num_subseqs= (len(corpus)-1)//num_steps
	initial_indices=list(range(0,num_subseqs*num_steps,num_steps))
	random.shuffle(initial_indices)

	def data(pos):
		return corpus[pos:pos+num_steps]

	num_batches=num_subseqs//batch_size
	for i in range(0,batch_size*num_batches,batch_size):
		initial_indices_per_batch =initial_indices[i:i+batch_size]
		X=[data(j) for j in initial_indices_per_batch]
		Y=[data(j+1) for j in initial_indices_per_batch]
		yield torch.tensor(X),torch.tensor(Y)


#循环神经网络的从零实现
import torch
import math
from torch import nn 
from d2l import torch as d2l
from torch.nn import functional as F 
batch_size,num_steps =32,35
train_iter,vocab =d2l.load_data_time_machine(batch_size,num_steps)

#独热编码
F.one_hot(torch.tensor([0,2]),len(vocab))
#小批量数据形状是（批量大小，时间步数）
X=torch.arange(10).reshape((2,5))
F.one_hot(X.T,28).shape   

#初始化循环神经网络模型的模型参数
def get_params(vocab_size,num_hiddens,device):
	num_inputs=num_outputs=vocab_size

	def normal(shape):
		return torch.randn(size=shape,device=device)*0.01

	W_xh=normal((num_inputs,num_hiddens))
	W_hh=normal((num_hiddens,num_hiddens))
	b_h =torch.zeros(num_hiddens,device=device)
	W_hq=normal((num_hiddens,num_outputs))
	b_q=torch.zeros(num_outputs,device=device)
	params=[W_xh,W_hh,b_h,W_hq,b_q]
	for param in params:
		param.requires_grad_(True)
		return params 

def init_rnn_state(batch_size,num_hiddens,device):
	return (torch.zeros((batch_size,num_hiddens),device=device),)

#下面的RNN定义了在一个时间步内的隐藏状态的计算和输出
def rnn(inputs,state,params):
	W_xh,W_hh,b_h,W_hq,b_q=params 
	H, =state 
	outputs=[]
	for X in inputs:
		H=torch.tanh(torch.mm(X,W_xh)+ torch.mm(H,W_hh) +b_h)
		Y=torch.mm(H,W_hq)+b_q
		outputs.append(Y)
	return torch.cat(outputs,dim=0),(H,)
 
#创建一个类来包装这些函数
class RNNModelScratch:
	def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
		self.vocab_size,self.num_hiddens =vocab_size,num_hiddens
		self.params =get_params(vocab_size,num_hiddens,device)
		self.init_state,self.forward_fn =init_state,forward_fn

	def __call__(self,X,state):
		X=F.one_hot(X.T,self.vocab_size).type(torch.float32)
		return self.forward_fn(X,state,self.params)

	def begin_state(self,batch_size,device):
		return self.init_state(batch_size,num_hiddens,device)

#检查输出是否具有正确的形状
num_hiddens=512
net =RNNModelScratch(len(vocab),num_hiddens,d2l.try_gpu(),get_params,init_state,rnn)
state =net.begin_state(X.shape[0],d2l.try_gpu())
Y,new_state =net(X,state)
Y.shape,len(new_state)

def predict_ch8(prefix,num_preds,net,vocab,device):
	state =net.begin_state(batch_size=1,device=device)
	outputs =[vocab[prefix[0]]]
	get_input =lambda: torch.tensor([outputs[-1]],device=device).reshape((1,1))
	for y in prefix[1:]:
		_,state =net(get_input(),state)
		outputs.append(vocab[y])
	for _ in range(num_preds):
		y,state =net(get_input(),state)
		outputs.append(int(y.argmax(dim=1).reshape(1)))
	return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ',10,net,vocab,d2l.try_gpu())

#梯度剪裁
def grad_clipping(net,theta):
	"""剪裁梯度"""
	if isinstance(net,nn.Module):
		params =[p for p in net.parameters() if p.requires_grad_]
	else:
		params =net.params 
	norm =torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
	if norm >theta:
		for param in params:
			param.grad[:]*=theta/norm

def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
	state,timer=None,d2l.Timer()
	metric =d2l.Accumulator(2)
	for X,Y in train_iter:
		if state is None or use_random_iter:
			state=net.begin_state(batch_size=X.shape[0],device=device)
		else:
			if isinstance(net,nn.Module):
				state.detach_()
			else:
				for s in state:
					s.detatch_()
		y=Y.T.reshape(-1)
		X,y =X.to(device),y.to(device)
		y_hat,state =net(X,state)
		l=loss(y_hat,y.long()).mean()
		if isinstance(updater,torch.optim.Optimizer):
			updater.zero_grad()
			l.backward()
			grad_clipping(net,1)
			updater.step()
		else:
			l.backward()
			grad_clipping(net,1)
			updater(batch_size=1)
		 metric.add(l*y.numel(),y.numel())
	return math.exp(metric[0]/metric[1])

#循环神经网络的简洁实现
import torch 
from torch import nn 
from d2l import torch as d2l 
from torch.nn import functional as F 
batch_size,num_steps =32,35
train_iter,vocab =d2l.load_data_time_machine(batch_size,num_steps)

#定义模型
num_hiddens =256
rnn_layer =nn.RNN(len(vocab),num_hiddens)
state =torch.zeros((1,batch_size,num_hiddens))

#通过一个隐藏状态和一个输入，来更新隐藏状态
X=torch.rand(size=(num_steps,batch_size,len(vocab)))
Y,state_new =rnn_layer(X,state)
Y.shape,state_new.shape 

#门控循环单元GRU

#初始化模型参数
def get_params(vocab_size,num_hiddens,device):
	num_inputs =num_outputs =vocab_size
	def normal(shape):
		return torch.randn(shape)*0.01
	def three():
		return (normal((num_inputs,num_hiddens)),
				normal((num_hiddens,num_hiddens)),
				torch.zeros(num_hiddens,device=device))
	W_xz,W_hz,b_z =three()
	W_xr,W_hr,b_r =three()
	W_xh,W_hh,b_h =three()
	W_hq =normal((num_hiddens,num_outputs))
	b_q =torch.zeros(num_outputs)
	params =[W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]
	for param in params:
		param.requires_grad_(True)
	return params 

def init_gru_state(batch_size,num_hiddens):
	return (torch.zeros(batch_size,num_hiddens),)

def gru(inputs,state,params):
	W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q =params 
	H, =state 
	outputs =[]
	for X in inputs:
		Z=torch.sigmoid((X@ W_xz) + (H @ W_hz) + b_z)
		R=torch.sigmoid((X@ W_xr) + (H @ W_hr) + b_r)
		H_tilda =torch.tanh((X@W_xh) +((R*H)@ W_hh) +b_h)
		H = Z*H +(1-Z)*H_tilda 
		Y =H@ W_hq +b_q	
		outputs.append(Y)
	return torch.cat(outputs,dim=0),(H,)
#GRU简洁实现
num_inputs =vocab_size
gru_layer =nn.GRU(num_inputs,num_hiddens)
model = d2l.RNNModel(gru_layer,len(vocab))
model =model.to(device)
d2l.trian_ch8(model,train_iter,vocab,lr,num_epochs,device)

class RNNModel(nn.Block):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

#LSTM的实现
def get_lstm_params(vocab_size,num_hiddens):
	num_inputs =num_outputs =vocab_size

	def normal(shape):
		return np.random.normal(scale=0.01,size =shape)
	def three():
		return (normal((num_inputs,num_hiddens)),
			normal((num_hiddens,num_hiddens)),np.zeros(num_hiddens))

	W_xi,W_hi,b_i =three() #输入门
	W_xf,W_hf,b_f =three() #遗忘门
	W_xo,W_ho,b_o =three() #输出门的参数
	W_xc,W_hc,b_c =three() #候选记忆元参数
	#输出层参数
	W_hq =normal((num_hiddens,num_outputs))
	b_q  =np.zeros(num_outputs)
	#附加梯度
	params =[W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]
	for param in params:
		param.requires_grad_()
	return params 

#定义模型
def init_lstm_state(batch_size,num_hiddens):
	return (np.zeros(batch_size,num_hiddens),np.zeros(batch_size,num_hiddens))

def lstm(inputs,state,params):
	[W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q] =params 
	(H,C) =state 
	outputs =[]
	for X in inputs:
		I =torch.Sigmoid(X@W_xi +H@W_hi+b_i)
		F =torch.Sigmoid(X@W_xf +H@W_hf+b_f)
		O =torch.Sigmoid(X@W_xo +H@W_ho+b_o)	
		C_tilda =np.tanh(X@W_xc+H@W_hc +b_c)
		C =F*C +I*C_tilda
		H=O*np.tanh(C)
		Y=H@W_hq+b_q
		outputs.append(Y)
	return torch.cat(outputs,dim=0) ,(H,C)

#LSTM的简洁实现
lstm_layer =nn.LSTM(num_inputs,num_hiddens)
model =d2l.RNNModel(lstm_layer,len(vocab_size))
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs)


# def ConvMixer(h,depth,kernel_size=9,patch_size =7,n_classes=1000):
# 	Seq,ActBn =nn.Sequential(),lambda x: Seq(x, nn.GELU() ,nn.BatchNorm2d())
# 	Residual =type('Residual',(Seq,),{'forward ' : lambda self, x :self[0](x)+x})
# 	return Seq(ActBn(nn.Conv2d(3,h,patch_size,stride =patch_size)),)

#深度循环神经网络的实现
import torch 
from torch import nn 
from d2l import torch  as d2l
batch_size ,num_steps =32,35

#通过num_layers 来设置隐藏层的数目
vocab_size,num_hiddens,num_layers =len(vocab),256,2
num_inputs =vocab_size
device =mps 
lstm_layer =nn.LSTM(num_inputs,num_hiddens,num_layers)
model =d2l.RNNModel(lstm_layer,len(vocab_size)) 
model =model.to(device) 

#训练
num_epochs,lr =500,2
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)


#合并编码器和解码器
class EncodeDecode(nn.Module):
	def __init__(self,encode,decode):
		super(self).__init__()
		self.encoder =encode 
		self.decoder =decode 

	def forward(self,encode_X,decode_X):
		encode_outputs =self.encoder(encode_X)
		decode_state =self.decoder.iniit_state(encode_outputs)
		return self.decoder(decode_X,decode_state)
#实现循环神经网络的编码器
class Seq2SeqEncoder(d2l.Encoder):
	def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
		super(self).__init__()
		self.embedding =nn.Embedding(vocab_size,embed_size)
		self.rnn =nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

	def forward(self,X):
		X=self.embedding(X)
		X=X.permute(1,0,2)
		output,state =self.rnn(X)
		return output,state 
		