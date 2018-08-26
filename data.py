from PIL import Image
import numpy as np
import os
import pickle

def clearNoise(data):
    '''
    去除图片噪点
    '''
    height=data.shape[0]
    width=data.shape[1]
    for i in range(height):
        for j in range(width):
            if i==0 or i== height-1 or j==0 or j== width-1:
                data [i][j]=1
                continue
            if data[i][j]==0:
                num=0
                for da in data[i-1:i+2,j-1:j+2]:
                    if da[0]>0 :
                        num+=1
                    if da[1]>0 :
                        num+=1
                    if da[2]>0 :
                        num+=1
                if num>4:
                    data[i][j]=1

def processImg(inputfile):
    '''
    将图片黑白化
    返回numpy数组
    '''
    image = Image.open(inputfile)
    img=image.convert("L")
    data=img.getdata()
    da=np.array(data,np.int32)
    da[da<=170]=0
    da[da>170]=1
    da=da.reshape((27,72))#图片原始尺寸为(60,180)
    clearNoise(da)
    clearNoise(da)
    clearNoise(da)
    im=da[0:20,]
    im1=im[:,4:17].flatten().tolist()
    im2=im[:,17:30].flatten().tolist()
    im3=im[:,30:43].flatten().tolist()
    im4=im[:,43:56].flatten().tolist()
    l=list(inputfile.split('/')[-1].split('.')[0]) 

    return (im1,im2,im3,im4),(l[0],l[1],l[2],l[3])


    # img1=Image.fromarray(da)
    # img1=img1.convert('RGB')
    # img1.save(outputfile)
def image2one_hot(pathname):
    '''
    将训练数据集保存为 pickle 文件
    '''
    filename_list=os.listdir(pathname) 
    x_train=[]
    t_train=[]
    word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for filename in filename_list:
        (im1,im2,im3,im4),(a,b,c,d)=processImg(pathname+'/'+filename)
        x_train.append(im1)
        t1=[0 for i in range(36)]
        t1[word.index(a)]=1
        t_train.append(t1)

        x_train.append(im2)
        t2=[0 for i in range(36)]
        t2[word.index(b)]=1
        t_train.append(t2)

        x_train.append(im3)
        t3=[0 for i in range(36)]
        t3[word.index(c)]=1
        t_train.append(t3)

        x_train.append(im4)
        t4=[0 for i in range(36)]
        t4[word.index(d)]=1
        t_train.append(t4)

    print(np.array(x_train).shape)
    print(np.array(t_train).shape)

    with open('x_train.pickle', 'wb') as f:
        pickle.dump(x_train, f, -1)

    with open('t_train.pickle', 'wb') as f:
        pickle.dump(t_train, f, -1)
    


if __name__ == '__main__':
     image2one_hot('验证码/未完成')

    