import pickle
from network import TwoLayerNet
from data_process import processImg
import numpy as np


def predict(img_path):
    '''
    img_path: 图片文件路径
    ----------------
    s: 以字符串形式返回验证码
    '''

    (im1,im2,im3,im4),(a,b,c,d)=processImg(img_path)

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)

    network = TwoLayerNet(input_size=260, hidden_size=60, output_size=36,params=params)


    ims=np.array([im1,im2,im3,im4])
    x=network.predict(ims)
    index=np.argmax(x,axis=1)

    word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    s=''
    for w in index:
        s+=word[w]

    return s

def main():
    print(predict('0ak7.gif'))

if __name__ == '__main__':
    main()

