import os
from run import predict

def main(file_path):
    img_list=os.listdir(file_path)

    sum_right=0
    for img in img_list:
        s=predict(file_path+'/'+img)
        s1=img.split('.')[0]
        if s==s1:
            sum_right+=1

    acc=sum_right/len(img_list)
    return acc


if __name__ == '__main__':
    acc=main('验证码/完成')
    print('acc:'+str(acc))