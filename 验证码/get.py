import requests


indetifyUrl='http://218.64.216.236/CheckCode.aspx'
for i in range(1000):
    filename=i
    content=requests.get(indetifyUrl).content
    with open('未完成/'+str(filename)+'.gif','wb') as f:
        f.write(content)
