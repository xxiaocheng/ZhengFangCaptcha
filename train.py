from network import TwoLayerNet
import pickle
import numpy as np

with open('train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)


with open('test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)

x_train=np.array(train_data[0])
t_train=np.array(train_data[1])
x_test=np.array(test_data[0])
t_test=np.array(test_data[1])

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

network = TwoLayerNet(input_size=260, hidden_size=60, output_size=36)

iters_num = 100000
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(int(train_size / batch_size), 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)   

with open('params.pickle', 'wb') as f:
        pickle.dump(network.params, f, -1)