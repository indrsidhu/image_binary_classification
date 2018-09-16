import numpy as np
import h5py
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s

def predict(w,b,X):
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    return A

def gradient(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    
    cost = (1/m) * np.sum((Y-A)**2)
    print(cost)
    
    #dw = -(2/m) * np.sum( np.dot(X, (Y-A).T) )
    #db = -(2/m) * np.sum(Y-A)
    
    dw = (1./m)*np.dot(X,((A-Y).T))
    db = (1./m)*np.sum(A-Y, axis=1)    
    
    return [dw,db]
    

def gradient_decent(w,b,X,Y,num_itrations,learning_rate):
    for i in range(num_itrations):
        dw, db = gradient(w,b,X,Y)
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
    return [w,b]

def model(w,b,X,Y,num_itrations,learning_rate,test_set_x,test_set_y):
    w, b = gradient_decent(w,b,X,Y,num_itrations,learning_rate)

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction = predict(w, b, X)
    Y_prediction_test = predict(w, b, test_set_x)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
    
    return [w,b]

train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")

train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # input variable1s
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # output variable1s

test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # input variable1s
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # output variable1s


#plt.imshow(train_set_x_orig[4])

# Flattern image data it is about making input variable x to (nx,m)
# where nx are number of inputs for one image example and m are number of examples
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_set_y = train_set_y_orig.reshape(train_set_y_orig.shape[0],-1).T

test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
test_set_y = test_set_y_orig.reshape(test_set_y_orig.shape[0],-1).T

#Let's standardize our inputs to make all inputs between 0 to 1 range eg, 0.45, 0.50
train_set_x = train_set_x / 255
test_set_x = test_set_x / 255

# w should be equal to number of inputs number of inputs in each images are 12288
w = np.zeros(shape=(train_set_x.shape[0],1))
b = 0
num_itrations = 2000
learning_rate = 0.04
w,b = model(w,b,train_set_x,train_set_y,num_itrations,learning_rate,test_set_x,test_set_y)

#=========== TEST =========== ==>
'''
index = 20
test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")
test_set_x = np.array(test_dataset["test_set_x"][:]) # input variable1s
plt.imshow(test_set_x[index])

test_image = test_set_x[index]

# flatteren input
test_image = test_image.reshape(1,64*64*3).T

test_image = test_image / 255
my_predicted_image = predict(w,b,test_image)
print("y = " + str(np.squeeze(my_predicted_image)))
'''