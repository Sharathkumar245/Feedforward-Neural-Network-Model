# def tanh_function(a_s):
#   return (np.power(np.exp(a_s),2)-1)/(np.power(np.exp(a_s),2)+1)
# def tanh_function_derivative(a_s):
#   fx=tanh_function(a_s)
#   return (1-np.power(fx,2))
# def reLU_function(a_s):
#   return np.maximum(0,a_s)
# def reLU_function_derivative(a_s):
#   fx=reLU_function(a_s)
#   for i in range(len(fx)):
#     if(fx[i]>0):
#       fx[i]=1
#     else: fx[i]=0
#   return fx



# name_of_loss_function="cross_entropy"
# name_of_optimization_function="sgd"
# name_of_weight_function="random"
# max_epochs=5
# weight_decay_value=0.0005
# batch_size_value=32
# name_of_weight_function="sigmoid"
# eta=0.00001
# beta=0.5
# beta1=0.9
# beta2=0.999
# no_of_layers=3
# no_of_neurons_at_each_layer=128
# def initializing_w_and_b_function(no_of_layers,name_of_weight_function,no_of_neurons_at_each_layer):
#   if(name_of_weight_function=="random"):
#     initialize_w_and_b(no_of_layers,no_of_neurons_per_layer)
#   if(name_of_weight_function=="xavier"):
#     weight_intialization_Xavier(no_of_layers,no_of_neurons_per_layer)
# def loss_function(h_s,w_s,b_s,images_to_train,labels_to_train,name_of_loss_function):
#   # if(name_of_loss_function="cross_entropy"):
#   #   cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
#   # if(name_of_loss_function="mse"):
#   #   mean_squared_error(h_s,w_s,b_s,images_to_train,labels_to_train)
# def activation_function(name_of_activation_function):
#   # if(name_of_activation_function="sigmoid"):
#   #   sigmoid_function()
#   # if(name_of_activation_function="tanh"):
#   #   tanh_function()
#   # if(name_of_activation_function="ReLU"):
  #   reLU_function()
# !pip install wandb
import argparse
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist,mnist
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import wandb
from wandb.keras import WandbCallback
import socket
socket.setdefaulttimeout(30)  # Set timeout to 30 seconds (adjust as needed)
wandb.login()
wandb.init(project='dL_assignment_1')

# (images_to_train, labels_to_train), (test_images,test_labels) = fashion_mnist.load_data()
# images_to_train, labels_to_train=shuffle(images_to_train,labels_to_train)
# images_to_train=images_to_train/255.0
# test_images=test_images/255.0

def plot_samples(images, labels, class_names):
    plt.figure(figsize=(10, 5))
    # for i in range(len(class_names)):
    i=0
    while i<len(class_names):
        # Find the index of the first image with the current class label
        idx = np.where(labels == i)[0][0]
        image = images[idx]
        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
        i+=1
    plt.tight_layout()
    plt.show()



def ploting_sample_images():
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Function to plot one sample image for each class
  plot_samples(images_to_train, labels_to_train, class_names)



def initialize_w_and_b_random(no_of_layers,no_neurons_per_layer):
  W_s=[]
  B_s=[]
  W_s.append((np.random.randn(no_neurons_per_layer,784)))
  B_s.append((np.random.randn(no_neurons_per_layer,1)))
  for i in range(1,no_of_layers):
    W_s.append((np.random.randn(no_neurons_per_layer,no_neurons_per_layer)))
    B_s.append((np.random.randn(no_neurons_per_layer,1)))
  W_s.append((np.random.randn(10,no_neurons_per_layer)))
  B_s.append((np.random.randn(10,1)))
  # print(W_s[0].shape,W_s[-1].shape)
  # print(B_s[0].shape,B_s[-1].shape)
  return W_s,B_s


def weight_intialization_Xavier(no_of_layers,no_neurons_per_layer):
  w_s_xavier=[]
  b_s_xavier=[]
  w_s_xavier.append((np.random.randn(no_neurons_per_layer,784)))
  b_s_xavier.append(np.zeros((no_neurons_per_layer,1)))
  for i in range(1,no_of_layers):
    w_s_xavier.append((np.random.randn(no_neurons_per_layer,no_neurons_per_layer)))
    b_s_xavier.append(np.zeros((no_neurons_per_layer,1)))
  w_s_xavier.append((np.random.randn(10,no_neurons_per_layer)))
  b_s_xavier.append(np.zeros((10,1)))
  # print(w_s_xavier[i].shape for i in range(len(w_s_xavier)))
  # print(b_s_xavier[i].shape for i in range(len(b_s_xavier)))
  return w_s_xavier,b_s_xavier



def intitalize_w_and_b(name_of_function,no_of_layers,no_neurons_per_layer):
  if(name_of_function=="random"):
    return initialize_w_and_b_random(no_of_layers,no_neurons_per_layer)
  if(name_of_function=="xavier"):
    return weight_intialization_Xavier(no_of_layers,no_neurons_per_layer)



def find_loss(h_s,labelled_image,name_of_loss_function):
  if(name_of_loss_function=="cross_entropy"):
    # return -np.log(h_s[np.argmax(labelled_image)])
    # return -np.sum(labelled_image*np.log(h_s))
    # small_constant=1e-15
    # h_s=np.clip(h_s,small_constant,1-small_constant)
    # return -np.sum(labelled_image*np.log(h_s))
    return -np.log(h_s[np.argmax(labelled_image)]+(1e-5))
  if(name_of_loss_function=="mean_squared_error"):
    return np.sum(np.power((h_s[-1]-labelled_image),2))


def sigmoid_function(a_s):
  # return np.exp(a_s)/(1+np.exp(a_s))
  z_clipped=np.clip(a_s,-100,100)
  return 1/(1+np.exp(-z_clipped))

def tanh_function(a_s):
  # return (np.power(np.exp(a_s),2)-1)/(np.power(np.exp(a_s),2)+1)
  # return (np.exp(a_s) - np.exp(-a_s)) / (np.exp(a_s) + np.exp(-a_s))
  # z = (np.exp(2*a_s) - 1) / (np.exp(2*a_s) + 1)
  # z_clipped=np.clip(z,-1,1)
  # return z_clipped
  z_clipped=np.clip(a_s,-50,50)
  return np.tanh(z_clipped)

def reLU_function(a_s):
  return np.maximum(0,a_s)


def activation_function(a_s,name_of_activation):
  if(name_of_activation=="sigmoid"):
    return sigmoid_function(a_s)
  if(name_of_activation=="tanh"):
    return tanh_function(a_s)
  if(name_of_activation=="ReLU"):
    return reLU_function(a_s)
  if(name_of_activation=="identiy"):
    return a_s

def sigmoid_derivative(a_s):
  fx=sigmoid_function(a_s)
  return fx*(1-fx)

def tanh_function_derivative(a_s):
  fx=tanh_function(a_s)
  return (1-np.power(fx,2))
  # return 1-np.fx**2

def reLU_function_derivative(a_s):
  fx=reLU_function(a_s)
  for i in range(len(fx)):
    if(fx[i]>0):
      fx[i]=1
    else: fx[i]=0
  return fx


def derivative_function(a_s,name_of_activation):
  if(name_of_activation=="sigmoid"):
    return sigmoid_derivative(a_s)
  if(name_of_activation=="tanh"):
    return tanh_function_derivative(a_s)
  if(name_of_activation=="ReLU"):
    return reLU_function_derivative(a_s)
  if(name_of_activation=="identity"):
    return np.ones(a_s.shape)

def soft_max_function(a_s):
  exps = np.exp(a_s - np.max(a_s))
  # print((exps/np.sum(exps)).shape)
  return exps / np.sum(exps)
  # exps=np.exp(a_s)
  # sum_of_a_s=np.sum(exps)
  # return exps/sum_of_a_s


def calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers):
  total_loss=0
  total_accuracy=0
  training_limit=(len(labels_to_train))*(1-validation_split)
  # print(training_limit)
  for i in range(int(training_limit)):
    flattened_image=images_to_train[i].flatten().reshape(784,1)
    a_s,h_s=forward_propagation(w_s,b_s,flattened_image,no_of_layers,images_to_train,name_of_activation)
    labelled_image=np.zeros((10,1))
    labelled_image[labels_to_train[i]]=1
    total_loss+=find_loss(h_s[-1],labelled_image,name_of_loss_function)
    if(np.argmax(h_s[-1])==labels_to_train[i]):
      total_accuracy+=1
  return (total_loss/training_limit),(total_accuracy/training_limit)*100



def calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers):
  validation_loss=0
  validation_accuracy=0
  validation_limit=(len(labels_to_train))*(1-validation_split)
  for i in range(int(validation_limit),int(len(labels_to_train))):
    flattened_image=images_to_train[i].flatten().reshape(784,1)
    a_s,h_s=forward_propagation(w_s,b_s,flattened_image,no_of_layers,images_to_train,name_of_activation)
    labelled_image=np.zeros((10,1))
    labelled_image[labels_to_train[i]]=1
    validation_loss+=find_loss(h_s[-1],labelled_image,name_of_loss_function)
    if(np.argmax(h_s[-1])==labels_to_train[i]):
      validation_accuracy+=1
  return (validation_loss/(len(labels_to_train)*validation_split)),(validation_accuracy/(len(labels_to_train)*validation_split))*100



def according_to_loss_gradient(label_image,h_s,name_of_loss_function):
  if(name_of_derivation_function=="cross_entropy"):
    return -(label_image-h_s[-1])
  if(name_of_loss_function=="mean_squared_error"):
    # x=np.multiply(h_s[-1],np.subtract(1,h_s[-1]))
    # return np.multiply((-2*np.subtract(h_s[-1],label_image)),np.multiply(h_s[-1],np.subtract(1,h_s[-1])))
    return (h_s[-1]-label_image)*(h_s[-1])*(1-h_s[-1])




def forward_propagation(w_s,b_s,flattened_image,no_of_layers,images_to_train,name_of_activation):
  a_s=[]
  h_s=[]
  h_s.append(flattened_image)
  for i in range(no_of_layers):
    # a_s.append(np.matmul(w_s[i],flattened_image)+b_s[i])
    a=np.matmul(w_s[i],flattened_image)+b_s[i]
    a_clipped=np.clip(a,-1e10,1e10)
    a_s.append(a_clipped)
    h_s.append(activation_function(a_s[i],name_of_activation))
    flattened_image=h_s[-1]
  a_s.append(np.dot(w_s[-1],flattened_image)+b_s[-1])
  h_s.append(soft_max_function(a_s[-1]))
  # print(h_s[-1].shape)
  # print(a_s[0].shape,a_s[-1].shape)
  # print(h_s[0].shape,h_s[-1].shape)
  return a_s,h_s


def backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function):
  theta_w=[]
  theta_b=[]
  # nabla_a_last=according_to_loss_gradient(label_image,h_s,name_of_loss_function)
  # nabla_a_last=-(label_image-h_s[-1])
  nabla_a_last=np.zeros((10,1))
  if(name_of_loss_function=='cross_entropy'):
    nabla_a_last=-(label_image-h_s[-1])
  if(name_of_loss_function=='mean_square_error'):
    nabla_a_last=(h_s[-1]-label_image)*(h_s[-1])*(1-h_s[-1])
  for k in range(no_of_layers,-1,-1):
    nabla_w_last=np.matmul(nabla_a_last,h_s[k].T)
    theta_w.append(nabla_w_last)
    nabla_b_last=nabla_a_last
    theta_b.append(nabla_b_last)
    if(k==0):
      break
    nabla_h_last=np.matmul(w_s[k].T,nabla_a_last)
    # print(nabla_a_last.shape)
    # print(w_s[k].T.shape,nabla_a_last.shape)
    nabla_a_last=np.multiply(nabla_h_last,derivative_function(a_s[k-1],name_of_activation))
  return theta_w,theta_b


  # nabla_a_last=-(label_image-h_s[-1])
  # for k in range(no_of_layers-1,-1,-1):
  #   nabla_w_last=np.matmul(nabla_a_last,h_s[k+1].T)
  #   theta_w.append(nabla_w_last)
  #   nabla_b_last=nabla_a_last
  #   theta_b.append(nabla_b_last)
  #   nabla_h_last=np.matmul(w_s[k].T,nabla_a_last)
  #   # print(w_s[k].T.shape,nabla_a_last.shape)
  #   nabla_a_last=np.multiply(nabla_h_last,derivative_function(a_s[k],name_of_activation))
  # nabla_w_last=np.matmul(nabla_a_last,h_s[0].T)
  # theta_w.append(nabla_w_last)
  # nabla_b_last=nabla_a_last
  # theta_b.append(nabla_b_last)
  # return theta_w,theta_b



def stochastic_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers):
  batch_size=1
  for iterations in range(max_epochs):
    total_accuracy=0
    total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if (i + 1) % batch_size == 0:
          w_s = [w - (eta * grad) -(w*weight_decay) for w, grad in zip(w_s, dw)]
          b_s = [b- eta * grad for b, grad in zip(b_s, db)]
          dw=[np.zeros_like(every_ele) for every_ele in w_s]
          db=[np.zeros_like(every_ele) for every_ele in b_s]
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # loss=cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})



def momentum_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers):
  last_uw=[np.zeros_like(every_ele) for every_ele in w_s]
  last_ub=[np.zeros_like(every_ele) for every_ele in b_s]
  for iterations in range(max_epochs):
    # total_accuracy=0
    # total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    points_touched_till_now=0
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      # total_loss+=find_loss(h_s[-1],label_image,)
      # if(np.argmax(h_s[-1])==labels_to_train[i]):
        # total_accuracy+=1
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if (i + 1) % mini_batch_size == 0:
          u_w = [beta * uw + grad for uw, grad in zip(last_uw, dw)]
          u_b = [beta * ub + grad for ub, grad in zip(last_ub, db)]
          w_s = [w*(1-weight_decay) - eta * uw for w, uw in zip(w_s, u_w)]
          b_s = [b - eta * ub for b, ub in zip(b_s, u_b)]
          last_uw = u_w
          last_ub = u_b
          dw=[np.zeros_like(every_ele) for every_ele in w_s]
          db=[np.zeros_like(every_ele) for every_ele in b_s]
    # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})
  # return w_s,b_s




def nesterov_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers):
  last_uw=[np.zeros_like(every_ele) for every_ele in w_s]
  last_ub=[np.zeros_like(every_ele) for every_ele in b_s]
  for iterations in range(max_epochs):
    # total_accuracy=0
    # total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    points_touched_till_now=0
    u_w = [beta * uw for uw in last_uw]
    u_b = [beta * ub for ub in last_ub]
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      # total_loss+=find_loss(h_s[-1],label_image)
      # if(np.argmax(h_s[-1])==labels_to_train[i]):
        # total_accuracy+=1
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,[w-u for w, u in zip(w_s,u_w)],[b-u for b,u in zip(b_s,u_b)],no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if (i + 1) % mini_batch_size == 0:
        u_w = [beta * uw + grad for uw, grad in zip(last_uw, dw)]
        u_b = [beta * ub + grad for ub, grad in zip(last_ub, db)]
        w_s = [w*(1-weight_decay) - eta * uw for w, uw in zip(w_s, u_w)]
        b_s = [b - eta * ub for b, ub in zip(b_s, u_b)]
        last_uw = u_w
        last_ub = u_b
        dw=[np.zeros_like(every_ele) for every_ele in w_s]
        db=[np.zeros_like(every_ele) for every_ele in b_s]
    # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})




def rmsprop_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers):
  last_uw=[np.zeros_like(every_ele) for every_ele in w_s]
  last_ub=[np.zeros_like(every_ele) for every_ele in b_s]
  for iterations in range(max_epochs):
    # total_accuracy=0
    # total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    points_touched_till_now=0
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      # total_loss+=find_loss(h_s[-1],label_image)
      # if(np.argmax(h_s[-1])==labels_to_train[i]):
        # total_accuracy+=1
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if((i+1)%mini_batch_size==0):
        u_w = [beta * uw + (1-beta)*(grad**2) for uw, grad in zip(last_uw, dw)]
        u_b = [beta * ub + (1-beta)*(grad**2) for ub, grad in zip(last_ub, db)]
        w_s = [w*(1-weight_decay) - eta * x / (np.sqrt(uw + eps)) for w, x, uw in zip(w_s, dw, u_w)]
        b_s = [b - eta * x / (np.sqrt(ub + eps)) for b, x, ub in zip(b_s, db, u_b)]
        last_uw=u_w
        last_ub=u_b
        dw=[np.zeros_like(every_ele) for every_ele in w_s]
        db=[np.zeros_like(every_ele) for every_ele in b_s]
    # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})



def adam_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,beta1,beta2,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers,mini_batch_size):
  m_w=[np.zeros_like(every_ele) for every_ele in w_s]
  m_b=[np.zeros_like(every_ele) for every_ele in b_s]
  v_w=[np.zeros_like(every_ele) for every_ele in w_s]
  v_b=[np.zeros_like(every_ele) for every_ele in b_s]
  for iterations in range(max_epochs):
    # total_accuracy=0
    # total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    points_touched_till_now=0
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      # total_loss+=find_loss(h_s[-1],label_image)
      # if(np.argmax(h_s[-1])==labels_to_train[i]):
      #   total_accuracy+=1
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if((i+1)%mini_batch_size==0):
        m_w = [beta1* mw + (1-beta1)*(grad) for mw, grad in zip(m_w, dw)]
        m_b = [beta1* mb + (1-beta1)*(grad) for mb, grad in zip(m_b, db)]
        m_w = [beta2* vw + (1-beta2)*(grad**2) for vw, grad in zip(v_w, dw)]
        m_b = [beta2* vb + (1-beta2)*(grad**2) for vb, grad in zip(v_b, db)]


        m_w_hat=[m/(1-np.power(beta1,i+1)) for m in m_w]
        m_b_hat=[m/(1-np.power(beta1,i+1)) for m in m_b]
        v_w_hat=[v/(1-np.power(beta2,i+1)) for v in v_w]
        v_b_hat=[v/(1-np.power(beta2,i+1)) for v in v_b]
        for k in range(len(w_s)):
          for j in range(len(w_s[k])):
              w_s[k][j] =w_s[k][j]*(1-weight_decay)-(eta * m_w_hat[k][j]) / (np.sqrt(v_w_hat[k][j]) + eps)
              b_s[k][j]-= (eta * m_b_hat[k][j]) / (np.sqrt(v_b_hat[k][j]) + eps)
        dw=[np.zeros_like(every_ele) for every_ele in w_s]
        db=[np.zeros_like(every_ele) for every_ele in b_s]
    # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})



def nadam_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,beta1,beta2,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers,mini_batch_size):
  m_w=[np.zeros_like(every_ele) for every_ele in w_s]
  m_b=[np.zeros_like(every_ele) for every_ele in b_s]
  v_w=[np.zeros_like(every_ele) for every_ele in w_s]
  v_b=[np.zeros_like(every_ele) for every_ele in b_s]
  for iterations in range(max_epochs):
    # total_accuracy=0
    # total_loss=0
    dw=[np.zeros_like(every_ele) for every_ele in w_s]
    db=[np.zeros_like(every_ele) for every_ele in b_s]
    # points_touched_till_now=0
    for i in range(int(60000*1-(60000*validation_split))):
      label_image=np.zeros((10,1))
      label_image[labels_to_train[i]]=1
      a_s,h_s=forward_propagation(w_s,b_s,images_to_train[i].flatten().reshape((784,1)),no_of_layers,images_to_train,name_of_activation)
      # total_loss+=find_loss(h_s[-1],label_image)
      # if(np.argmax(h_s[-1])==labels_to_train[i]):
      #   total_accuracy+=1
      grad_w,grad_b=backward_propagation(a_s,h_s,label_image,w_s,b_s,no_of_layers,name_of_activation,name_of_loss_function)
      grad_w.reverse()
      grad_b.reverse()
      dw = [dw_current + grad for dw_current, grad in zip(dw, grad_w)]
      db = [db_current + grad for db_current, grad in zip(db, grad_b)]
      if((i+1)%mini_batch_size==0):
        m_w = [beta1* mw + (1-beta1)*(grad) for mw, grad in zip(m_w, dw)]
        m_b = [beta1* mb + (1-beta1)*(grad) for mb, grad in zip(m_b, db)]
        v_w = [beta2* vw + (1-beta2)*(grad**2) for vw, grad in zip(v_w, dw)]
        v_b = [beta2* vb + (1-beta2)*(grad**2) for vb, grad in zip(v_b, db)]
        m_w_hat=[m/(1-np.power(beta1,i+1)) for m in m_w]
        m_b_hat=[m/(1-np.power(beta1,i+1)) for m in m_b]
        v_w_hat=[v/(1-np.power(beta2,i+1)) for v in v_w]
        v_b_hat=[v/(1-np.power(beta2,i+1)) for v in v_b]
        for k in range(len(w_s)):
          for j in range(len(w_s[k])):
              update_w = np.dot((eta / (np.sqrt(v_w_hat[k][j]) + eps)), (beta1 * m_w_hat[k][j] + (((1 - beta1) * dw[k][j]) / (1 - np.power(beta1, k + 1)))))
              w_s[k][j] =w_s[k][j]*(1-weight_decay)-update_w
        for k in range(len(b_s)):
          for j in range(len(b_s[k])):
              update_b = np.dot((eta / (np.sqrt(v_b_hat[k][j]) + eps)), (beta1 * m_b_hat[k][j] + (((1 - beta1) * db[k][j]) / (1 - np.power(beta1, k + 1)))))
              b_s[k][j] -= update_b
        dw=[np.zeros_like(every_ele) for every_ele in w_s]
        db=[np.zeros_like(every_ele) for every_ele in b_s]
    # validation_accuracy,validation_loss=cal_val_acc_and_loss(h_s,images_to_train,validation_split)
    # # loss=(total_loss/(60000*(1-validation_split)))
    # cross_entropy_loss_function(h_s,w_s,b_s,images_to_train,labels_to_train)
    # # accuracy=(total_accuracy/(60000*(1-validation_split)))
    # accuracy=calculate_accuracy_and_loss(images_to_train,labels_to_train,w_s,b_s,h_s)
    # valid_loss=(validation_loss/(60000*(validation_split)))
    # valid_accuracy=(validation_accuracy/(60000*(validation_split)))
    train_loss,train_accuracy=calculate_train_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    valid_loss,valid_accuracy=calculate_valid_loss_and_accuracy(name_of_loss_function,w_s,b_s,h_s,images_to_train,labels_to_train,validation_split,name_of_activation,no_of_layers)
    print(f"training loss: {train_loss},training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}, validation loss: {valid_loss}")
    # wandb.log({'valid_loss':valid_loss})
    wandb.log({"train_loss":train_loss,"train_accuracy":train_accuracy,"valid_loss":valid_loss,"valid_accuracy":valid_accuracy,"epochs":iterations+1})

def accuracy_for_test(test_images,test_labels,better_w,better_b,no_of_layers,name_of_activation):
  a_s,h_s=forward_propagation(better_w,better_b,test_images.flatten().reshape((784,1)),no_of_layers,test_images,name_of_activation)
  return np.argmax(h_s[-1])


def ConfusionMatrix(test_images,test_labels,better_w,better_b,name_of_activation,no_of_layers):
  predicted_labels=[]
  true_labels=[]
  for x,y in zip(test_images,test_labels):
    # test_images=test_images.flatten().reshape((784,1))
    pred_label=accuracy_for_test(x,y,better_w,better_b,no_of_layers,name_of_activation)
    predicted_labels.append(pred_label)
    true_labels.append(y)
  conf_mat=confusion_matrix(true_labels,predicted_labels)
  plt.figure(figsize=(10,10))
  sn.heatmap(conf_mat,annot=True, fmt="d",cmap="Blues")
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.savefig('conf_matrix.png')
  wandb.log({"conf_matrix": wandb.Image('conf_matrix.png')})
  plt.show()

def optimization_function(name_of_optimizer,w_s,b_s,labels_to_train,validation_split,eta,max_epochs,beta,mini_batch_size,eps,beta1,beta2,weight_decay,name_of_activation,
                        name_of_loss_function,name_of_derivation_function,no_of_layers):
  if(name_of_optimizer=="sgd"):
    stochastic_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers)
  if(name_of_optimizer=="momentum"):
      better_w,better_b=momentum_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers)
      # ConfusionMatrix(test_images,test_labels,better_w,better_b,name_of_activation,no_of_layers)
  if(name_of_optimizer=="nesterov"):
    nesterov_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers)
  if(name_of_optimizer=="rmsprop"):
    rmsprop_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,mini_batch_size,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers)
  if(name_of_optimizer=="adam"):
    adam_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,beta1,beta2,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers,mini_batch_size)
  if(name_of_optimizer=="nadam"):
    nadam_gradient_descent(w_s,b_s,labels_to_train,validation_split,eta,max_epochs,name_of_activation,beta,beta1,beta2,eps,weight_decay,name_of_derivation_function,name_of_loss_function,no_of_layers,mini_batch_size)



# def no_of_epochs(max_epochs)
# def weight_decay(weight_decay_value)
# def batch_size(batch_size_value)
# name_of_function="random"
# no_of_layers=4
# no_neurons_per_layer=32
# name_of_optimizer="sgd"
# validation_split=0.1
# eta=0.001
# eps=1e-5
# beta=0.5
# beta1=0.9
# beta2=0.999
# max_epochs=10
# mini_batch_size=64
# weight_decay=0.5
# name_of_activation="ReLU"
# name_of_loss_function="cross entropy"
name_of_derivation_function="sigmoid_derivative"




def arguement_parsing():
  neural_network_parse_values = argparse.ArgumentParser(description='Training arguements for the network of neurons')
  neural_network_parse_values.add_argument('-wp','--wandb_project',type=str,default='dL_assignment_1',help='Project name used to track experiments in Weights & Biases dashboard')
  neural_network_parse_values.add_argument('-we','--wandb_entity',type=str,default='cs23m033_dl_assignment_1',help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
  neural_network_parse_values.add_argument('-d','--dataset',type=str,default='fashion_mnist',choices= ["mnist", "fashion_mnist"],help='Select the dataset either fashion_mnist or mnist')
  neural_network_parse_values.add_argument('-e','--epochs',type=int,default=5,help='Number of epochs to train the neural network')
  neural_network_parse_values.add_argument('-b','--batch_size',type=int,default=64,help='Batch size used to train neural network')
  neural_network_parse_values.add_argument('-l','--loss',type=str,default='cross_entropy',choices=["mean_squared_error", "cross_entropy"],help='Select the loss function')
  neural_network_parse_values.add_argument('-o','--optimizer',type=str,default='momentum',choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help='Select the optimiser function')
  neural_network_parse_values.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
  neural_network_parse_values.add_argument('-m','--momentum',type=float,default=0.5,help='Momentum used by momentum and nag optimizers')
  neural_network_parse_values.add_argument('-beta','--beta',type=float,default=0.5,help='Beta used by rmsprop optimizer')
  neural_network_parse_values.add_argument('-beta1','--beta1',type=float,default=0.9,help='Beta1 used by adam and nadam optimizers')
  neural_network_parse_values.add_argument('-beta2','--beta2',type=float,default=0.99,help='Beta2 used by adam and nadam optimizers')
  neural_network_parse_values.add_argument('-eps','--epsilon',type=float,default=1e-10,help='Epsilon used by optimizers')
  neural_network_parse_values.add_argument('-w_d','--weight_decay',type=float,default=0.0005,help='Weight decay used by optimizers')
  neural_network_parse_values.add_argument('-w_i','--weight_init',type=str,default='random',choices=["random", "Xavier"],help='Select any weight intialization')
  neural_network_parse_values.add_argument('-nhl','--num_layers',type=int,default=4,help='Number of hidden layers used in feedforward neural network')
  neural_network_parse_values.add_argument('-sz','--hidden_size',type=int,default=64,help='Number of hidden neurons in a feedforward layer')
  neural_network_parse_values.add_argument('-a','--activation',type=str,default='sigmoid',choices=["identity", "sigmoid", "tanh", "ReLU"],help='Select activation function')
  return neural_network_parse_values.parse_args()

parser=arguement_parsing()
wandb.init(project=parser.wandb_project)
if(parser.dataset=='mnist'):
  (images_to_train, labels_to_train), (test_images,test_labels) = mnist.load_data()
if(parser.dataset=='fashion_mnist'):
  (images_to_train, labels_to_train), (test_images,test_labels) = fashion_mnist.load_data()
if(parser.dataset==None):
  (images_to_train, labels_to_train), (test_images,test_labels) = mnist.load_data()
images_to_train, labels_to_train=shuffle(images_to_train,labels_to_train)
images_to_train=images_to_train/255.0
test_images=test_images/255.0	
validation_split=0.1
wandb.run.name=f'function_name->{parser.weight_init}_activation_func->{parser.activation}_optimizer->{parser.optimizer}_DataSet->{parser.dataset}'
w_s,b_s=intitalize_w_and_b(parser.weight_init,parser.num_layers,parser.hidden_size)
optimization_function(parser.optimizer,w_s,b_s,labels_to_train,0.1,parser.learning_rate,parser.epochs,parser.momentum,parser.batch_size,parser.epsilon,parser.beta1,parser.beta2,parser.weight_decay,parser.activation,parser.loss,name_of_derivation_function,parser.num_layers)



# def main_function(name_of_function,no_of_layers,no_neurons_per_layer,name_of_optimizer,labels_to_train,validation_split,eta,max_epochs,beta,mini_batch_size,eps,beta1,beta2,name_of_activation,
#                         name_of_loss_function,weight_decay,name_of_derivation_function):
#   w_s,b_s=intitalize_w_and_b(name_of_function,no_of_layers,no_neurons_per_layer)
#   optimization_function(name_of_optimizer,w_s,b_s,labels_to_train,validation_split,eta,max_epochs,beta,mini_batch_size,eps,beta1,beta2,weight_decay,name_of_activation,
#                         name_of_loss_function,name_of_derivation_function)

# !pip install wandb
# import wandb
# wandb.login()
# wandb.init(project="CS6910_dL_assignment_1")

# def main_function():
#   wandb.init(project='dL_assignment_1')
#   params=wandb.config
#   with wandb.init(project='dL_assignment_1',name='fun_of_w_b->'+params.name_of_function+'fun_opt->'+params.name_of_optimizer) as run :
#     w_s,b_s=intitalize_w_and_b(params.name_of_function,params.no_of_layers,params.no_neurons_per_layer)
#     optimization_function(params.name_of_optimizer,w_s,b_s,labels_to_train,0.1,params.eta,params.max_epochs,0.5,params.mini_batch_size,params.eps,0.9,0.999,params.weight_decay,params.name_of_activation,
#                         params.name_of_loss_function,name_of_derivation_function,params.no_of_layers)
# # main_function(name_of_function,no_of_layers,no_neurons_per_layer,name_of_optimizer,labels_to_train,validation_split,eta,max_epochs,beta,mini_batch_size,eps,beta1,beta2,name_of_activation,
# #               name_of_loss_function,weight_decay,name_of_derivation_function)


# sweep_params={
#     'method':'bayes',
#     'name': 'test',
#     'metric':{
#         'goal':'maximize',
#         'name':'valid_accuracy'
#     },
#     'parameters':{
#         'name_of_function':{'values':['random','xavier']},
#         'no_of_layers':{'values':[3,4,5]},
#         'no_neurons_per_layer':{'values':[64,32,128]},
#         'name_of_optimizer':{'values':['sgd','momentum','nestrov','rmsprop','adam','nadam']},
#         'eta':{'values':[0.00001,0.0001,0.001]},
#         'eps':{'values':[1e-10,1e-8]},
#         'mini_batch_size':{'values':[16,32,64]},
#         'weight_decay':{'values':[0,0.0005,0.5]},
#         'max_epochs':{'values':[5]},
#         'name_of_activation':{'values':['tanh','sigmoid','ReLU']},
#         'name_of_loss_function':{'values':['mean_squared_error','cross_entropy']}
#     }
# }

# sid=wandb.sweep(sweep_params,project='dL_assignment_1')
# wandb.agent(sid,function=main_function,count=5)