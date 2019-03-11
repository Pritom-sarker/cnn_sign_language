import pandas as pd
import tensorflow as tf
import My_Dl_lib as mdl
import matplotlib.pyplot as plt
mdl.off_warning()
#---------------- Data processing

train_data=pd.read_csv('data/sign_mnist_train.csv')
test_data=pd.read_csv('data/sign_mnist_test.csv')

# training set----->>
import numpy as np

t_x=np.array(train_data.drop(['label'],1))#.values.tolist()
train_X=t_x.reshape(len(t_x),28,28,1)
#print(X.shape)

#mdl.plot_a_pic(train_X[0])


train_Y = mdl.one_hot(train_data['label'].values.tolist(),25)

#print(train_Y[1])

#test set---->>

test_x=np.array(test_data.drop(['label'],1))#.values.tolist()
test_X=test_x.reshape(len(test_x),28,28,1)
#print(X.shape)

test_Y=mdl.one_hot(test_data['label'].values.tolist(),25) #(list,num of output class)

#-------------------model-------------------


#model peramiter
# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 25), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

epochs = 10
batch_size = 128
keep_probability = 0.7
learning_rate = 0.0001
conv="2 conv 4 full connected"

#model
logits = mdl.CNN(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

#Running model/
test_ite=0
saver = tf.train.Saver(save_relative_paths=True)
a=[]
l=[]
if __name__=='__main__':

    sess=tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    mdl._check_restore_parameters(sess, saver)
    #saver.restore(sess, "G:\Own_project\Data Science\Machine Learning\sign language\model\checkpoint")
    itter=10000

    for itr in range(0,itter):

        for i in range(0,int(len(train_X)/batch_size)):
                    batch_X,batch_Y=mdl.getBatch(i,batch_size,train_X,train_Y)
                    sess.run(optimizer,feed_dict={x:batch_X,y:batch_Y,keep_prob:keep_probability})


        #check accuracy


        loss,acc = sess.run([cost,accuracy], feed_dict={x: test_X, y: test_Y, keep_prob: keep_probability})
        a.append(acc*100)
        l.append(loss)
        print(" {} -> Accuracy is {}% && Loss -> {} ".format(itr, acc * 100,loss))


        if (itr % 10 == 0):

            saver.save(sess, 'model/my_test_model',global_step=itr)
            print("------- >> Model saved")

        if (itr%25==0):
            plt.clf()
            plt.plot(a)
            plt.title("Accuracy curve ")
            plt.xlabel("Accuracy")
            plt.xlabel("Iteration")

            plt.savefig("curve/accuracy dropout-{} lr - {} conv- {}.jpg".format(keep_probability,learning_rate,conv))

            plt.clf()
            plt.plot(l)
            plt.title("Loss")
            plt.savefig("curve/loss dropout- {} lr - {} conv- {} .jpg".format(keep_probability,learning_rate,conv))

