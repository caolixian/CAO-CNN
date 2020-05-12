
import numpy as np
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import glob




from keras.models import Sequential
from keras.layers.convolutional import Conv2D,Conv3D
from keras.layers.pooling import MaxPool2D,MaxPool3D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import Adam


from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
# from skimage.transform import resize



example_filename = '/media/large/datasets/nifti_201604-201808/'

csv_path='/media/large/dnas_member/miura/research/research_7/big_five_final.csv'
f=open(csv_path,'r')
reader=csv.reader(f)

header=next(reader)
next(reader)


#把所有的图片都放在一个最大尺寸的黑色盒子里，这样所有的3D尺寸都会统一了

#back_img = np.zeros([195,256,256],dtype=int)
# print(back_img.shape)
#？back_image，这里img_data[0]是什么
X=[]
Y=[]
n=[0,1]
n_one=np.identity(2)[n]
c=0
a=0
b=0
for row in reader:

    c+=1
    files=example_filename + row[10] + '_3DGEIR_SAG_15.nii.gz'
    # print(files)
    if os.path.exists(files)==True:



        if int(row[13])>9 and int(row[15])<8 and a<195:
            img = nib.load(files)
            img_arr=img.get_fdata()

            x_offset=0
            y_offset=0
            z_offset=0

            back_img = np.zeros([195,256,256],dtype=int)

            back_img[z_offset:z_offset+img_arr.shape[0],y_offset:y_offset+img_arr.shape[1], x_offset:x_offset+img_arr.shape[2]] = img_arr

            img=np.reshape(back_img,(195,256,256,1))
            X.append(img)
            Y.append(n_one[0])
            a+=1
            print('a='+str(a))
        if int(row[13])<8 and int(row[15])>9 and b<300:
            img = nib.load(files)
            img_arr=img.get_fdata()

            x_offset=0
            y_offset=0
            z_offset=0

            back_img = np.zeros([195,256,256],dtype=int)

            back_img[z_offset:z_offset+img_arr.shape[0],y_offset:y_offset+img_arr.shape[1], x_offset:x_offset+img_arr.shape[2]] = img_arr

            img=np.reshape(back_img,(195,256,256,1))
            X.append(img)
            Y.append(n_one[1])
            b+=1
            print('b='+str(b))
            # print(img.shape)

            # print('a='+str(a))
            # print('b='+str(b))

        # if c>=500:
        #     break

X=np.array(X)
# File "predicr.py", line 83, in <module>
# X=np.array(X)
# MemoryError

Y=np.array(Y)
#Found input variables with inconsistent numbers of samples: [48, 96]
print(len(X))
print(len(Y))

X = X.astype('float32')
# gazou_max = np.amax(X)
# X /= gazou_max
X = (X - np.mean(X)) / np.std(X)
#这里还需要是float32吗

# print(len(X))
# print(len(Y))

trainX,testX,trainY,testY = train_test_split(X,Y, test_size=0.2, random_state=111)

# print(len(trainX))


model = Sequential()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model.add(Conv3D(20,(4,4,4),padding='same',activation='relu',input_shape=(195,256,256,1)))
#第一层是输入数据相关的shape,nii3D文件尺寸读取猜想，先转换成image再读取，结果第三个参数会不停的变
model.add(MaxPool3D(pool_size=(8,8,8)))

model.add(Conv3D(20,(3,3,3),padding='same',activation='relu'))
model.add(MaxPool3D(pool_size=(4,4,4)))

model.add(Conv3D(20,(2,2,2),padding='same',activation='relu'))
model.add(MaxPool3D(pool_size=(2,2,2)))

model.add(Flatten())
model.add(Dense(3000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(trainX, trainY, batch_size=2, epochs=200,verbose=1, validation_data=(testX, testY))
predict


plt.plot(history.history['acc'],marker='o')
plt.plot(history.history['val_acc'],marker='o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
#she zhi wanggexian
plt.savefig('/media/share/dnas_member/miura/research/research_7/saved_data/share/cao/critical_aggressive/final_acc3.jpg')
#plt.show()

plt.close()

plt.plot(history.history['loss'],marker='o')
plt.plot(history.history['val_loss'],marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.savefig('/media/share/dnas_member/miura/research/research_7/saved_data/share/cao/critical_aggressive/final_loss3.jpg')
#plt.show()
plt.close()
