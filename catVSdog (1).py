
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import math

from keras.applications import resnet50, inception_v3, xception, inception_resnet_v2
from keras.preprocessing import image
from tqdm import tqdm


# In[2]:


print(random.choice(os.listdir("../input/train/cat")))


# <h1>图片展示</h1>

# In[2]:


path_cat = "../input/train/cat"
path_dog= "../input/train/dog"

def show_img(img_path_list):
    fig = plt.figure(figsize=(16, 4 * math.ceil(len(img_path_list)/4.0)))
    for i in range(len(img_path_list)):
        img = cv2.imread(img_path_list[i])
        img = img[:,:,::-1]

        ax = fig.add_subplot(math.ceil(len(img_path_list)/4),4,i+1)
        ax.axis('off')
        ax.set_title(img_path_list[i])
        img = cv2.resize(img, (224,224))
        ax.imshow(img)
    
    plt.show()
        


# In[5]:


random.seed(21)
show_path_list = random.sample(os.listdir(path_cat),8)
for i in range(len(show_path_list)):
    show_path_list[i] = path_cat+"/"+show_path_list[i]
    
show_img(show_path_list)


# In[6]:


random.seed(21)
show_path_list = random.sample(os.listdir(path_dog),8)
for i in range(len(show_path_list)):
    show_path_list[i] = path_dog+"/"+show_path_list[i]
    
show_img(show_path_list)


# In[41]:


show_img("../input/train/dog/dog.8110.jpg".split())


# <h1>图片清洗</h1>
# 参考https://zhuanlan.zhihu.com/p/34068451?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0IQskNR8

# In[3]:


# imgNet中的猫狗种类
Dogs = [ 'n02085620','n02085782','n02085936','n02086079','n02086240','n02086646','n02086910','n02087046','n02087394','n02088094','n02088238',
        'n02088364','n02088466','n02088632','n02089078','n02089867','n02089973','n02090379','n02090622','n02090721','n02091032','n02091134',
        'n02091244','n02091467','n02091635','n02091831','n02092002','n02092339','n02093256','n02093428','n02093647','n02093754','n02093859',
        'n02093991','n02094114','n02094258','n02094433','n02095314','n02095570','n02095889','n02096051','n02096177','n02096294','n02096437',
        'n02096585','n02097047','n02097130','n02097209','n02097298','n02097474','n02097658','n02098105','n02098286','n02098413','n02099267',
        'n02099429','n02099601','n02099712','n02099849','n02100236','n02100583','n02100735','n02100877','n02101006','n02101388','n02101556',
        'n02102040','n02102177','n02102318','n02102480','n02102973','n02104029','n02104365','n02105056','n02105162','n02105251','n02105412',
        'n02105505','n02105641','n02105855','n02106030','n02106166','n02106382','n02106550','n02106662','n02107142','n02107312','n02107574',
        'n02107683','n02107908','n02108000','n02108089','n02108422','n02108551','n02108915','n02109047','n02109525','n02109961','n02110063',
        'n02110185','n02110341','n02110627','n02110806','n02110958','n02111129','n02111277','n02111500','n02111889','n02112018','n02112137',
        'n02112350','n02112706','n02113023','n02113186','n02113624','n02113712','n02113799','n02113978']
Cats=['n02123045','n02123159','n02123394','n02123597','n02124075','n02125311','n02127052']


# In[4]:


def error_img(model, path, animal):
    error_path=[]
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        dir_list[i] = path+"/"+dir_list[i]
        tf_img = image.load_img(dir_list[i], target_size=(224,224))
        
        x = image.img_to_array(tf_img)
        x = np.expand_dims(x, axis=0)
        x = model.preprocess_input(x) 
        
        preds = real_model.predict(x)
        
        a = []
        for z in range(len(model.decode_predictions(preds, top=50)[0])):
            a.append(model.decode_predictions(preds, top=50)[0][z][0])
        
        if not (set(a) & set(animal)):
            show_img(dir_list[i].split())
            error_path.append(dir_list[i])
    
    return error_path
            
        
        
        


# In[5]:


real_model = resnet50.ResNet50(weights='imagenet')
cat_error_renet50 = error_img(resnet50, path_cat, Cats)


# In[6]:


real_model = inception_v3.InceptionV3(weights='imagenet')
cat_error_inception_v3 = error_img(inception_v3, path_cat, Cats)


# In[7]:


real_model = xception.Xception(weights='imagenet')
cat_error_xception = error_img(xception, path_cat, Cats)


# In[8]:


real_model = resnet50.ResNet50(weights='imagenet')
dog_error_renet50 = error_img(resnet50, path_dog, Dogs)


# In[9]:


real_model = inception_v3.InceptionV3(weights='imagenet')
dog_error_inception_v3 = error_img(inception_v3, path_dog, Dogs)


# In[10]:


real_model = xception.Xception(weights='imagenet')
dog_error_xception = error_img(xception, path_dog, Dogs)


# In[11]:


cat_error_path = set(cat_error_renet50+cat_error_inception_v3+cat_error_xception)
dog_error_path = set(dog_error_renet50+dog_error_inception_v3+dog_error_xception)


# In[17]:


print(len(set(list(cat_error_path)+list(dog_error_path))))


# In[20]:


(list(cat_error_path)+list(dog_error_path))[0]


# 清除错误数据

# In[27]:


all_error = list(cat_error_path)+list(dog_error_path)
# 谨慎运行
for q in all_error:
    os.remove(str(q))


# In[29]:


print(len(os.listdir("../input/train/cat")))
print(len(os.listdir("../input/train/dog")))


# In[2]:


from keras.models import *
from keras.layers import *
from sklearn.utils import shuffle

#模型保存
import h5py


# In[5]:


def save_feature(model_main, size, model_name, model_func = None):
    input_tensor = Input((size[1], size[0], 3))
    x = input_tensor
    if model_func:
        x = Lambda(model_func)(x)
    
    base_model = model_main(input_tensor=x, weights='imagenet', include_top=False)
#     对激活层求平均值防止过拟合
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    
    gen = image.ImageDataGenerator()
#     返回标签
    train_generator = gen.flow_from_directory("../input/train", size, shuffle = False, batch_size=16)
#     不返回标签
    test_generator = gen.flow_from_directory("../input/test", size, shuffle=False, class_mode=None, batch_size=16)
    
    train = model.predict_generator(train_generator, 1547)
    test = model.predict_generator(test_generator, 782)
#     保存为文件
    with h5py.File("gap_%s.h5"%model_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        
save_feature(inception_v3.InceptionV3, (299,299),model_name='InceptionV3m', model_func =inception_v3.preprocess_input)
save_feature(xception.Xception, (299, 299), model_name='Xceptionm', model_func =xception.preprocess_input)
save_feature(inception_resnet_v2.InceptionResNetV2, (299,299), model_name='InceptionResNetV2m', model_func = inception_resnet_v2.preprocess_input)
save_feature(resnet50.ResNet50, (224,224), model_name='ResNet50m')


# In[26]:


print(len(os.listdir('../input/test/test')))


# 模型读取

# In[4]:


X_train = []
X_test = []

np.random.seed(777)

for filename in ["gap_InceptionResNetV2m.h5", "gap_Xceptionm.h5", "gap_InceptionV3m.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)


X_train, y_train = shuffle(X_train, y_train)


# 自定义底部全连接层
# - 二分类神经网络，只有一个节点，使用sigmoid作为激活函数
# - 二分类问题，binary_crossentropy作为损失函数 adam作为优化函数
# - 参考 https://keras.io/zh/models/model/

# In[5]:


from keras.optimizers import Adadelta
input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

adadelta = Adadelta(lr=1e-4)

model.compile(optimizer=adadelta,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[6]:


model.summary()


# In[7]:


from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='catVSdog.weights.best.hdf5', verbose=1, save_best_only=True)
# 每一批64， 20个epoch，将0.2拆分为验证集，显示训练日志
history = model.fit(X_train, y_train, batch_size=128,callbacks=[checkpointer], epochs=20, validation_split=0.2, verbose=1)
# 加载最优模型

model.load_weights('catVSdog.weights.best.hdf5')


# In[8]:


print(history.history.keys())


# In[9]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[48]:


y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

df = pd.read_csv("sample_submission.csv")

gen = image.ImageDataGenerator()
test_generator = gen.flow_from_directory("../input/test", (224, 224), shuffle=False, 
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('grade.csv', index=None)
df.head(10)


# In[16]:


def show_end_img(img_path_list,csv):
    fig = plt.figure(figsize=(16, 4 * math.ceil(len(img_path_list)/4.0)))
    for i in range(len(img_path_list)):
        num = int(img_path_list[i])
        pred = csv.loc[int(num-1), 'label']
        path = '../input/test/test/'+str(num)+'.jpg'
        img = cv2.imread(path)
        img = img[:,:,::-1]

        ax = fig.add_subplot(math.ceil(len(img_path_list)/4),4,i+1)
        ax.axis('off')
        ax.set_title(path+'\n'+str(pred))
        img = cv2.resize(img, (224,224))
        ax.imshow(img)
    
    plt.show()


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
csv = pd.read_csv('pred.csv')
img_np_array = np.random.randint(1, 12500, size=12, dtype='int')
img_list = img_np_array.tolist()
show_end_img(img_list, csv)

