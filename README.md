### 框架

ubuntu18.04

tensorflow 2.6.0

### 主要目的

在過去常說AI是黑合子，主要的原因是不知道透過CNN抓到特徵長什麼樣子或是在圖像上是哪個區域。在2015年CAM被提出主要是用來觀察透過CNN抓出來的特徵的合理性，但是因為CAM有一個架構上的限制，需要將fully connected layer使用GAP(gloable average pooling)取代才可以使用。2016年Grad-CAM被提出可以適用於所有的網路進行相同的操作不受到GAP限制。

### CAM(Class activation mapping)

先說明一下cam的操作，在影像經過最後一層cnn之後依照特徵圖數量透過GAP將整個影像變成一個直，也就是說如果最後一層CNN的輸出是10*256*256*16 通過GAP之後就會變成10*1*1*16，接著將10*1*1*16的特徵向量乘上W加上bias最後送入activateion function(softmax之類的)，最後再反向。根據這樣的推倒過程可以發現，如果W愈到則該特徵圖愈重要，因此作者提出將W直接乘上該特徵圖上每一個像素，最後進行加總接著顯示出來。

### Grad-CAM

使用CAM需要將FCL更換成GAP才可以觀察，因此2016年時Gras-CAM發表，可以是用於所有種類的類神經網路，不需要重新修改架構可以直接使用。基於CAM的方式進行修改，主要的操作原理是透過反像傳遞時計算梯度的能力計算出梯度之後將這個梯度當成是CAM中的W回乘到影像上，因為梯度計算時會針對特徵圖上每一個像素計算一次梯度，因此針對每一張特徵圖會將計算出來的梯度進行整面的平均，這個操作類似GAP。

將最算出來的結果透過heatmap顯示出來

### 程式碼

1. 挑選出已訓練完成之網路最後一個conv的輸出以及整體網路輸入shape建構成一個網路，主要是需要取出這依整段的weight，才可以輸出最後一層conv的特徵圖

```python
model_x = self.model.get_layer(self.pr_model_name)
last_conv = model_x.get_layer(self.pr_model_output_layer_name)
last_conv_model = Model(model_x.input,last_conv.output)
```

1. 建構最後一個conv到整個模型輸出的網路，取出weight才可以計算梯度

```python
pre_input = Input(shape=(last_conv.output.shape[1],last_conv.output.shape[2],last_conv.output.shape[3]))
x = pre_input
for layer in self.classifier_layer_name:
    x = self.model.get_layer(layer)(x)
predict_mode = Model(pre_input,x)
```

1. 計算feature以及predict結果的梯度結果並且應設到最後一層conv輸出的特徵圖上面

```python
single_img = self.input_image.reshape((-1,self.input_image.shape[0],self.input_image.shape[1],self.input_image.shape[2]))/255.0
with tf.GradientTape() as tape:
    last_layer_output = last_conv_model(single_img)
    tape.watch(last_layer_output)
    preds = predict_mode(last_layer_output)
    top_pre = tf.argmax(preds[0])
    top_class_channel = preds[:,top_pre]
grads = tape.gradient(top_class_channel,last_layer_output) #梯度結果

pooled_grads = tf.reduce_mean(grads,axis=(0,1,2)).numpy() 
#將梯度結果針對每一個特徵圖取平均
last_layer_output = last_layer_output.numpy()[0]
#將權重值回程特徵圖
for i in range(pooled_grads.shape[-1]):
    last_layer_output[:,:,i]*=pooled_grads[i]
```

1. 計算heatmap 的最大值最小值，需要考慮到所有feature ma

```python
heatmap = np.mean(last_layer_output,axis=-1)
heatmap = np.maximum(heatmap,0)
heatmap/=np.max(heatmap)
result = self.show_heatmap(heatmap,self.input_image)
```

### 參考網站

[https://medium.com/手寫筆記/grad-cam-introduction-d0e48eb64adb](https://medium.com/%E6%89%8B%E5%AF%AB%E7%AD%86%E8%A8%98/grad-cam-introduction-d0e48eb64adb)