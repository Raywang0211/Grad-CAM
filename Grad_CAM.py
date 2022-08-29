import numpy
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Flatten,Dense,Dropout
from tensorflow.keras.applications.xception import Xception 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import cv2
import tensorflow_addons as tfa
import numpy as np 
from keras import backend as k
from tensorflow.python.keras import Model
import matplotlib.cm as cm
from tensorflow.keras.utils import img_to_array,array_to_img

class GradCam():
    def __init__(self,input_model,image_path,pretrainmodel_name,pretrainmodel_output_layer_name,classifier_layer_names) -> None:
        """
        input_model: trained model
        image_path: input image path
        pretrainmodel_name: pretrainmodel name ex:'xception'
        pretrainmodel_output_layer_name: the name of output layer ex:'block14_sepconv2'
        classifier_layer_names: layer names after feature extracter [a list], Using this list to predict and calculate
        gradient and sum. 


        """
        self.model = input_model        
        self.input_image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        self.pr_model_name = pretrainmodel_name
        self.pr_model_output_layer_name = pretrainmodel_output_layer_name
        self.classifier_layer_name = classifier_layer_names
        pass    
    def gradcam_calculate(self):
        
        model_x = self.model.get_layer(self.pr_model_name)
        last_conv = model_x.get_layer(self.pr_model_output_layer_name)
        last_conv_model = Model(model_x.input,last_conv.output)

        #generate predict model (input:last conv,output:model output)
        pre_input = Input(shape=(last_conv.output.shape[1],last_conv.output.shape[2],last_conv.output.shape[3]))
        x = pre_input
        for layer in self.classifier_layer_name:
            x = self.model.get_layer(layer)(x)
        predict_mode = Model(pre_input,x)

        single_img = self.input_image.reshape((-1,self.input_image.shape[0],self.input_image.shape[1],self.input_image.shape[2]))/255.0
        with tf.GradientTape() as tape:
            last_layer_output = last_conv_model(single_img)
            tape.watch(last_layer_output)
            preds = predict_mode(last_layer_output)
            top_pre = tf.argmax(preds[0])
            top_class_channel = preds[:,top_pre]
        grads = tape.gradient(top_class_channel,last_layer_output)

        pooled_grads = tf.reduce_mean(grads,axis=(0,1,2)).numpy()
        last_layer_output = last_layer_output.numpy()[0]

        for i in range(pooled_grads.shape[-1]):
            last_layer_output[:,:,i]*=pooled_grads[i]

        heatmap = np.mean(last_layer_output,axis=-1)
        heatmap = np.maximum(heatmap,0)
        heatmap/=np.max(heatmap)
        # plt.imshow(heatmap)
        result = self.show_heatmap(heatmap,self.input_image)
        # plt.imshow(result)
        # plt.show()
        return result

    def show_heatmap(self,heatmap,img):
    
        single_img = img_to_array(img)
        heatmap = np.uint8(255*heatmap)
        jet = cm.get_cmap("jet")
        jet_colos = jet(np.arange(256))[:,:3]
        jet_heatmap = jet_colos[heatmap]

        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((single_img.shape[1],single_img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        super_img = jet_heatmap*0.4+single_img
        super_img = array_to_img(super_img)

        return super_img



if __name__=="__main__":
    model = models.load_model("/media/g4user/m2_disk1/contact_lens_classification/models_file/model.h5",compile=False)
    model.summary()
    img_path ="/media/g4user/m2_disk1/contact_lens_classification/datas/test/G4_projectoron_nogamma/3/3_r_c/bot_lc_2022-07-18-122121_p190e10d0f1.png"
    layer_names=["flatten","dense","group_normalization",'dropout',"dense_1"]
    GC = GradCam(
            model,
            img_path,
            'xception',
            'block14_sepconv2',
            layer_names,
            )
    GC.gradcam_calculate()


