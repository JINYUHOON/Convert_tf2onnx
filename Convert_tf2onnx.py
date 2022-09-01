from tensorflow import keras
import tf2onnx
import onnx
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import Keras_object

NME = Keras_object.NME
wing_loss = Keras_object.wing_loss

custom_obj = {'NME':NME,'wing_loss':wing_loss}

class get_onnx:
    def __init__(self, model_path :str):
        self.model_path = model_path + '.h5'
        
        model = keras.models.load_model(self.model_path, custom_objects=custom_obj)
        if model.input.shape[0] != 1:
            model.input.set_shape((1,256,256, 3))
        onnx_model, _ = tf2onnx.convert.from_keras(model, opset=16)
        onnx.save(onnx_model, self.model_path + '.onnx')
        print('Complete Save : ', self.model_path + 'onnx')
        
if __name__ == '__main__':
    input_path = input('>')
    get_onnx(model_path = input_path)
    print('Complete')
