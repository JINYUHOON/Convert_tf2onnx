import tensorflow as tf
import math

def wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1]), axis=0)
        return loss
    
def NME(y_true,y_pred):
    w = h = 256 
    d = (w**2+h**2)**(1/2)
    nme = 0
    label_count=82
    for i in range(0,label_count,2):
        diff = tf.math.sqrt((y_pred[:,i] - y_true[:,i])**2 + (y_pred[:,i+1] - y_true[:,i+1])**2)/d
        nme += tf.reduce_mean(diff)
    return nme