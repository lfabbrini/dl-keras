from __future__ import print_function #print withouth newline
import sklearn.metrics as sklm
import keras
import numpy as np
#import sys #flush

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        self.y_pred = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        self.y_val = self.validation_data[1]

#        self.auc.append(sklm.roc_auc_score(targ, score))
#        self.confusion.append(sklm.confusion_matrix(targ, predict))
#        self.precision.append(sklm.precision_score(targ, predict))
#        self.recall.append(sklm.recall_score(targ, predict))
#        self.f1s.append(sklm.f1_score(targ, predict))
#        self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return
    
class F1(Metrics):
    def __init__(self,pos_label=0):
        super(F1,self).__init__()
        self.pos_label=pos_label
        self.wrap = []
    def on_epoch_end(self, epoch, logs=None):
        super(F1,self).on_epoch_end(epoch, logs=logs)
        self.wrap.append(sklm.f1_score(self.y_val, self.y_pred,self.pos_label))
        
        
class MetricsGenerator(keras.callbacks.Callback):
    def __init__(self,validation_data=None):
        super(MetricsGenerator,self).__init__()
        self.generator = validation_data
    def on_train_begin(self, logs=None):
        if self.generator is None:
            raise RuntimeError('Requires a generator.')
        self.n = self.generator.batch_size * len(self.generator)
        self.y_pred = np.zeros((self.n,1),dtype='uint8')
        self.y_val = np.zeros((self.n,1),dtype='uint8')
    def on_epoch_end(self, epoch, logs=None):
        for idx in range(len(self.generator)):
            (x_val,y_val) = self.generator[idx]
            self.y_pred[idx*self.generator.batch_size:(idx+1)*self.generator.batch_size] = np.round(np.asarray(self.model.predict(x_val)))
            self.y_val[idx*self.generator.batch_size:(idx+1)*self.generator.batch_size] = y_val
#            print(idx,x_val.shape,y_val.shape)
        
        
class F1Generator(MetricsGenerator):
    def __init__(self,validation_data=None,pos_label=0):
        super(F1Generator,self).__init__(validation_data=validation_data)
        self.pos_label=pos_label
        self.wrap = []
    def on_epoch_end(self, epoch, logs=None):
        super(F1Generator,self).on_epoch_end(epoch, logs)
        val = sklm.f1_score(self.y_val, self.y_pred,self.pos_label)
        self.wrap.append(val)
        print (' — val_f1: {:f}'.format(val), flush=True)
#        sys.stdout.flush()
#        print('x'*30)
#        print('f1{:.3f}'.format(val))
#        print('x'*30)