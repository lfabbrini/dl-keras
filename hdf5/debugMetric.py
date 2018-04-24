#%%
import numpy as np
import keras 
from keras import metrics
from keras import backend as K #img_to_array


def test_stateful_metrics(metrics_mode):
    np.random.seed(1334)
    class ConfMtx(keras.layers.Layer):
        """Stateful Metric to compuet f1 over all batches.

        Assumes predictions and targets of shape `(samples, 1)`.

        # Arguments
            name: String, name for the metric.
        """

        def __init__(self, name='conf_mtx', label_list=[0,1], **kwargs):
            super(ConfMtx, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.label_list = label_list
            self.n_label = len(label_list)
            self.conf_mtx_nd = np.zeros(shape=(self.n_label,self.n_label), dtype='int32')
            self.conf_mtx = K.variable(value=self.conf_mtx_nd, dtype='int32')
            
        def reset_states(self):
            self.conf_mtx_nd = np.zeros(shape=(self.n_label,self.n_label), dtype='int32')
            K.set_value(self.conf_mtx, self.conf_mtx_nd)
        def __call__(self, y_true, y_pred):
            """Computes f1 in a batch.

            # Arguments
                y_true: Tensor, batch_wise labels
                y_pred: Tensor, batch_wise predictions

            # Returns
                The total number of true positives seen this epoch at the
                    completion of the batch.
            """
            y_true = K.cast(y_true, 'int32')
            y_pred = K.cast(K.round(y_pred), 'int32')
            for i,l_true in enumerate(self.label_list):
                are_true_l = K.cast(K.equal(y_true,l_true), 'int32')
                for j,l_pred in enumerate(self.label_list):
                    are_pred_l = K.cast(K.equal(y_pred,l_pred), 'int32')
                    value = K.cast(K.sum(are_true_l * are_pred_l), 'int32')
                    self.conf_mtx_nd[i,j] += K.get_value(value)
                    
            
            self.add_update(K.update_add(self.conf_mtx, self.conf_mtx_nd),
                            inputs=[y_true, y_pred])
            return np.trace(self.conf_mtx_nd)
#    class BinaryF1(keras.layers.Layer):
#        """Stateful Metric to count the total true positives over all batches.
#
#        Assumes predictions and targets of shape `(samples, 1)`.
#
#        # Arguments
#            name: String, name for the metric.
#        """
#
#        def __init__(self, pos_label=0,name='f1', **kwargs):
#            super(BinaryF1, self).__init__(name=name, **kwargs)
#            self.stateful = True
#            self.pos_label = pos_label
#            self.true_positives = K.variable(value=0, dtype='int32')
#            self.false_positives = K.variable(value=0, dtype='int32')
#            self.false_negatives = K.variable(value=0, dtype='int32')
#        def reset_states(self):
#            K.set_value(self.true_positives, 0)
#            K.set_value(self.false_positives, 0)
#            K.set_value(self.false_negatives, 0)
#        def __call__(self, y_true, y_pred):
#            """Computes the number of true positives in a batch.
#
#            # Arguments
#                y_true: Tensor, batch_wise labels
#                y_pred: Tensor, batch_wise predictions
#
#            # Returns
#                The total number of true positives seen this epoch at the
#                    completion of the batch.
#            """
#            y_true = K.cast(y_true, 'int32')
#            y_pred = K.cast(K.round(y_pred), 'int32')
#            are_true_l = K.cast(K.equal(y_true,self.pos_label), 'int32')
#            are_pred_l = K.cast(K.equal(y_pred,self.pos_label), 'int32')
#            are_not_true_l = K.cast(K.not_equal(y_true,self.pos_label), 'int32')
#            are_not_pred_l = K.cast(K.not_equal(y_pred,self.pos_label), 'int32')
#            tp = K.cast(K.sum(are_true_l * are_pred_l), 'int32')
#            fp = K.cast(K.sum(are_not_true_l * are_pred_l), 'int32')
#            fn = K.cast(K.sum(are_true_l * are_not_pred_l), 'int32')
#            
#            prec = tp/(tp+fp)
#            rec = tp/(tp+fn)
#            f1 = 2*(prec*rec)/rec
#            
#            current_true_pos = self.true_positives * 1
#            self.add_update(K.update_add(self.true_positives,
#                                         tp),
#                            inputs=[y_true, y_pred])
#            return current_true_pos + tp    
    class BinaryTruePositives(keras.layers.Layer):
        """Stateful Metric to count the total true positives over all batches.

        Assumes predictions and targets of shape `(samples, 1)`.

        # Arguments
            name: String, name for the metric.
        """

        def __init__(self, name='true_positives', **kwargs):
            super(BinaryTruePositives, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.true_positives = K.variable(value=0, dtype='int32')

        def reset_states(self):
            K.set_value(self.true_positives, 0)

        def __call__(self, y_true, y_pred):
            """Computes the number of true positives in a batch.

            # Arguments
                y_true: Tensor, batch_wise labels
                y_pred: Tensor, batch_wise predictions

            # Returns
                The total number of true positives seen this epoch at the
                    completion of the batch.
            """
            y_true = K.cast(y_true, 'int32')
            y_pred = K.cast(K.round(y_pred), 'int32')
            correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
            true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
            current_true_pos = self.true_positives * 1
            self.add_update(K.update_add(self.true_positives,
                                         true_pos),
                            inputs=[y_true, y_pred])
            return current_true_pos + true_pos
#%%
#    metric_fn = BinaryTruePositives()
#    config = metrics.serialize(metric_fn)
#    metric_fn = metrics.deserialize(
#        config, custom_objects={'BinaryTruePositives': BinaryTruePositives})

    metric_fn = ConfMtx()
    config = metrics.serialize(metric_fn)
    metric_fn = metrics.deserialize(
    config, custom_objects={'ConfMtx': ConfMtx})

    # Test on simple model
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(1, activation='sigmoid', name='out')(inputs)
    model = keras.Model(inputs, outputs)

    if metrics_mode == 'list':
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['acc', metric_fn])
    elif metrics_mode == 'dict':
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics={'out': ['acc', metric_fn]})

    samples = 1000
    x = np.random.random((samples, 2))
    y = np.random.randint(2, size=(samples, 1))

    val_samples = 10
    val_x = np.random.random((val_samples, 2))
    val_y = np.random.randint(2, size=(val_samples, 1))

    # Test fit and evaluate
    history = model.fit(x, y, validation_data=(val_x, val_y), epochs=1, batch_size=10)
    outs = model.evaluate(x, y, batch_size=10)
    preds = model.predict(x)

    def ref_true_pos(y_true, y_pred):
        return np.sum(np.logical_and(y_pred > 0.5, y_true == 1))

    # Test correctness (e.g. updates should have been run)
    np.testing.assert_allclose(outs[2], ref_true_pos(y, preds), atol=1e-5)

    # Test correctness of the validation metric computation
    val_preds = model.predict(val_x)
    val_outs = model.evaluate(val_x, val_y, batch_size=10)
    np.testing.assert_allclose(val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
#    np.testing.assert_allclose(val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)
    np.testing.assert_allclose(val_outs[2], history.history['val_conf_mtx'][-1], atol=1e-5)

    # Test with generators
    gen = [(np.array([x0]), np.array([y0])) for x0, y0 in zip(x, y)]
    val_gen = [(np.array([x0]), np.array([y0])) for x0, y0 in zip(val_x, val_y)]
    history = model.fit_generator(iter(gen), epochs=1, steps_per_epoch=samples,
                                  validation_data=iter(val_gen), validation_steps=val_samples)
    outs = model.evaluate_generator(iter(gen), steps=samples)
    preds = model.predict_generator(iter(gen), steps=samples)

    # Test correctness of the metric re ref_true_pos()
    np.testing.assert_allclose(outs[2], ref_true_pos(y, preds), atol=1e-5)

    # Test correctness of the validation metric computation
    val_preds = model.predict_generator(iter(val_gen), steps=val_samples)
    val_outs = model.evaluate_generator(iter(val_gen), steps=val_samples)
    np.testing.assert_allclose(val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
#    np.testing.assert_allclose(val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)
    np.testing.assert_allclose(val_outs[2], history.history['val_conf_mtx'][-1], atol=1e-5)
    
    #%%
#    metrics_mode = ['list', 'dict']
test_stateful_metrics('list')