#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_move.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.landmark_lists = [0 for i in range(37*42)]
        self.result_list = [0 for i in range(100)]
        self.e_list = []
        self.count = 0

    def __call__(
        self,
        landmark_list,
        label
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.list_update(landmark_list)

        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([self.landmark_lists], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return self.result_update(result[0][label])
        # return self.count
    
    def result_update(self, result):
        self.result_list[1:] = self.result_list[0:99] 
        self.result_list[0] = result

        if result > 0.2:
            self.count +=1
            self.e_list.append(result)
            return 0
        
        elif self.count != 0 and result < 0.2:
            max_accuracy = max(self.e_list)
            
            self.count = 0
            self.e_list = []

            return max_accuracy

    def list_update(self, landmark_list):
        self.landmark_lists[0:36*42] = self.landmark_lists[42:37*42]
        self.landmark_lists[36*42:37*42] = landmark_list
