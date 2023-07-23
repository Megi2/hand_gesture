#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_move_2.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.landmark_lists = [[0 for j in range(42)] for i in range(37)]

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        for i in range(36):
            self.landmark_lists[i] = self.landmark_lists[i+1]
        self.landmark_lists[36] = landmark_list
        arr = np.array([self.landmark_lists], dtype=np.float32).reshape(37, 42, 1)
        arr = tf.expand_dims(arr, axis=0)

        self.interpreter.set_tensor(
            input_details_tensor_index,
            arr)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
