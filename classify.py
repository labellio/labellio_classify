#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import sys
import os
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import imghdr


def get_image_data(img_path=None):
    raw_image = None
    if img_path is not None:
        raw_image = tf.read_file(img_path)
        with open(img_path, 'rb') as fp:
            imagedata = fp.read()
        imagetype = imghdr.what(None, h=imagedata)
        if imagetype in ["jpg", "jpeg"]:
            return tf.image.decode_jpeg(raw_image, channels=3)
        if imagetype in ["png"]:
            return tf.image.decode_png(raw_image, channels=3)
    return tf.image.decode_image(raw_image, channels=3)


def result(checkpoint_path=None,
           model_name=None,
           num_classes=0,
           img_path=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.Graph().as_default():
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=(num_classes - 0),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = network_fn.default_image_size
        sample_image = get_image_data(img_path=img_path)
        image = image_preprocessing_fn(sample_image,
                                       eval_image_size,
                                       eval_image_size)
        image = tf.expand_dims(image, 0)

        logits, end_points = network_fn(image)
        predictions = tf.argmax(logits, 1)
        probabilities = tf.nn.softmax(logits)

        ####################
        # Print Probability #
        ####################

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            proba = sess.run(probabilities)[0]
            result = np.array([[i, p] for i, p in zip(range(num_classes),
                                                      proba)])
        return result


def main(args):
    model_dir = args.model_dir
    image_dir = args.image_dir
    network_name = args.network_name
    print("...")
    images = os.listdir(image_dir)
    f = open(os.path.join(model_dir, 'label.txt'))
    lines2 = f.readlines()
    f.close()
    num_classes = len(lines2)
    label_number_name = {}
    for line in lines2:
        number, name = line.split(":", 1)
        label_number_name[number] = name.replace('\n', '')
    iteration = os.path.basename(model_dir)[6:]
    for image in images:
        try:
            results = result(checkpoint_path=os.path.join(model_dir,
                                                          "model.ckpt-{}"
                                                          .format(iteration)),
                             model_name=network_name,
                             num_classes=num_classes,
                             img_path=os.path.join(image_dir, image))
            tmp = results[np.argsort(results, axis=0)[::-1][0, 1]]
        except Exception as e:
            print("{}\t{}\t{}"
                  .format(os.path.join(image_dir, image),
                          "error",
                          "error"))
            continue
        print("{}\t{}\t{}"
              .format(os.path.join(image_dir, image),
                      label_number_name[str(int(tmp[0]))],
                      tmp[1]))


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description=" Labellio Classifier")
    parser.add_argument("model_dir",
                        help="(exp. model-1000 :this number is the number of iteration)\
                              Path to a model directory\
                              which is exported from Labellio.\
                              (Please extract the archive before you use it.)")
    parser.add_argument("image_dir",
                        help="Path to an image directory.\
                              The directory should contain only images.")
    parser.add_argument("network_name", help="Network name of the model.")

    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
