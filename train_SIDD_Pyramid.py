from __future__ import print_function

import h5py

try:
    import moxing as mox
    import npu_bridge
    mox.file.shift('os', 'mox')
    h5py_File_class = h5py.File

    class OBSFile(h5py_File_class):
        def __init__(self, name, *args, **kwargs):
            self._tmp_name = None
            self._target_name = name
            if name.startswith('obs://') or name.startswith('s3://'):
                self._tmp_name = os.path.join('cache', 'h5py_tmp',
                                              name.replace('/', '_'))
                if mox.file.exists(name):
                    mox.file.copy(name, self._tmp_name)
                name = self._tmp_name
            print(name)
            super(OBSFile, self).__init__(name, *args, **kwargs)

        def close(self):
            if self._tmp_name:
                mox.file.copy(self._tmp_name, self._target_name)
            super(OBSFile, self).close()

    setattr(h5py, 'File', OBSFile)
except:
    pass
import argparse
import glob
import os
import time

import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from network import network

_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr,
            high=255,
            low=0,
            cmin=None,
            cmax=None,
            pal=None,
            mode=None,
            channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and ((3 in shape) or
                                                       (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data,
                                 high=high,
                                 low=low,
                                 cmin=cmin,
                                 cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3, ), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        default="/home/zyq/Dataset/SIDD_Medium/SIDD_Medium_Raw/Data")
    parser.add_argument('--train_url', default="./checkpoint/SIDD_Pyramid/")
    parser.add_argument('--result_dir', default="./result/SIDD_Pyramid/")

    args = parser.parse_args()
    data_url = args.data_url
    train_url = args.train_url
    result_dir = args.result_dir
    file_list = glob.glob(data_url + '/*/*NOISY_RAW_010*')
    gt_list = glob.glob(data_url + '/*/*GT_RAW_010*')

    # train_ids = [os.path.basename(train_fn)[0:4] for train_fn in file_list]

    mat_img = {}
    gt_img = {}
    start = time.time()
    index = 0
    train_ids = []
    for file, gt_file in zip(file_list, gt_list):
        key = os.path.basename(file)[0:4]
        file_1 = file[:-5] + '1.MAT'
        gt_file_1 = gt_file[:-5] + '1.MAT'

        index = index + 1
        print(index, 'loading file: ', key)
        m = h5py.File(file)['x']
        m = np.expand_dims(np.expand_dims(m, 0), 3)
        m_1 = h5py.File(file_1)['x']
        m_1 = np.expand_dims(np.expand_dims(m_1, 0), 3)
        mat_img[key] = np.concatenate([m, m_1], 0)

        m_gt = h5py.File(gt_file)['x']
        m_gt = np.expand_dims(np.expand_dims(m_gt, 0), 3)
        m_gt_1 = h5py.File(gt_file_1)['x']
        m_gt_1 = np.expand_dims(np.expand_dims(m_gt_1, 0), 3)
        gt_img[key] = np.concatenate([m_gt, m_gt_1], 0)
        train_ids.append(key)
        if (len(train_ids) >= 20):
            break
    ps = 256  # patch size for training
    save_freq = 500

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    sess = tf.Session(config=config)
    # sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, None, None, 1])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 1])

    out_image = network(in_image)

    # h_tv = tf.nn.l2_loss(feature_map[:, 1:, :, :] - feature_map[:, :-1, :, :])
    # w_tv = tf.nn.l2_loss(feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :])
    # tv_loss = (h_tv + w_tv) / (255 * 256)
    G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
    # G_loss = G_loss_2 + 0.1 * tv_loss

    tf.summary.scalar('G_loss', G_loss)
    merged = tf.summary.merge_all()

    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32)

    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

    saver = tf.train.Saver(max_to_keep=15)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(train_url)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    g_loss = np.zeros((5000, 1))

    allfolders = glob.glob(result_dir + '/*0')
    lastepoch = 0
    for folder in allfolders:
        lastepoch = np.maximum(lastepoch, int(folder[-4:]))

    summary_writer = tf.summary.FileWriter(train_url, sess.graph)

    learning_rate = 1e-4

    epoch_loss_list = []
    min_epoch_loss = 50
    for epoch in range(lastepoch, 4001):
        if os.path.isdir("result/%04d" % epoch):
            continue
        if epoch > 1500:
            learning_rate = 5e-5
        if epoch > 2000:
            learning_rate = 1e-5
        if epoch > 2500:
            learning_rate = 5e-6
        if epoch > 3000:
            learning_rate = 1e-6
        if epoch > 3500:
            learning_rate = 5e-7

        cnt = 0
        epoch_loss = 0

        for ind in np.random.permutation(len(train_ids)):

            st = time.time()
            cnt += 1

            train_id = train_ids[ind]  #string
            train_batch = mat_img[train_id]
            gt_batch = gt_img[train_id]

            # crop
            H = train_batch.shape[1]
            W = train_batch.shape[2]

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_patch = train_batch[:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_batch[:, yy:yy + ps, xx:xx + ps, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            _, G_current, output, summary = sess.run(
                [G_opt, G_loss, out_image, merged],
                feed_dict={
                    in_image: input_patch,
                    gt_image: gt_patch,
                    lr: learning_rate
                })

            output = np.minimum(np.maximum(output, 0), 1)
            g_loss[ind] = G_current
            epoch_loss += G_current
            summary_writer.add_summary(summary, cnt + epoch * len(train_ids))

            print("%d %d Loss=%.4f Time=%.3f" %
                  (epoch, cnt, np.mean(
                      g_loss[np.where(g_loss)]), time.time() - st))

            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d' % epoch):
                    os.makedirs(result_dir + '%04d' % epoch)

                temp = np.concatenate(
                    (gt_patch[0, :, :, 0], output[0, :, :, 0]), axis=1)
                toimage(temp * 255, high=255, low=0, cmin=0,
                        cmax=255).save(result_dir + '%04d/%04d_00_train.jpg' %
                                       (epoch, int(train_id)))

        epoch_loss /= len(train_ids)
        epoch_loss_list.append(epoch_loss)
        epoch_summary = tf.Summary(value=[
            tf.Summary.Value(tag='epoch_loss', simple_value=epoch_loss)
        ])
        summary_writer.add_summary(summary=epoch_summary, global_step=epoch)
        summary_writer.flush()

        if epoch_loss_list[epoch] < min_epoch_loss:
            saver.save(sess, train_url + 'model.ckpt')
            with open(train_url + '/log.txt', 'a+') as log:
                log.write('saved epoch: %04d, epoch loss = ' % epoch +
                          str(epoch_loss) + '\n')
            print('saved epoch: %04d' % epoch)
            print(epoch_loss)
            min_epoch_loss = epoch_loss_list[epoch]
        if epoch >= 3990:
            saver.save(sess, train_url + 'model-%04d.ckpt' % epoch)
            with open(train_url + '/log.txt', 'a+') as log:
                log.write('final saved epoch: %04d, epoch loss = ' % epoch +
                          str(epoch_loss) + '\n')
