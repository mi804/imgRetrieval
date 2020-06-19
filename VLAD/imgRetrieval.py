import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import argparse
import pickle
import mmcv

from folder_101dataset import testset, trainset


def parse_args():
    parser = argparse.ArgumentParser(description='store binary')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='work_dirs/retrieval_result_101/')
    parser.add_argument('--dic_dir',
                        type=str,
                        default='work_dirs/dictionary_101/')
    parser.add_argument('--treeIndex',
                        type=str,
                        default='work_dirs/dictionary_101/train_tree.pickle')
    parser.add_argument(
        '--pathVD',
        type=str,
        default='work_dirs/dictionary_101/visualDictionary.pickle')
    parser.add_argument('--num_words', type=int, default=1000)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--leafsize', type=int, default=40)
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--runtrain', action='store_true', help='run train')
    args = parser.parse_args()
    return args


def imshow(img, wind_name='img', key=0):
    cv2.imshow(wind_name, img)
    cv2.waitKey(key)


def VLAD(X, visualDictionary):
    X = X.astype(float)
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k, d])
    # computing the differences
    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V) * np.sqrt(np.abs(V))

    # L2 normalization
    V = V / np.sqrt(np.dot(V, V))
    return V


def train(args):
    sift_det = cv2.xfeatures2d.SIFT_create()
    print('computing des of trainset!')
    mbk = MiniBatchKMeans(init='k-means++',
                          n_clusters=args.num_words,
                          batch_size=50000,
                          n_init=10,
                          max_no_improvement=10,
                          verbose=0)
    des_matrix = np.zeros((0, 128))
    prog_bar = mmcv.ProgressBar(100)
    for i in range(len(trainset)):
        data = trainset[i]
        image, label = data
        kp, des = sift_det.detectAndCompute(image, None)
        if des is not None:
            des_matrix = np.row_stack((des_matrix, des))
        prog_bar.update()
        if i == (len(trainset) - 1) or i % 100 == 99:
            print('')
            print('computing clusters!')
            print('[%d of %d]' % (i + 1, len(trainset)))
            mbk.partial_fit(des_matrix)
            des_matrix = np.zeros((0, 128))
            prog_bar = mmcv.ProgressBar(100)
    mmcv.mkdir_or_exist(args.dic_dir)
    filepath = args.dic_dir + 'visualDictionary.pickle'
    print('dumping visualDictionary!')
    with open(filepath, 'wb') as pk:
        pickle.dump(mbk, pk)
    print('computing VLAD of trainset')
    prog_bar = mmcv.ProgressBar(len(trainset))
    descriptors = list()
    idImage = list()
    for i in range(len(trainset)):
        data = trainset[i]
        image, label = data
        kp, des = sift_det.detectAndCompute(image, None)
        if des is not None:
            v = VLAD(des, mbk)
            descriptors.append(v)
            idImage.append(i)
        prog_bar.update()
    descriptors = np.asarray(descriptors)
    tree = BallTree(descriptors, leaf_size=args.leafsize)
    print('')
    print('dumping dictionary tree')
    file_path = args.dic_dir + "train_tree.pickle"
    with open(file_path, 'wb') as f:
        pickle.dump([idImage, tree], f, pickle.HIGHEST_PROTOCOL)
    print('')


def test(args):
    # load the index
    with open(args.treeIndex, 'rb') as f:
        indexStructure = pickle.load(f)

    # load the visual dictionary
    with open(args.pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)
    sift_det = cv2.xfeatures2d.SIFT_create()
    imageID = indexStructure[0]
    tree = indexStructure[1]
    prog_bar = mmcv.ProgressBar(len(testset))

    K = args.topk
    AP = np.zeros((len(testset), 1))
    sum_tp = np.zeros((K, 1))
    for i in range(len(testset)):
        data = testset[i]
        image, label = data
        kp, des = sift_det.detectAndCompute(image, None)
        v = VLAD(des.astype(float), visualDictionary)
        dist, ind = tree.query(v.reshape(1, -1), args.topk)
        if args.show:
            dir_ = args.work_dirs + 'query of ' + str(i)
            mmcv.mkdir_or_exist(dir_)
            imshow(image, wind_name=('Query ' + str(i)))
            query_path = dir_ + '/Query.png'
            cv2.imwrite(query_path, image)
            # loop over the results
            kkk = 1
            for j in ind[0]:
                # load the result image and display it
                id = imageID[j]
                result = trainset[id][0]
                imshow(result, 'Result ' + str(kkk))
                cv2.destroyWindow('Result ' + str(kkk))
                result_path = dir_ + '/Result_' + str(kkk) + '.png'
                cv2.imwrite(result_path, result)
                kkk += 1
            cv2.destroyWindow(('Query ' + str(i)))

        buffer_yes = np.zeros((K, 1))
        total_relevant = 0
        for kk in range(len(ind[0])):
            j = ind[0][kk]
            # load the result image and display it
            id = imageID[j]
            result, tr_label = trainset[id]
            if tr_label == label:
                buffer_yes[kk] = 1
                total_relevant = total_relevant + 1
        P = np.divide(buffer_yes.cumsum().reshape(-1, 1),
                      np.arange(1, K + 1, 1).reshape(-1, 1))
        if buffer_yes.sum() != 0:
            AP[i] = sum(P * buffer_yes) / sum(buffer_yes)
        sum_tp += buffer_yes.cumsum().reshape(-1, 1)
        prog_bar.update()
    precision_at_k = sum_tp / (np.arange(1, K + 1, 1).reshape(-1, 1) *
                               len(testset))
    mmcv.mkdir_or_exist('work_dirs/precision_101')
    precision_path = 'work_dirs/precision_101/precision.txt'
    Map_path = 'work_dirs/precision_101/Map.txt'
    for i in range(K):
        print('precision at top %d: %.4f' % (i + 1, precision_at_k[i][0]))
    Map = AP.mean()
    print('map of flowers test set: %f ' % (Map))
    np.savetxt(precision_path, precision_at_k)
    np.savetxt(Map_path, np.array([Map]))
    print('')


def main():
    args = parse_args()
    if args.runtrain:
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
