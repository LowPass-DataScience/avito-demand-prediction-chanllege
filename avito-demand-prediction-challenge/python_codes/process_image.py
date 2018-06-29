# Process raw image located in each chunked image folders
# e.g. {train_jpg_1} means chunk #1 of the training image set
# Image features consists of simple aggregated features like
# brightness and colorfulness in the spatial-temporal domain
# and average pixel width and number of coners in the transformed
# domain. In addition, feature vector is extracted through pre-trained
# tensorflow image models trained on ImageNet. Finally, these features
# are joined together to form the image feature dataset
#
# Chen Chen

## Load packages
import pandas as pd 
import numpy as np
import os 
from os import scandir, listdir
from scipy.stats import itemfreq, entropy
# Image processing libraries
import cv2
import imutils
from skimage import feature
from skimage.exposure import is_low_contrast
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte
from skimage.morphology import disk
# data IO
import dask as da
import dask.dataframe as dd
# Multi-threading
import multiprocessing as mp
from multiprocessing import Process, Pool, cpu_count, Array, JoinableQueue
# Progress visualization
from tqdm import tqdm
# Error handling and argument io
import sys
# Garbage collection
import gc
# Tensorflow - image feature extractor
import tensorflow as tf
import tensorflow_hub as hub


class ImageProcessor(Process):
    def __init__(self, task_queue, thread_id, module):
        Process.__init__(self)
        self.task_queue = task_queue
        self.thread_id = thread_id
        self.module = module        

    def image_analysis(self, image):
        ## basic image features
        img_width = image.shape[0]
        img_height = image.shape[1]
        img_size = img_width * img_height
        if img_height != 0:
            aspect_ratio = img_width / img_height
        else:
            aspect_ratio = 0

        # mean color and color variance
        mean_color_B, mean_color_G, mean_color_R, _ = cv2.mean(image)

        ## colorfulness
        (B, G, R) = cv2.split(image.astype("float"))
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
         # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric
        colorfulness = stdRoot + (0.3 * meanRoot)

        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))

        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        # Convert image to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # average pixel width
        edges_sigma1 = feature.canny(image, sigma=3)
        apw = float(np.sum(edges_sigma1)) / img_size * 100.0

        # Count number of over and under exposed pixels through histogram
        cnts, _ = np.histogram(image, bins=35)
        over_exposed = sum(cnts[-3:])
        under_exposed = sum(cnts[:4])
        pixel_count = sum(cnts)
        light_percent = round((float(over_exposed)/pixel_count)*100, 2)
        dark_percent = round((float(under_exposed)/pixel_count)*100, 2)

        # check entropy of histogram
        entropy_hist = entropy(cnts)

        # check entropy of image 
        entropy_image = shannon_entropy(image)

        # check number of detected corners
        num_corners = len(feature.corner_peaks(feature.corner_harris(image), min_distance=2))

        # check number of local peaks
        num_local_peaks = len(feature.peak_local_max(image, min_distance=2))

        # check blurriness score
        blurriness = cv2.Laplacian(image, cv2.CV_64F).var()

        returnList = [
            light_percent, 
            dark_percent, 
            entropy_hist, 
            entropy_image, 
            num_corners, 
            num_local_peaks, 
            blurriness,
            mean_color_B,
            mean_color_G,
            mean_color_R,
            img_size,
            img_width,
            img_height,
            aspect_ratio,
            apw,
            colorfulness
        ]
        return returnList
        
    def run(self):
        ## Initialize tensorflow
        print(f'Initializing worker #{self.thread_id}')
        #import tensorflow as tf
        module = self.module
        #tf.logging.set_verbosity(tf.logging.WARN)
        height, width = hub.get_expected_image_size(module)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        # Define image feature extraction work
        tfImage = tf.placeholder(tf.float32, shape=(1, height, width, 3))
        featureVec = module(tfImage)

        ## Handle jobs
        while True:
            # Populate new job and start           
            jobPath = self.task_queue.get()
            if jobPath is None:
                self.task_queue.task_done()
                print(f'All done, shutting down worker #{self.thread_id}')
                break
            else:
                ## Load data
                imgID = jobPath.split("/")[-1]
                saveFile = f'{featurePath}/{imgID.replace("jpg", "h5x")}'
                if not os.path.exists(saveFile):
                    if VERBOSE > 0: print(f'#{self.thread_id} INFO: starting new job {imgID}')
                    image = cv2.imread(jobPath)
                    ## Initialize dataset
                    features = pd.DataFrame()
                    features['image'] = [imgID]
                    
                    # Check is image is valid
                    if image is None or len(image) < 16:
                        print(f'#{self.thread_id} WARN: {imgID} invalid, skipping')
                        features.to_hdf(f'{featurePath}/{imgID[:-4]}.h5', key='features', **USE_HDF5_COMPRESSION_ARG)
                        gc.collect()
                        self.task_queue.task_done()
                        continue
                    else:
                        ### image is valid
                        ## Process basic image features
                        try:
                            analysis_raw = self.image_analysis(image)
                        except:
                            errInfo = sys.exc_info()[:2]
                            print(f'#{self.thread_id} WARN: {imgID} analysis has exception {errInfo[0].__name__}: {errInfo[1]}')
                            # Fallback, save data without features
                            features.to_hdf(f'{featurePath}/{imgID[:-4]}.h5', key='features', **USE_HDF5_COMPRESSION_ARG)
                            gc.collect()
                            self.task_queue.task_done()
                            continue
                        
                        features['PCENT_over_exposed'] = analysis_raw[0]
                        features['PCENT_under_exposed'] = analysis_raw[1]
                        features['PCENT_normal_exposed'] = 100.0 - features['PCENT_over_exposed'] - features['PCENT_under_exposed']
                        features['entropy_hist'] = analysis_raw[2]
                        features['entropy_image'] = analysis_raw[3]
                        features['CNT_corners'] = analysis_raw[4]
                        features['CNT_local_peaks'] = analysis_raw[5]
                        features['blurriness'] = analysis_raw[6]
                        features['mean_color_B'] = analysis_raw[7]
                        features['mean_color_G'] = analysis_raw[8]
                        features['mean_color_R'] = analysis_raw[9]
                        features['image_size'] = analysis_raw[10]
                        features['img_width'] = analysis_raw[11]
                        features['img_height'] = analysis_raw[12]
                        features['aspect_ratio'] = analysis_raw[13]
                        features['average_pixel_width'] = analysis_raw[14]
                        features['colorfulness'] = analysis_raw[15]
                        # display(features.head())
                        ## Extract image feature vector
                        image = cv2.resize(image, (height, width))
                        image = image.astype(np.float) / np.iinfo(image.dtype).max
                        image = image.reshape((1, height, width, 3))
                        feature_vector = np.squeeze(sess.run(featureVec, feed_dict={tfImage: image}))
                        ## Finally, merge into a single dataframe
                        tmpDf = pd.DataFrame([feature_vector], columns=[f'image_feature_{i+1}' for i in range(len(feature_vector))])
                        tmpDf['image'] = imgID
                        tmpDf = pd.merge(features, tmpDf, on='image')
                        ## Prepare to save data
                        tmpDf.to_hdf(f'{featurePath}/{imgID[:-4]}.h5', key='features', **USE_HDF5_COMPRESSION_ARG)
                else:
                    if VERBOSE > 0: print(f'#{self.thread_id} INFO: skipping finished job {imgID}')
                
                ## Clean up and finish
                gc.collect()
                self.task_queue.task_done()
        
        ## Close session and return
        sess.close()
        return

if __name__ == "__main__":
    args = sys.argv
    sourceImgDir = str(args[1])
    # Parallel training parameters
    nInstances = int(args[2])
    instanceID = int(args[3])
    if len(args) == 5:
        nThread = int(args[4])
    else:
        nThread = cpu_count()

    # Tune flags for Intel MKL
    # Reference: https://www.tensorflow.org/performance/performance_guide#tensorflow_with_intel%C2%AE_mkl_dnn
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    # Control flags
    USE_HDF5_COMPRESSION_ARG = {
        'format': 'table',
        'complib': 'blosc:zstd', 
        'complevel': 6
    }
    VERBOSE = 0
    
    # Enable gc
    gc.enable()

    ## Initialize tensorflow
    # ImageNet Pre-trained models
    mobilenet_v1 = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/quantops/feature_vector/1"  # dim 512
    mobilenet_v2 = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1"           # dim 1280
    inception_resnet_v2 = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"     # dim 1536
    hubModule = mobilenet_v1
    tf.logging.set_verbosity(tf.logging.WARN)
    module = hub.Module(hubModule)

    # Specify path variables
    dataRootPath = '../../../data/avito-demand-prediction/images'
    imagePath = f'{dataRootPath}/{sourceImgDir}'
    featurePath = f'{dataRootPath}/{sourceImgDir}'

    # Check feature path
    if not os.path.exists(featurePath):
        os.makedirs(featurePath)

    # Get joblist
    fileList = np.array_split(
        np.array(
            sorted(listdir(imagePath)),
            dtype=str
        ),
        nInstances
    )[instanceID-1]
    fileList = np.array([f'{imagePath}/{img}' for img in fileList])
    print(f'Process #{instanceID}/{nInstances} with {nThread} threads, total jobs {len(fileList)}')
    print(f'Feature output path {featurePath}')

    # Submit job
    worker_list = []
    jobList = fileList
    pbar = tqdm(total=nThread+len(jobList))
    job_queue = JoinableQueue()

    # Initialize worker
    for id in range(nThread):
        worker = ImageProcessor(job_queue, id, module)
        worker.start()
        worker_list.append(worker)

    # Submit all jobs to queue
    for job in jobList:
        job_queue.put(job)
    # Append empty jobs at last to make sure worker process terminates
    for i in range(nThread):
        job_queue.put(None)

    # Wait for finish, with fancy progress bar
    while job_queue.qsize() > nThread:
        pbar.update(pbar.total - job_queue.qsize() - pbar.n)
    job_queue.join()
    for w in worker_list:
        w.join()
    pbar.update(pbar.total - pbar.n + 1)
