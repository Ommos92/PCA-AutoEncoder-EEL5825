from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np 
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image

def cumulative_variance(data):
    k_total = 0
    data = grayscaleall_img(data)
    for i in range(data.shape[0]):
        pca = PCA().fit(data[i,:,:])
        c_variance = np.cumsum(pca.explained_variance_ratio_)*100
        k = np.argmax(c_variance>FLAGS.variance)
        k_total = k + k_total
    k_avg = k_total/6000
    return k_avg

def grayscaleall_img(airplane_images):
    #Grayscale image for PCA
    img = airplane_images
    img_sum = img.sum(axis=3)
    gray_img = img_sum/img_sum.max()
    print(gray_img.shape)
    return gray_img

def grayscale_img(airplane_images, img_number):
    #Grayscale image for PCA
    img = airplane_images[img_number,:,:,:]
    print(img.shape)

    img_sum = img.sum(axis=2)
    gray_img = img_sum/img_sum.max()
    return gray_img

def plot_k(k,img):
    ipca = IncrementalPCA(n_components=k)
    img_rcn = ipca.inverse_transform(ipca.fit_transform(img))
    plt.imshow(img_rcn,cmap=plt.cm.gray)

def run_main(FLAGS):

    #load Cifar10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print('Training data shape:', x_train.shape)
    print('Test Data Shape:', x_test.shape)

    print("Training Label shape:",y_train.shape)
    print("Test Label shape:",y_test.shape)

    # Steps of PCA
    # 1. Standardize the range of continuous initial variables
    # 2. Compute the covariance matrix to identify correlations
    # 3. Compute the eigenvectors and eigenvales of the covariance matrix to identify the principle compoentns
    # 4. Create a feature vector to decide which pricipal components to keep
    # 5. Recast the data alon hte principal component axes

    #Combine X train and X test
    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train,y_test))
    print(X.shape,Y.shape)

    #Extract only airplane images
    airplane_index = np.array(np.where(Y[:,0]==0))
    #Get all airplane images
    airplane_images = X[airplane_index[0,:]]
    print("Number of Airplane Images: {}".format(airplane_images.shape))

    '''
    #Plot some airplane images
    plt.figure(figsize=(10,10))
    n_images = 64
    for i in range(n_images):  
        plt.subplot(8, 8, i+1)
        plt.imshow(airplane_images[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.show()
    '''
        
    gray_img = grayscale_img(airplane_images, 63)
    #Plot the image
    plt.figure(figsize=[12,8])
    plt.imshow(gray_img, cmap=plt.cm.gray)
    plt.show()
    #PCA per Channel
    
    pca = PCA()
    #Find Variance overcomplete dataset
    k_avg = cumulative_variance(airplane_images)
    print("Average Components for 70% Variance: {}".format(k_avg))

    #Fit data onto PCA model
    pca.fit(gray_img)
    print("Number of components used to capture variance target: {}".format(pca.n_components_))

    #Get the cumulative variance
    c_variance = np.cumsum(pca.explained_variance_ratio_)*100
    #Number of Principle components for 70% variacne
    k = np.argmax(c_variance>FLAGS.variance)


    print("Number of components for 70% variance: {}".format(k))

    plt.title('Cumulative Explained Variance explained by the components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=FLAGS.variance, color="r", linestyle="--")
    ax = plt.plot(c_variance)

    #Image reconstruction for image with 70% Variance target
    ipca = IncrementalPCA(n_components=k)
    image_reconstruction = ipca.inverse_transform(ipca.fit_transform(gray_img))

    # Plotting the reconstructed image
    #plt.figure(figsize=[12,8])
    #plt.imshow(image_reconstruction,cmap = plt.cm.gray)
    #plt.show()
    #print('P')
    ks = [2,5,8,11,14,17,20,23,26,29,30,32]
    plt.figure(figsize=[15,9])
    for i in range(len(ks)):
        plt.subplot(4,3,i+1)
        plot_k(ks[i],gray_img)
        plt.title("Components: {}".format(ks[i]))

    plt.subplots_adjust(wspace=0.2, hspace=0.35)
    plt.show()
    print("2")

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--variance',
                        type=float, default=70,
                        help='PCA Variance')
    
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    