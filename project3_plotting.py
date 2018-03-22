import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import euclidean
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


def plot_various_k(title, X, ylim=None,reps = 3, step_size= 10, min_k=2, max_k= 100):
                          
    nodes = np.linspace(min_k,max_k,step_size)
    scores = np.zeros([step_size,reps])
    sil_scores = np.zeros([step_size, reps])
    i = 0
    
    for node in nodes:
        model = KMeans(n_clusters = int(node))
        for j in range (reps):
            model.fit(X)
            scores[i][j] = model.inertia_
            sil_scores[i][j] = silhouette_score(X, model.labels_)
        i += 1
        
    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)
    
    sil_scores_mean = np.mean(sil_scores, axis=1)
    sil_scores_std = np.mean(sil_scores, axis=1)
   

    plt.figure()
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.fill_between(nodes, scores_mean - scores_std,
                     scores_mean + scores_std, alpha=0.1, color="g")
 
    plt.plot(nodes, scores_mean, '--', color="g")
    plt.show()
    
    plt.figure()
    plt.title(title)
    
    
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("S Score")
    plt.fill_between(nodes, sil_scores_mean - sil_scores_std,
                     sil_scores_mean + sil_scores_std, alpha=0.1, color="r")
 
    plt.plot(nodes, sil_scores_mean, '--', color="r")
    plt.show()

    return plt

    # Visualize the results on PCA-reduced data
def plot_kmeans_1(datasetName,data, n_digits):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(n_clusters=n_digits)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    title = 'K-means clustering on {0}dataset, K= {1}\n Centroids are marked with white cross'.format(datasetName, n_digits)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_various_k_EM(title, X, ylim=None,reps = 3, step_size= 10, min_k=2, max_k= 100):

                      
    nodes = np.linspace(min_k,max_k,step_size)
    scores = np.zeros([step_size,reps])
    i = 0
    
    for node in nodes:
        model = GaussianMixture(n_components = int(node))
        for j in range (reps):
            model.fit(X)
            scores[i][j] = model.lower_bound_
  
        i += 1
        
    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("K")
    plt.ylabel("Log-Likelihood")
    plt.grid()

    plt.fill_between(nodes, scores_mean - scores_std,
                     scores_mean + scores_std, alpha=0.1, color="g")
 
    plt.plot(nodes, scores_mean, '--', color="g")

    return plt

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(title, gmm, data, label=True, ax=None):
    X = PCA(n_components=2).fit_transform(data)
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.title(title)

def pca_num_components_plot(title,pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('title')

def plot_eigenvalue_distribution(title, eigenValues, bins):
    plt.hist(eigenValues, 20, facecolor='blue')
    plt.title(title)
    plt.xlabel('Eiganvalue')
    plt.ylabel('Frequency')

def autoEncoderStuff(input_dim, x_train, x_test, num_hidden_layers=1):
    input_dim = x_train.shape[1]
    encoding_dim = input_dim
    print(encoding_dim)

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="relu", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    if(num_hidden_layers >1):
        encoder = Dense(int(encoding_dim / 2), activation="sigmoid")(encoder)
    if(num_hidden_layers >2):
        encoder = Dense(int(encoding_dim / 2), activation="sigmoid")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='sigmoid')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    nb_epoch = 100
    nb_epoch = 100
    batch_size = 32
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="model.h5",
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    history = autoencoder.fit(x_train, x_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right');
    plt.show()
    
    predictions = autoencoder.predict(x_test)
    sparcity = ((np.count_nonzero(predictions==0))/predictions.size)*100
    return (predictions,sparcity)

def plot_autoEncoder_projections(title,data):
    tsne = TSNE(2, init='pca', random_state=0, verbose=1, n_iter=500)
    Y = tsne.fit_transform(data)
    print('plotting')
    plt.figure()
    plt.title(title)
    plt.scatter(Y[:, 0], Y[:, 1],cmap=plt.cm.Spectral)
    plt.show()