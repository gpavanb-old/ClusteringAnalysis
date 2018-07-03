import csv
import numpy as np
from sklearn.cluster import DBSCAN

def load_part_data(filename):
  with open(filename) as f:
    reader = csv.reader(f,delimiter=' ')
    data = []
    for row in reader:
      # Remove blank spaces
      filt_row = filter(None,row)
      data.append(filt_row)

  # Store as numpy array
  np_data = np.array(data)
  # Convert to float
  np_data = np_data.astype(np.float)
  return np_data

def coords(part_data):
  return part_data[:,1:4]

def dbscan(X,dist,min_size):
  # Compute DBSCAN
  db = DBSCAN(eps=dist, min_samples=min_size, metric ='euclidean').fit(X)
  labels = db.labels_

  return labels,db

def visualize_dbscan(X,labels,db):
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt

  # Create plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Get core samples
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True

  # Black removed and is used for noise instead.
  unique_labels = set(labels)
  colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
  for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    n = xy.shape[0]
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=30, marker='o', c=tuple(col),
             edgecolors='k')

    # Dont show outliers - too dense
    #xy = X[class_member_mask & ~core_samples_mask]
    #n = xy.shape[0]
    #ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=1, marker='o', c=tuple(col),
    #         edgecolors='k')

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.show()
