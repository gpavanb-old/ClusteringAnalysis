from clustering import *
from sklearn.datasets.samples_generator import make_blobs

def main():
  filename = 'data/tracking_parts_0066000'
  part_data = load_part_data(filename)
  crds = coords(part_data)

  # Test DBSCAN
  # Minimum distance to call a cluster
  eps = 0.88
  # It takes two to cluster
  min_neighbors = 2
  labels,db = dbscan(crds,eps,min_neighbors)
  visualize_dbscan(crds,labels,db) 

if __name__ == "__main__":
    main()
