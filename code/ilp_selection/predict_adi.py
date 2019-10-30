import pickle
import numpy as np
import os
from math import sqrt

scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

dataset_path = '/path/to/dataset/'
object_names = ["dove", "toothpaste"]
model = pickle.load(open("GTB.p", 'rb'))

all_preds = []
all_features = []

for scene_id in scenes:
	for object_id, object_name in enumerate(object_names):
		features = []
		curr_features = []
		feature_filepath = dataset_path + '/%05d/data_%s.txt' % (scene_id, object_name)
		real_filepath = dataset_path + '/%05d/predicted_scores_%s.txt' % (scene_id, object_name)

		with open(feature_filepath, 'r') as f:
			for line in f:
				f1, f2, f3, f4, f5 = line.split()
				features.append([float(object_id), float(f1), float(f2), float(f3), float(f4), float(f5)])
				all_features.append(float(f1) + float(f2))
				curr_features.append(float(f1) + float(f2))

		X = np.array(features)
		pred = model.predict(X)

		if os.path.exists(real_filepath):
			os.remove(real_filepath)

		with open(real_filepath, 'w') as f:
			for i in range(0, len(pred)):
				if curr_features[i] < 20:
					all_preds.append(1.0)
					f.write('%f\n' % 1.0)
				else:
					all_preds.append(float(pred[i]))
					f.write('%f\n' % pred[i])

P = np.array(all_preds)

print ('Before removing zeros: ', len(P))
I = np.nonzero(all_features)
P = P[I]
print ('After removing zeros: ', len(P))
print("Predictions min/max are", np.amin(P), np.amax(P))