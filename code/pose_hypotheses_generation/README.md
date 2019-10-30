### Dependency
1. OpenCV
2. PCL

### Installation
1. Download the repository.
2. mkdir build
3. cd build
4. cmake ../
5. make

### Inputs
1. RGB and depth images
2. Per-pixel object class probability (scaled to range 0-10000 and stored as uint16) and edge map.

### Output
1. ```pose_candidate_{object_name}.txt``` 6D pose candidates of the object (3 rows of the transformation matrix) stored in row-major order.

### Running code for one object on one scene
1. Set the ```repo_path``` in files ```model_preprocess.cpp``` and ```stocs_match_one_object.cpp```
2. Preprocess the 3d model
```./build/model_preprocess "dove"```
3. Run pose estimation
```./build/stocs_single "{path_to_scene_folder}/" "dove"```

# Code citation
The code is built upon:
1) https://github.com/cmitash/model_matching
Robust 6D Object Pose Estimation with Stochastic Congruent Sets, Chaitanya Mitash, Abdeslam Boularias, Kostas Bekris, British Machine Vision Conference 2018.
2) https://github.com/nmellado/Super4PCS
Super4PCS: Fast Global Pointcloud Registration via Smart Indexing Nicolas Mellado, Dror Aiger, Niloy J. Mitra Symposium on Geometry Processing 2014.