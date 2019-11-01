More details coming soon!

To Run the entire pipeline:

1) Install dependencies for segmentation module, download an already trained model from the link below and run prediction based on instructions within "segmentation" module. You should be able to see semantic and boundary predictions within "dbg" folder of the scene.
https://www.dropbox.com/s/npw4owb6buipmid/latest_net_D_L.pth?dl=0
 
2) Compile "pose hypotheses_generation" and run according to instructions for "Running code for one object on one scene". You should be able to see a list of pose candidates for the specified object in the specified scene directory.

3) Compile and run "pose_selection" based on the instructions. This should compute features for each pose hypotheses.

4) Next run "predict_adi.py" to get a single quality score for each hypotheses. Dataset path and scene numbers needs to be set in this file.

5) Run "select_poses.py". This selects the poses using the ILP-solver and the final solution will be available in the scene folder. 
