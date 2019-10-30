#include <stocs.hpp>
#include <pose_clustering.hpp>

std::string repo_path = "/path/to/this/folder/pose_hypotheses_generation";

// rgbd parameters
float voxel_size = 0.005; // In m

// camera parameters
std::vector<float> cam_intrinsics = {619.2578, 320.0, 619.2578, 240.0};
float depth_scale = 1/10000.0f;

int image_width = 640;
int image_height = 480;

void run_icp_refinement(std::string scene_path, 
					    std::string object_name
						) {

	std::string rgb_path = scene_path + "/rgb.png";
	std::string depth_path = scene_path + "/depth.png";
	std::string model_path = repo_path + "/models/" + object_name + "/model_search.ply";
	std::string object_pose_filepath = scene_path + "/pose_candidates_" + object_name + ".txt";
	std::string output_pose_file = scene_path + "/refined_candidates_" + object_name + ".txt";

	using PointType = pcl::PointXYZRGBNormal;
	using PCLPointCloud = pcl::PointCloud<pcl::PointXYZRGBNormal>;

	PCLPointCloud::Ptr model_cloud (new PCLPointCloud);
	PCLPointCloud::Ptr segment_cloud (new PCLPointCloud);
	PCLPointCloud::Ptr transformed_model_cloud (new PCLPointCloud);

	cv::Mat rgb_image;
  	cv::Mat depth_image;
  	cv::Mat surface_normals;
  	cv::Mat_<cv::Vec3f> surface_normals3f;

  	// read model
	pcl::io::loadPLYFile(model_path, *model_cloud);
	std::cout << "|M| = " << model_cloud->points.size() << std::endl;

	// read segment
  	rgb_image = cv::imread(rgb_path, CV_LOAD_IMAGE_COLOR);
  	depth_image = cv::imread(depth_path, CV_16UC1);
	cv::Mat K = (cv::Mat_<double>(3, 3) << cam_intrinsics[0], 0, cam_intrinsics[1], 0, cam_intrinsics[2], cam_intrinsics[3], 0, 0, 1);
	cv::rgbd::RgbdNormals normals_computer(depth_image.rows, depth_image.cols, CV_32F, K, 5, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
	normals_computer(depth_image, surface_normals);
	surface_normals.convertTo(surface_normals3f, CV_32FC3);

	for (int i = 0; i < depth_image.rows; i++) {
		for (int j = 0; j < depth_image.cols; j++) {

	    float depth = (float)depth_image.at<unsigned short>(i,j)*depth_scale;

	    PointType pt;
	    pt.x = (float)((j - cam_intrinsics[1]) * depth / cam_intrinsics[0]);
	    pt.y = (float)((i - cam_intrinsics[3]) * depth / cam_intrinsics[2]);
	    pt.z = depth;

	    cv::Vec3b rgb_val = rgb_image.at<cv::Vec3b>(i,j);
	    uint32_t rgb = ((uint32_t)rgb_val.val[2] << 16 | (uint32_t)rgb_val.val[1] << 8 | (uint32_t)rgb_val.val[0]);
	    pt.rgb = *reinterpret_cast<float*>(&rgb);

	    cv::Vec3f cv_normal = surface_normals3f(i, j);

	    pt.normal[0] = cv_normal[0];
	    pt.normal[1] = cv_normal[1];
	    pt.normal[2] = cv_normal[2];

	    segment_cloud->points.push_back(pt);

	  }
	} 

	pcl::VoxelGrid<PointType> sor;
	sor.setInputCloud (segment_cloud);
	sor.setLeafSize (voxel_size, voxel_size, voxel_size);
	sor.filter (*segment_cloud);

	// read poses, refine and store final poses
	std::ifstream object_pose_file;
	object_pose_file.open (object_pose_filepath, std::ifstream::in);

	std::ofstream out_file_refined_ptr;
	out_file_refined_ptr.open (output_pose_file, std::ofstream::out);

	Eigen::Matrix4f obj_pose;
	obj_pose.setIdentity();

	int count = 0;
	while (object_pose_file >> obj_pose(0,0) >> obj_pose(0,1) >> obj_pose(0,2) >> obj_pose(0,3)
            >> obj_pose(1,0) >> obj_pose(1,1) >> obj_pose(1,2) >> obj_pose(1,3)
            >> obj_pose(2,0) >> obj_pose(2,1) >> obj_pose(2,2) >> obj_pose(2,3)) {

		pcl::transformPointCloudWithNormals(*model_cloud, *transformed_model_cloud, obj_pose);
		pcl::io::savePLYFile(scene_path + "/dbg/model_" + std::to_string(count) + "_pre.ply", *transformed_model_cloud);
		pcl::io::savePLYFile(scene_path + "/dbg/segment_" + std::to_string(count) + "_pre.ply", *segment_cloud);


		Eigen::Matrix4f final_transform, offset_transform;

    	// Performing ICP
    	clustering::point_to_plane_icp(segment_cloud, transformed_model_cloud, offset_transform);
    	offset_transform = offset_transform.inverse().eval();
    	final_transform = offset_transform *  obj_pose;

		// Visualize transformed model
		pcl::transformPointCloud(*model_cloud, *transformed_model_cloud, final_transform);
		pcl::io::savePLYFile(scene_path + "/dbg/model_" + std::to_string(count) + "_post.ply", *transformed_model_cloud);

    	out_file_refined_ptr << final_transform(0, 0) << " " << final_transform(0, 1) << " " << final_transform(0, 2) << " " << final_transform(0, 3) << " " << 
    				final_transform(1, 0) << " " << final_transform(1, 1) << " " << final_transform(1, 2) << " " << final_transform(1, 3) << " " << 
    				final_transform(2, 0) << " " << final_transform(2, 1) << " " << final_transform(2, 2) << " " << final_transform(2, 3) << std::endl;

    	count++;
	}

}

int
main (int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Enter scene path and object name as arguments!" << std::endl;
        exit(-1);
    }

	std::string scene_path = argv[1];
	std::string object_name = argv[2];

	system(("rm -rf " + scene_path + "/dbg").c_str());
	system(("mkdir " + scene_path + "/dbg").c_str());

	std::cout << "############# RUNNING ICP for Scene: " << scene_path << ", Object: " << object_name << " ##############" << std::endl;

	run_icp_refinement(scene_path, object_name);

 	return 0;
}
