#include <stocs.hpp>
#include <pose_clustering.hpp>

std::string repo_path = "/path/to/this/folder/pose_hypotheses_generation";

// rgbd parameters
float voxel_size = 0.005; // In m
float distance_threshold = 0.005; // for Congruent Set Matching and LCP computation
int ppf_tr_discretization = 5; // In mm
int ppf_rot_discretization = 5; // degrees
float edge_threshold = 0; // Not used
float class_threshold = 0.10; // Cut-off probability
float sample_dispersion = 0.9;

// stocs parameters
int number_of_bases = 100;
int maximum_congruent_sets = 200;

// camera parameters
std::vector<float> cam_intrinsics = {615.957763671875, 308.1098937988281, 615.9578247070312, 246.33352661132812}; //DENSELY-PACKED
float depth_scale = 1/8000.0f;

// clustering parameters
float acceptable_fraction = 0.1;
int maximum_pose_count = 10000;
float min_distance = 0.03;
float min_angle = 30;

int image_width = 640;
int image_height = 480;

class BaseGraph{
    public:
        std::vector<int> baseIds_;
        float invariant1_;
        float invariant2_;

        std::vector<Quadrilateral> congruent_quads;

        BaseGraph(std::vector<int> base_ids, float invariant1, float invariant2) {
        	baseIds_.clear();

			baseIds_.push_back(base_ids[0]);
			baseIds_.push_back(base_ids[1]);
			baseIds_.push_back(base_ids[2]);
			baseIds_.push_back(base_ids[3]);

			invariant1_ = invariant1;
			invariant2_ = invariant2;

			congruent_quads.clear();
        }
        ~BaseGraph(){}
}; // class BaseGraph


void run_stocs_estimation(std::string scene_path, 
					      std::string object_name, 
						  PPFMapType& ppf_map_preloaded) {

	std::string rgb_path = scene_path + "/rgb.png";
	std::string depth_path = scene_path + "/depth.png";
	std::string class_probability_path = scene_path + "/probability_maps/" + object_name + ".png";
	std::string edge_probability_path = scene_path + "/probability_maps/edge.png";
	std::string model_path = repo_path + "/models/" + object_name + "/model_search.ply";
	std::string output_pose_file = scene_path + "/pose_candidate_" + object_name + ".txt";

	std::vector<BaseGraph*> base_set;

	stocs::stocs_estimator stocs_ptr(model_path,
									ppf_map_preloaded,
									rgb_path,
									depth_path,
									class_probability_path,
									edge_probability_path,
									scene_path + "/dbg",
									cam_intrinsics,
                                    image_width, image_height,
									depth_scale,
									1.0f,
									voxel_size, distance_threshold,
									ppf_tr_discretization, ppf_rot_discretization,
									edge_threshold, class_threshold);

	// Step 1: Sample n bases on scene
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < number_of_bases; i++) {
		bool valid_base_found = false;

		std::vector<int> base_indices(4,-1);
		float invariant1, invariant2;
		std::vector<Point3D> segment;
		segment.clear();

        struct stat buffer;
	    if(stat (edge_probability_path.c_str(), &buffer) == 0)
		    valid_base_found =	stocs_ptr.sample_instance_base(base_indices, invariant1, invariant2, segment, sample_dispersion, i+1);
        else
		    valid_base_found =	stocs_ptr.sample_class_base(base_indices, invariant1, invariant2);

		if(valid_base_found) {
			BaseGraph *b = new BaseGraph(base_indices, invariant1, invariant2);
			base_set.push_back(b);
		}

		// get the number of base successfully sampled, store pixels for bases, get time for each sampling
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "Sampled " << base_set.size() << " bases in "
			<< std::chrono::duration_cast<micro>(finish - start).count()
			<< " microseconds\n";
	
	auto total_time = std::chrono::duration_cast<micro>(finish - start).count();

	// Step 2: Get corresponding congruent sets on the model for each of the sampled bases
    start = std::chrono::high_resolution_clock::now();
	for (auto base_iterator: base_set) {
		bool congruent_sets_found = false;

		congruent_sets_found = stocs_ptr.find_congruent_sets_on_model(base_iterator->baseIds_,
																	base_iterator->invariant1_, base_iterator->invariant2_,
																	&base_iterator->congruent_quads);
		// get number of congruent sets found, time for each computation
	}

	// Step 3: Sample a maximum of k congruent pairs from each base and get rigid transformations
	int total_congruent_set_found = 0;
	int base_number = 0;
	for (auto base_iterator: base_set) {
		int congruent_set_size = base_iterator->congruent_quads.size();

		if(congruent_set_size < maximum_congruent_sets){

			for (int i = 0; i < congruent_set_size; i++)
				stocs_ptr.get_rigid_transform_from_congruent_pair(base_iterator->baseIds_, base_iterator->congruent_quads[i], base_number);

		} 
		else {

			std::vector<int> c_set_indices(congruent_set_size);

			for (int i = 0; i < congruent_set_size; i++) 
				c_set_indices.push_back(i);
			
			std::random_shuffle ( c_set_indices.begin(), c_set_indices.end() );

			for (int i=0; i < maximum_congruent_sets; i++)
				stocs_ptr.get_rigid_transform_from_congruent_pair(base_iterator->baseIds_, base_iterator->congruent_quads[c_set_indices[i]], base_number);
		}

		total_congruent_set_found += congruent_set_size;
		base_number++;
	}
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "found " << total_congruent_set_found << " congruent sets in "
              << std::chrono::duration_cast<micro>(finish - start).count()
              << " microseconds\n";

    total_time += std::chrono::duration_cast<micro>(finish - start).count();

	// Verify all transforms to get the best pose
    start = std::chrono::high_resolution_clock::now();

	stocs_ptr.compute_best_transform();
	
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "evaluated transforms in "
              << std::chrono::duration_cast<micro>(finish - start).count()
              << " microseconds\n";

    total_time += std::chrono::duration_cast<micro>(finish - start).count();

	stocs_ptr.visualize_best_pose();
    PoseCandidate* best_pose = stocs_ptr.get_best_pose();

    std::vector< PoseCandidate* > hypotheses_set;
	std::vector< PoseCandidate* > clustered_hypotheses_set;
	float best_score;

	hypotheses_set = stocs_ptr.get_pose_candidates();
	best_score = stocs_ptr.get_best_score();

	clustering::greedy_clustering(hypotheses_set,
				acceptable_fraction,
				best_score,
				maximum_pose_count,
				min_distance,
				min_angle,
				{0, 0, 0},
				clustered_hypotheses_set);

	// store the clustered poses
	std::ofstream out_file_ptr;
	out_file_ptr.open (output_pose_file, std::ofstream::out);

	using PointType = pcl::PointXYZRGBNormal;
	pcl::PointCloud<PointType>::Ptr model_cloud(new pcl::PointCloud<PointType> () );
	pcl::io::loadPLYFile(model_path, *model_cloud);
	int count = 0;
    for (auto pose_it: clustered_hypotheses_set) {

    	pcl::PointCloud<PointType>::Ptr segment_cloud(new pcl::PointCloud<PointType> () );
		pcl::PointCloud<PointType>::Ptr transformed_model_cloud(new pcl::PointCloud<PointType> () );

		pcl::io::loadPLYFile(scene_path + "/dbg/segment_" + object_name + "_" + std::to_string(pose_it->base_index) + ".ply", *segment_cloud);

		pcl::transformPointCloudWithNormals(*model_cloud, *transformed_model_cloud, pose_it->transform);
		pcl::io::savePLYFile(scene_path + "/dbg/model_" + std::to_string(count) + "_pre.ply", *transformed_model_cloud);

    	Eigen::Matrix4f final_transform, offset_transform;

    	// Performing ICP
    	clustering::point_to_plane_icp(segment_cloud, transformed_model_cloud, offset_transform);
    	offset_transform = offset_transform.inverse().eval();
    	final_transform = offset_transform *  pose_it->transform;

		// Visualize transformed model
		pcl::transformPointCloud(*model_cloud, *transformed_model_cloud, final_transform);
		pcl::io::savePLYFile(scene_path + "/dbg/model_" + std::to_string(count) + "_post.ply", *transformed_model_cloud);

    	out_file_ptr << final_transform(0, 0) << " " << final_transform(0, 1) << " " << final_transform(0, 2) << " " << final_transform(0, 3) << " " << 
    				final_transform(1, 0) << " " << final_transform(1, 1) << " " << final_transform(1, 2) << " " << final_transform(1, 3) << " " << 
    				final_transform(2, 0) << " " << final_transform(2, 1) << " " << final_transform(2, 2) << " " << final_transform(2, 3) << std::endl;

    	count++;
    }
    out_file_ptr.close();
}

int
main (int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Enter scene path and object name as arguments!" << std::endl;
        exit(-1);
    }

	std::string scene_path = argv[1];
	std::string object_name = argv[2];

	std::cout << "############# LOADING OBJECT MAPS ################" << std::endl;

	// load PPF map
	PPFMapType model_map;
    std::string model_map_path = repo_path + "/models/" + object_name + "/ppf_map";
    rgbd::load_ppf_map(model_map_path, model_map);

    std::cout << "############# LOADING OBJECT COMPLETE ################" << std::endl;

	system(("rm -rf " + scene_path + "/dbg").c_str());
	system(("mkdir " + scene_path + "/dbg").c_str());

	std::cout << "############# RUNNING STOCS for Scene: " << scene_path << ", Object: " << object_name << " ##############" << std::endl;

	run_stocs_estimation(scene_path, object_name, model_map);

 	return 0;
}
