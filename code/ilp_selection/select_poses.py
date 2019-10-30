from gurobipy import *

scene_location = 'path/to/dataset/'

def printSolution(scene_path):
	poses = []
	for object_id, object_name in enumerate(object_names):
		poses.append([])
		pose_path = os.path.join(scene_path, 'pose_hypotheses_%s.txt' % object_name)
		best_pose_path = os.path.join(scene_path, 'best_hypotheses_%s.txt' % object_name)
		
		if os.path.exists(best_pose_path):
			os.remove(best_pose_path)
		
		with open(pose_path, 'r') as f:
			for line in f:
				poses[object_id].append(line.split())

	if m.status == GRB.Status.OPTIMAL:
		print('Score: %g' % m.objVal)
		print('\nSelection:')
		selectionx = m.getAttr('x', selection)
		for p in selected_poses:
			if selection[p].x == 1:

				# get object name and id
				obj_name = p[0]
				obj_id = -1
				for object_id, object_name in enumerate(object_names):
					if(obj_name == object_name):
						obj_id = object_id

				save_path = os.path.join(scene_path, 'best_hypotheses_%s.txt' % obj_name)
				with open(save_path, 'a') as f:
					print (p[1])
					f.write('%s %s %s %s %s %s %s %s %s %s %s %s\n' % (poses[obj_id][p[1]][0], poses[obj_id][p[1]][1], poses[obj_id][p[1]][2],
																	poses[obj_id][p[1]][3], poses[obj_id][p[1]][4], poses[obj_id][p[1]][5],
																	poses[obj_id][p[1]][6], poses[obj_id][p[1]][7], poses[obj_id][p[1]][8],
																	poses[obj_id][p[1]][9], poses[obj_id][p[1]][10], poses[obj_id][p[1]][11]))
				print('%s %g %f' % (p, selectionx[p], selected_poses[p]))
	else:
		print('No solution')

object_names = ['dove', 'toothpaste']
scene_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
num_instances = [[8, 6], [10, 5], [10, 6], [7, 6], [10, 5], [10, 6], [10, 6], [10, 6], [10, 8], [10, 8], [10, 8], [10, 6], [10, 8], [10, 7], [8, 6], [10, 6], [10, 5], [10, 5], [9, 6], [6, 5], [5, 6], [7,8], [6, 6], [8, 7], [7, 8], [8, 9], [9, 7], [9, 7], [7, 7], [5, 7]]

for scene_num, scene_id in enumerate(scene_ids):
	
	print ('scene: ', scene_num)
	selected_poses = dict()
	m = Model("pose_selection")
	num_candidates = [0, 0]
	
	for object_id, object_name in enumerate(object_names):
		scene_path = os.path.join(scene_location, '%05d/predicted_scores_%s.txt' % (int(scene_id), object_name))
		with open(scene_path, 'r') as f:
			for id, score in enumerate(f):
				 selected_poses[object_name, id] = 1.0 - float(score.strip())
				 num_candidates[object_id] = num_candidates[object_id] + 1

	for i in range(1, len(num_candidates)):
		num_candidates[i] = num_candidates[i-1]
	num_candidates[0] = 0

	selection = dict()
	for pid, pose in enumerate(selected_poses):
  		selection[pose] = m.addVar(name=str(pid), vtype=GRB.BINARY)
	
	m.setObjective(sum(selection[p]*selected_poses[p] for p in selected_poses), GRB.MAXIMIZE)

	# Setting number of instances constraints
	for object_id, object_name in enumerate(object_names):
		max_poses = num_instances[scene_num][object_id]
		expr = LinExpr()
		for sel in selection.keys():
			if(sel[0] == object_name):
				expr = expr + selection[sel]

		m.addConstr(expr <= max_poses, "pose_sum" + object_name)

	# Setting collision constraints
	collision_filepath = os.path.join(scene_location, '%05d/collisions.txt' % int(scene_id))
	with open(collision_filepath, 'r') as f:
		for lid, line in enumerate(f):
			ob_id1, ob_index1, ob_id2, ob_index2 = line.split()
			ob_index1 = int(ob_index1) - num_candidates[int(ob_id1)]
			ob_index2 = int(ob_index2) - num_candidates[int(ob_id2)]

			key1 = object_names[int(ob_id1)], int(ob_index1)
			key2 = object_names[int(ob_id2)], int(ob_index2)
			if key1 in selected_poses and key2 in selected_poses:
				expr = LinExpr()
				expr = expr + selection[object_names[int(ob_id1)], int(ob_index1)]
				expr = expr + selection[object_names[int(ob_id2)], int(ob_index2)]
				m.addConstr(expr <= 1, "collision" + str(lid))

	m.optimize()
	scene_path = os.path.join(scene_location, '%05d' % int(scene_id))
	printSolution(scene_path)