import numpy as np
import panda_robot
from grab_and_moveout_single import MyRobotPanda
from object_identification import from_image_to_position
import rospy
import os
import cv2
from python_tsp.exact import solve_tsp_dynamic_programming


FILE_2_ANALYSE = os.path.join("/","tmp", "camera_save_tutorial","default_camera_link_my_camera(1)-0000.jpg")
PATH_SAVE =os.path.join("img_exe_saved_rt")

def extract_camera_img():
    # Extract image  from saved destination
    img = cv2.imread(FILE_2_ANALYSE)
    return img
    
    
def generate_map_for_tsp(green : tuple, red: tuple):
    """
    !! GENERATE OPEN PROBLEM MAP !!
    """
    pt3d_green, pt3d_green_goal = green
    pt3d_red, pt3d_red_goal = red
    
    N_GREEN = pt3d_green.shape[0]
    N_RED = pt3d_red.shape[0]
    MTX_DIM = N_GREEN + N_RED + 1
    pts = np.concatenate([pt3d_green, pt3d_red])
    
    mtx = np.zeros((MTX_DIM, MTX_DIM))

    # solve startng point
    dist_start = np.linalg.norm(pts[:,:-1], axis=1)# I use origin since a constant point would be the same
    mtx[0,1:] = dist_start.reshape(1,-1)

    # mtx[i, j] = || goal(i) - pos(j) || ^ 2
    goal_i = np.concatenate([np.concatenate([pt3d_green_goal[np.newaxis] for i in range(N_GREEN)]), np.concatenate([pt3d_red_goal[np.newaxis] for i in range(N_RED)])])  # i.e. goal(i)
    assert goal_i.shape[0] == pts.shape[0]
    dist_goal_point = np.linalg.norm(pts[:, np.newaxis] - goal_i[np.newaxis], axis=2)
    mtx[1:, 1:] = dist_goal_point

    return mtx, pts, goal_i


def generate_min_path(green, red):
    

    pt3d_green, goal_pos_green = green
    pt3d_red, goal_pos_red = red
    # ADD Z INFOMRATION
    pt3d_green = np.concatenate([pt3d_green[:,1].reshape((-1,1)),pt3d_green[:,0].reshape((-1,1)),np.ones((pt3d_green.shape[0],1)) * .3], axis=1)    
    pt3d_red = np.concatenate([pt3d_red[:,1].reshape((-1,1)),pt3d_red[:,0].reshape((-1,1)),np.ones((pt3d_red.shape[0],1)) * .3], axis=1)
    print(pt3d_green)
    print(pt3d_red)

    mtx_tsp, pts, goal_i = generate_map_for_tsp((pt3d_green, goal_pos_green), (pt3d_red, goal_pos_red))
    permutation, distance = solve_tsp_dynamic_programming(mtx_tsp)
    permutation = [elem - 1 for elem in permutation[1:]]
    
    return permutation, distance, pts[permutation], goal_i[permutation]




if __name__ == "__main__":
    
    rospy.init_node("mymain")
    rospy.loginfo("NODE STARTED!!")

    # goals_definition:
    goal_pos_green = np.array([0.3, 0.3,0.3])
    goal_pos_red = np.array([0.3, -0.3,0.3]) 


    img = extract_camera_img()
    
    pt3d_green, pt3d_red = from_image_to_position(img, verbose=False, path_save=PATH_SAVE)
    permutation, distance, pts_in_order, goal_in_order = generate_min_path((pt3d_green, goal_pos_green), (pt3d_red, goal_pos_red))
    pts_in_order_list = pts_in_order.tolist()
    goal_in_order[:,-1] = .5
    goal_in_order_list = goal_in_order.tolist()
    print(pts_in_order_list, goal_in_order_list)
    robot = MyRobotPanda()
    robot.grab_multiple_objects_and_moveout(pts_in_order_list, goal_in_order_list)
