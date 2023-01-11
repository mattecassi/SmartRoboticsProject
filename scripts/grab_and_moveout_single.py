import numpy as np
import panda_robot
import rospy
from copy import deepcopy

TABLE_WIDTH = 0.8
HEIGHT_GRABBING = 0.4
HEIGHT_BEFORE_GRABBING = 0.5

HEIGH_MOVE_OUT = 0.50
DIST_GRIPPER_CLOSE = 0.03

MAX_EXTENSION = .70


class MyRobotPanda():

    """
    !!! ROSPY NODE NEED TO BE ACTIVE !!!
    """

    def __init__(self):
 
        rospy.loginfo("Init panda arm and gripper...")
        self._arm = panda_robot.PandaArm()
        self._gripper = self._arm.get_gripper() 
 
        rospy.loginfo("Move to neutral position...")
        self._arm.move_to_neutral()
        # self._gripper.open()
        self._arm.exec_gripper_cmd(.08)
        rospy.loginfo("Extract informations about the robot...")
        _, ori = self._arm.ee_pose()
        self._ori = ori
    
    def move_arm_to_cartesian_pos(self, final_pos):
        res, joint_final_pos =  self._arm.inverse_kinematics(final_pos, self._ori)
        
        if ((joint_final_pos - self._arm.angles()) > np.pi).any():
            # compute pos in the middle
            actual_pos, _ = self._arm.ee_pose()
            rospy.logwarn("Something went wrong... middle_movement_required")
            shift = (np.array(final_pos) - actual_pos)/2
            middle_pos = np.round_((actual_pos + shift), decimals=2).tolist()
            rospy.loginfo(f"Going to middle position {middle_pos}")
            self.move_arm_to_cartesian_pos(middle_pos)

        res, joint_final_pos =  self._arm.inverse_kinematics(final_pos, self._ori)
        if not res:     
            rospy.logerr(f"ERROR in computing final position inverse kinematics  {res,final_pos}")
            return False
        rospy.loginfo("Move arm to position...")
        self._arm.move_to_joint_position(joint_final_pos, use_moveit=False)
        return True


    def close_gripper(self, dist=DIST_GRIPPER_CLOSE):
        self._arm.exec_gripper_cmd(DIST_GRIPPER_CLOSE)

    

    def _compute_position_over_obj(self, obj_pos):
        return [
            obj_pos[0],
            obj_pos[1],
            HEIGHT_BEFORE_GRABBING
        ]
    
    def _grab_object(self):
        """
        !!! Gripper MUST BE OPEN !!!
        """
        pos_before_grabbing, _ = self._arm.ee_pose()
        grab_pos =  [
            pos_before_grabbing[0],
            pos_before_grabbing[1],
            HEIGHT_GRABBING
        ]
        rospy.loginfo("GO DOWN")
        rospy.logdebug(f"grab_pos {grab_pos}")
        res = self.move_arm_to_cartesian_pos(grab_pos)
        if not res:
            return False

        rospy.loginfo("Grabbing...")
        self.close_gripper()

        rospy.loginfo("Go up!!")
        rospy.logdebug(f"after grab {pos_before_grabbing}")
        res = self.move_arm_to_cartesian_pos(pos_before_grabbing)
        if not res:
            return False
        return True

    def grab_single_object_and_moveout(self, object_pos, move_out_pos):
        
        final_pos = self._compute_position_over_obj(object_pos)
        rospy.loginfo(f"over position computed {final_pos}")

        res = self.move_arm_to_cartesian_pos(final_pos)
        if not res:
            return False

        rospy.loginfo("Grab object") 
        res = self._grab_object()
        if not res:
            rospy.logerr("Error while grasping")
            return False


        rospy.loginfo(f"Move to MOVEOUT position {move_out_pos}...")
        res = self.move_arm_to_cartesian_pos(move_out_pos)
        if not res:
            rospy.logerr("Error while moving out")

            return False
        rospy.loginfo("Opening gripper...")
        self._arm.exec_gripper_cmd(.08)

    def grab_multiple_objects_and_moveout(self, object_list, goal_list):
        for (object_pos, goal_list) in zip(object_list, goal_list):
            rospy.loginfo(f"Moving out {object_pos}")
            self.grab_single_object_and_moveout(object_pos, goal_list)


if __name__ == "__main__":

    # init....
    object_poses = [
        [0.5, -0.2, .3],
        [0.605, 0.2,.3],
        [0.7, 0., .3],
    ]

    goal_poses = [
        [.3, .3 ,.5],
        [.3, -.3 ,.5],
        [.3, .3 ,.5],

    ]
    
    
    rospy.init_node("gamos")
    robot = MyRobotPanda()
    robot.grab_multiple_objects_and_moveout(object_poses, goal_poses)

    rospy.loginfo("End. I hope is the end we waited for...")



    
    

