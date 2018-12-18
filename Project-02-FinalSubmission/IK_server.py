#!/usr/bin/env python

import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print("No valid poses received")
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []

        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()
            # Define DH param symbols
            d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
            a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
            # Joint angle symbols
            q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
            alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
            # Define Modified DH Transformation matrix
            s = {alpha0:   0.0,   a0:    0.0,    d1:  0.75, 
                 alpha1: -pi/2,   a1:   0.35,    d2:   0.0,    q2:q2-pi/2,
                 alpha2:   0.0,   a2:   1.25,    d3:   0.0,    
                 alpha3: -pi/2,   a3: -0.054,    d4:   1.5,
                 alpha4:  pi/2,   a4:    0.0,    d5:   0.0,
                 alpha5: -pi/2,   a5:    0.0,    d6:   0.0,
                 alpha6:   0.0,   a6:    0.0,    d7: 0.303,   q7:0.0}
        
        
            # Create individual transformation matrices
            T0_1 = Matrix([[            cos(q1),           -sin(q1),           0,             a0],
                           [sin(q1)*cos(alpha0),cos(q1)*cos(alpha0),-sin(alpha0),-sin(alpha0)*d1],
                           [sin(q1)*sin(alpha0),cos(q1)*sin(alpha0), cos(alpha0), cos(alpha0)*d1],
                           [                  0,                  0,           0,              1]])
            T0_1 = T0_1.subs(s)
        
            T1_2 = Matrix([[            cos(q2),           -sin(q2),           0,             a1],
                           [sin(q2)*cos(alpha1),cos(q2)*cos(alpha1),-sin(alpha1),-sin(alpha1)*d2],
                           [sin(q2)*sin(alpha1),cos(q2)*sin(alpha1), cos(alpha1), cos(alpha1)*d2],
                           [                  0,                  0,           0,              1]])
            T1_2 = T1_2.subs(s)
            
            T2_3 = Matrix([[            cos(q3),             -sin(q3),             0,               a2],
                           [sin(q3)*cos(alpha2),  cos(q3)*cos(alpha2),  -sin(alpha2),  -sin(alpha2)*d3],
                           [sin(q3)*sin(alpha2),  cos(q3)*sin(alpha2),   cos(alpha2),   cos(alpha2)*d3],
                           [                  0,                    0,             0,                1]])
            T2_3 = T2_3.subs(s)
        
            T0_3 = T0_1 * T1_2 * T2_3
        
            #Rotation Maxtrix of X,Y,Z axis
            R_y = Matrix([[ cos(-pi/2),        0, sin(-pi/2)],
                          [          0,        1,          0],
                          [-sin(-pi/2),        0, cos(-pi/2)]])

            R_z = Matrix([[ cos(pi),     -sin(pi),          0],
                          [ sin(pi),      cos(pi),          0],
                          [       0,            0,          1]])
            
            R_corr = R_z*R_y
    
            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z
    
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
            [req.poses[x].orientation.x, req.poses[x].orientation.y,
             req.poses[x].orientation.z, req.poses[x].orientation.w])

            # Calculate joint angles using Geometric IK method
            R_roll = Matrix([[ 1,           0,          0],
                             [ 0,   cos(roll), -sin(roll)],
                             [ 0,   sin(roll),  cos(roll)]])

            R_pitch = Matrix([[ cos(pitch), 0,  sin(pitch)],
                              [       0,    1,           0],
                              [-sin(pitch), 0,  cos(pitch)]])
 
            R_yaw = Matrix([[ cos(yaw), -sin(yaw),       0],
                            [ sin(yaw),  cos(yaw),       0],
                            [ 0,              0,        1]])
    
     
            R0_6 = R_yaw*R_pitch*R_roll*R_corr
    
            # Calculate wc point
            wx = px - 0.303 * R0_6[0,2]
            wy = py - 0.303 * R0_6[1,2]
            wz = pz - 0.303 * R0_6[2,2]
            a1=0.35
            a2=1.25
            a3=-0.054
            d1=0.75
            d4=1.50

            # theta1
            theta1 = atan2(wy, wx)
            
            #distance 
            dis_2_3=a2
            dis_2_4=sqrt((sqrt(wx**2+wy**2)-a1)**2+(wz-d1)**2)
            dis_3_4=sqrt(d4**2+a3**2)

            # theta2
            cos_theta2_1=(dis_2_4**2+dis_2_3**2-dis_3_4**2)/(2*dis_2_4*dis_2_3)
            sin_theta2_1=sqrt(1-cos_theta2_1**2)
            theta2_1 = atan2(sin_theta2_1,cos_theta2_1)
            sin_theta2_2=(wz-d1)/(dis_2_4)
            cos_theta2_2=sqrt(1-sin_theta2_2**2)
            theta2_2 = atan2(sin_theta2_2,cos_theta2_2)
            theta2 = pi/2-theta2_2-theta2_1 

            #theta3
            theta3_1=atan2(a3,d4)
            cos_theta3_2 = (dis_2_3**2+dis_3_4**2-dis_2_4**2)/(2*dis_2_3*dis_3_4)
            sin_theta3_2 = sqrt(1-cos_theta3_2**2)
            theta3_2 = atan2(sin_theta3_2,cos_theta3_2)
            theta3 = pi/2 - theta3_2-theta3_1
    
            # Calculate R0_3  
            R0_3 = T0_3.evalf(subs={q1:theta1, q2:theta2, q3:theta3})[0:3, 0:3]
            
            # Calculate new_R3_6
            new_R3_6 = R0_3.transpose() * R0_6 
        
            # theta4,theta5,theta6
            theta4=atan2(new_R3_6[2,2],-new_R3_6[0,2])
            sin_theta5=sqrt(new_R3_6[2,2]**2+new_R3_6[0,2]**2)
            theta5=atan2(sin_theta5, new_R3_6[1,2])
            theta6=atan2(-new_R3_6[1,1],new_R3_6[1,0])

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)
        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print ("Ready to receive an IK request")
    rospy.spin()

if __name__ == "__main__":
    IK_server()


