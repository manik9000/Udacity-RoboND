#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from pcl_helper import *
from rospy_message_converter import message_converter
import yaml


# Global values
SCENE_NUMBER = 1
OUTPUT_FILENAME = "output_{}.yaml".format(str(SCENE_NUMBER))

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Returns the statistical outlier filtering
def statistical_outlier_filtering(data, k=20, thresh=0.5):
    sof = data.make_statistical_outlier_filter()
    sof.set_mean_k(k)
    sof.set_std_dev_mul_thresh(thresh)
    return sof.filter()

# Returns voxel grid downsampling
def voxel_grid_downsampling(data, leaf_size=0.01):
    vox = data.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return vox.filter()

# Returns passthrough filter
def passthrough_filter(data, axis, min, max):
    passthrough = data.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough_z.set_filter_limits(min, max)
    return passthrough.filter()

# Do the RANSAC plane segmentation
def __ransac_segmentation(data, max_distance=0.01):
    ransac = data.make_segmenter()
    ransac.set_model_type(pcl.SACMODEL_PLANE)
    ransac.set_method_type(pcl.SAC_RANSAC)
    ransac.set_distance_threshold(max_distance)
    return ransac.segment()

# Get objects and table from scene-
def get_scene_elements(data):
    inliners, coefficients = __ransac_segmentation(data)
    objects = data.extract(inliners, negative=True)
    table = data.extract(inliners, negative=False)
    return objects, table

# Do some euclidean clustering
def euclidean_clustering(white_cloud, tolerance=0.01, min=30, max=5000):
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min)
    ec.set_MaxClusterSize(max)
    ec.set_SearchMethod(tree)
    return ec.Extract()

# Get the cluster cloud of objects
def get_cluster_cloud(cluster_indices, white_cloud):
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

# Detect objects in the scene
def detect_objects(objects, cluster_indices, white_cloud):
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    return detected_objects, detected_objects_labels


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    # Do the filtering
    pcl_data = statistical_outlier_filtering(pcl_data)
    pcl_data = voxel_grid_downsampling(pcl_data)
    pcl_data = passthrough_filter(pcl_data, 'z', 0.6, 1.1)
    pcl_data = passthrough_filter(pcl_data, 'x', 0.3, 1.0) # Remove boxes

    # Get the objects and table for recognition
    objects, table = get_scene_elements(pcl_data)

    # Create the white cloud
    white_cloud = XYZRGB_to_XYZ(objects)

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_indices = euclidean_clustering(white_cloud)

    # Get the cluster cloud
    cluster_cloud = get_cluster_cloud(cluster_indices, white_cloud)

    # Convert PCL data to ROS messages and publish
    pcl_objects_pub.publish(pcl_to_ros(objects))
    pcl_table_pub.publish(pcl_to_ros(table))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))

    # Detect objects and publish them to ROS
    detected_objects, detected_labels = detect_objects(objects, cluster_indices, white_cloud)
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Assign test scene number and output for response
    test_scene_num = Int32()
    test_scene_num.data = SCENE_NUMBER
    output = []

    # Get object list and dropbox parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # Loop through all items
    for object in object_list:

        # Create response parameters
        object_name = String()
        arm = String()
        pick_pose = Pose()
        place_pose = Pose()

        # Get object name
        object_name.data = str(object.label)

        # Get the centroid of the object cloud
        cloud = ros_to_pcl(object.cloud).to_array()
        x, y, z = np.mean(cloud, axis=0)[:3]
        pick_pose.position.x = np.asscalar(x)
        pick_pose.position.y = np.asscalar(y)
        pick_pose.position.z = np.asscalar(z)

        # Get to what group the item belongs
        target_group = None
        for param in object_list_param:
            if param['name'] == object_name.data:
                target_group = param['group']
                break

        # Check for box information
        for param in dropbox_list_param:
            if param['group'] == target_group:
                arm.data = param['name']
                x, y, z = param['position']
                place_pose.position.x = np.float(x)
                place_pose.position.y = np.float(y)
                place_pose.position.z = np.float(z)
                break

        # Create yaml for object
        dict = make_yaml_dict(test_scene_num, arm, object_name, pick_pose, place_pose)

        # Add yaml to output
        output.append(dict)

        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Pass the data to the response
            resp = pick_place_routine(test_scene_num, object_name, arm, pick_pose, place_pose)
            print ("Response: ", resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # Create an output yaml file
    send_to_yaml(OUTPUT_FILENAME, output)


if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

