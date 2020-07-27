#!/usr/bin/env python
import glob
import os
import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np


def callback(data):
    npdata = np.array(data.data)
    files = glob.glob('corridor*.npy')
    np.save('corridor%d.npy' % (len(files)), npdata)
    print('corridor saved to ', os.getcwd())


def listener():
    rospy.init_node("corr_listener")
    rospy.Subscriber("/field_server/corridors", Float64MultiArray, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
