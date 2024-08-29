from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'ferl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'data'), glob('data/objects/*.xml')),
        (os.path.join('share', package_name, 'data', 'demonstrations', 'demos'), glob('data/demonstrations/demos/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ingui',
    maintainer_email='ingui2@illinois.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feature_elicitator = ferl.feature_elicitator:main',
            'test_vel = ferl.test_vel:main',
            'demo_recorder = ferl.demo_recorder:main',
            'user_input_node = ferl.user_input_node:main',
            'xr_ferl = ferl.xr_ferl:main',
        ],
    },
)
