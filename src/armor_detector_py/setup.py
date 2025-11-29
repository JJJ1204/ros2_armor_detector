import os
from setuptools import setup

package_name = 'armor_detector_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 我们手动把模型文件最好也申明一下，或者直接在代码里用绝对路径（更简单）
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jjj',
    maintainer_email='jjj@todo.todo',
    description='Armor Detector Python Node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = armor_detector_py.detector_node:main',
        ],
    },
)