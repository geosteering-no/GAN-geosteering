from setuptools import setup

setup(
   name='GanLOG',
   version='1.0',
   install_requires=["numpy","torch","tensorflow", "torchvision", "scikit-learn==0.22","scikit-image","seaborn"],
   py_modules=["GanLog"]
)
