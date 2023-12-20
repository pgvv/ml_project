from setuptools import find_packages, setup                 #finds all the packages and its setup files
from typing import List

hyphen_e_dot= '-e .'

def get_requirements(file_path:str)-> List[str]:
    # This function will return the list of requirements

    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]    #replaces "\n" from readlines

        #If trigger already present in text file, need not to trigger again
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    return requirements

setup(                                                          #metadeta of the project                                                            
    name= 'End to End Data Science Project Krish Naik',
    version= '0.0.1',
    author= 'Pagadala',
    author_email= 'g.pagadala06@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')      #installs required libraries for the project
)