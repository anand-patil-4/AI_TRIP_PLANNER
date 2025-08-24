from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    requirement_list:List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!="-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("'requirements.txt' file not found")

    return requirement_list

print(get_requirements())

setup(
    name="AI-TRAVEL-PLANNER",
    version="0.0.1",
    author="Anand Kumar Patil",
    author_email="patilanand9562@gmail.com",
    packages = find_packages(),  #find all the local packages(i.e. the folders which contains "__init__.py" file.)
    install_requires=get_requirements()
)
