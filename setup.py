from setuptools import find_packages, setup

if __name__ == "__main__":
    # Still necessary, otherwise we get a pip error
    setup(
        name='Text2Head',
        packages=find_packages(),
        version='0.1.0',
        description='Text2Head Project @ niessnerlab',
        author='Simon Langrieger & Katharina Schmid',
        license='MIT',
    )