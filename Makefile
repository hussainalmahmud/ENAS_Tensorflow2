install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# format:
# 	black *.py

# lint:
# 	pylint --disable=R,C src/cifar10/*.py 

# test:
# 	python -m pytest general_controller.py

# all: install lint test format
all: install
