RUNTEST=python -m unittest discover -v

.PHONY: test
test:
	${RUNTEST} test

.PHONY: black
black:
	black -l 79 --preview .

.PHONY: mypy
mypy:
	mypy lm_service

.PHONY: isort
isort:
	isort . --line-length=79


.PHONY: autoflake
autoflake:
	autoflake --remove-unused-variables --verbose --in-place  ./lm_service/**/*py

all: autoflake black isort mypy

#.PHONY: pylint
#pylint:
#	pylint lm_service


