RUNTEST=python -m unittest discover -v

.PHONY: test
test:
	${RUNTEST} test

.PHONY: black
black:
	black -l 79 .

.PHONY: mypy
mypy:
	mypy lm_service

.PHONY: isort
isort:
	isort . --line-length=79


.PHONY: autoflake
autoflake:
	autoflake --remove-unused-variables --remove-all-unused-imports --verbose --in-place  ./lm_service/**/*py


#.PHONY: pylint
#pylint:
#	pylint lm_service


