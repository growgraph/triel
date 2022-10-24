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

.PHONY: prettyyaml
prettyyaml:
	find . -name "*yaml" -and -not -ipath './.*' -type f | xargs pretty-format-yaml --autofix --indent 4

.PHONY: prettyjson
prettyjson:
	find . -name "*json" -and -not -ipath './.*' -type f | xargs pretty-format-json --autofix --indent 4

.PHONY: prettytoml
prettytoml:
	pretty-format-toml --autofix ./*toml
	toml-sort -ia ./*.toml


all: autoflake black isort mypy prettyjson prettytoml prettyyaml

#.PHONY: pylint
#pylint:
#	pylint lm_service


