# Sphinx instructions

```shell
sphinx-quickstart docs
sphinx-apidoc -o docs/source <package_name>
sphinx-build -M html docs/source/ docs/build/

```