.PHONY: init docs test publish

init:
	virtualenv -p python3 .
	source bin/activate
	pip3 install -U pip setuptools
	pip3 install pytest Sphinx pre-commit black

docs:
	sphinx-apidoc -f -o docs rddlgym --ext-autodoc
	[ -e "docs/_build/html" ] && rm -R docs/_build/html
	sphinx-build docs docs/_build/html

test:
	pytest tests/*.py -sv --disable-warnings 2>/dev/null

publish:
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
	rm -Rf dist/
