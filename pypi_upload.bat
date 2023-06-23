bumpversion patch
del /Q dist\*
rmdir /Q /S dist\
mkdir dist
python setup.py sdist bdist_wheel
twine upload dist/*