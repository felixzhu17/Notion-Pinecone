git add -A
git commit -m "update"
git push
bumpversion patch
del /Q dist\*
rmdir /Q /S dist\
mkdir dist
python setup.py sdist bdist_wheel
twine upload dist/*