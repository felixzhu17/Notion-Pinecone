git add -A
git commit -m "update"
git push
bumpversion patch
rm -rf dist/*
mkdir dist
python setup.py sdist bdist_wheel
twine upload dist/*