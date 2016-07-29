# Build

We develop in `python3` and we like to use `virtualenv`:


* Creating a virtual environment based on python3: this you only need to do the first time.
```sh
virtualenv -p python3 ~/workspace/envs/lola
```

* Sourcing it: this you need to do whenever you want to run (or develop) the code.
```sh
source ~/workspace/envs/lola/bin/activate
```


* In case you use `PyCharm` you will need to configure it to use your environment:        
```sh
# navigate to
PyCharm/Preferences/Project/Project Interpreter
# point your interpreter to
~/workspace/envs/lola/bin/python
```

* Requirements
```sh
# yolk3k is not a requirement, but it is helpful to list packages in our virtual environment
pip install yolk3k
pip install numpy
pip install scipy
pip install cython
pip install scikit-learn
```
        
        
* Build
```sh
# on OSX set also CC=clang
CFLAGS="-std=c++11" python setup.py develop
```


* Unit tests
```sh
cd test
python -m unittest *.py
```
        