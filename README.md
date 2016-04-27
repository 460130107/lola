# word-alignment-models
BSc project on feature-rich unsupervised word alignment models

# Build

We develop in `python3` and we like to use `virtualenv`:


* Creating a virtual environment based on python3


        virtualenv -p python3 ~/workspace/envs/lola

* Sourcing it


        source ~/workspace/envs/lola/bin/activate


* Configuring `PyCharm`        

        # navigate to
        PyCharm/Preferences/Project/Project Interpreter
        # point your interpreter to
        ~/workspace/envs/lola/bin/python

* Requirements


        # yolk3k is not a requirement, but it is helpful to list packages in our virtual environment
        pip install yolk3k
        # numpy is a requirement
        pip install numpy
        # and also cython
        pip install cython
        
* Build

        python setup.py develop


* Unit tests

        cd test
        python -m unittest *
        