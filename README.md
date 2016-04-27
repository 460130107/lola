# word-alignment-models
BSc project on feature-rich unsupervised word alignment models

# Build

We develop in `python3` and we like to use `virtualenv`:


* Creating a virtual environment based on python3: this you only need to do the first time.


        virtualenv -p python3 ~/workspace/envs/lola

* Sourcing it: this you need to do whenever you want to run (or develop) the code.


        source ~/workspace/envs/lola/bin/activate


* In case you use `PyCharm` you will need to configure it to use your environment:        

        # navigate to
        PyCharm/Preferences/Project/Project Interpreter
        # point your interpreter to
        ~/workspace/envs/lola/bin/python

* Requirements


        # yolk3k is not a requirement, but it is helpful to list packages in our virtual environment
        pip install yolk3k
        # numpy is a requirement
        pip install numpy
        # Cython is not yet a requirement, but might become 
        pip install cython
        
* Build

        python setup.py develop


* Unit tests

        cd test
        python -m unittest *
        

# Examples

You can try:


    python -m lola.wibm1
