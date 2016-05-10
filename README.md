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

        # on mac set CC=clang
        CFLAGS="-std=c++11" python setup.py develop


* Unit tests

        cd test
        python -m unittest *
        

# Examples

You can try:

    # here we train a model using a training corpus and use it to obtain Viterbi alignments
    time python -m lola.aligner -f training/example.f -e training/example.e -v --ibm1 5 --ibm2 5  --viterbi training/training --save-entropy training/training.entropy

    # here we train a model using a training corpus and use it to obtain Viterbi alignments to both training and test sets
    time python -m lola.aligner -f training/example.f -e training/example.e -v --ibm1 5 --ibm2 5 --test-f training/example.test.f --test-e training/example.test.e  --viterbi training/example --save-entropy training/example.entropy

    # here we train a model using the concatenation of training and test sets (this is basically to avoid OOVs) and use it to obtain Viterbi alignments to both training and test sets
    # note that it is okay to concatenate the sets because the task remains unsupervised and in normal uses there is no such a distinction between training and test sets, this distinction only makes sense when we are interested in computing AER for the test set based on some manual alignments
    time python -m lola.aligner -f training/example.f -e training/example.e -v --ibm1 5 --ibm2 5 --test-f training/example.test.f --test-e training/example.test.e --merge  --viterbi training/merged --save-entropy training/merged.entropy

For an example of feature extraction check

    python -m lola.extractor
