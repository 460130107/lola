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


    # current interface
    time python -m lola.aligner debug/5k -f training/5k.f -e training/5k.e -v --ibm1 3 --ibm2 2 --test-f training/fr-en/testing/test/test.f --test-e training/fr-en/testing/test/test.e --merge --viterbi --save-parameters --naacl --posterior --save-entropy --skip-null

* the first value is the output directory
* then we have chosen a training set
* a few iterations of ibm1 and ibm2
* a test set
* we decided to merge training and test to avoid OOVs
* we decided to save the viterbi alignments
* to save the parameters of the model (after EM)
* and to save alignments also in NAACL format
* in NAACL format we can also have the posterior probability of each alignment
* we are saving the entropy of EM iterations
* and we are not printing alignments to NULL (the model does align things to NULL, we just do not print it)

For an example of feature extraction check

    python -m lola.extractor


# Config files

The configuration file has 3 blocks.

## Extractors

The block `[extractors]` can be used to configure feature extractor objects.
You should start by naming the extractor (e.g. words), then separate the name from the configuration with a colon ':'.
Then, choose the extractor type (e.g. WholeWordFeatures, this is a class in lola.ff).
After that, provide key-value pairs for the configuration of the extractor (each extractor knows how to configure itself give a string of key-value pairs). 
These key-value pairs are evaluated literally using python's eval function.
For now, we do not support spaces in the value, thus be careful with that, for example if you need a list with two numbers use [2,3] as opposed to [2, 3].
A few examples of extractors:


    [extractors]
    words: type=WholeWordFeatures extract_e=True extract_f=True extract_ef=True
    affixes: type=AffixFeatures extract_e=False extract_f=False extract_ef=True suffix_sizes=[2,3] prefix_sizes=[]
    categories: type=CategoryFeatures extract_e=True extract_f=True extract_ef=True 


You can repeat extractors of a given type, as long as you name them uniquely, example:


    [extractors]    
    suffixes: type=AffixFeatures extract_e=False extract_f=False extract_ef=True suffix_sizes=[2,3] prefix_sizes=[]
    prefixes: type=AffixFeatures extract_e=True extract_f=False extract_ef=False suffix_sizes=[] prefix_sizes=[2]


## Components

The second block `[components]` is used to configure locally normalised components of several types.
Again, you should start by naming a component (e.g. lexical), then separate the name from the configuration with a colon ':'.
Then, choose the component's type (e.g. BrownLexical).
Then, choose the first model where this component is added to (starting with 1) using the key `model=`.
For example, a lexical component is typically added to the first model, which is optimised for a number of iterations of EM.
Then, a distortion component is typically added to the second model, which is further optimised for a number of iterations of EM.
This allows us to optimise easier models first and hopefully avoid local optima.

Finally, provide key-value pairs for the configuration of the component.

Examples with multinomial/categorical components:


    [components]
    lexical: type=BrownLexical model=1
    jump: type=VogelJump model=2


Examples with log-linear components:


    [components]
    lexical: type=BrownLexical model=1
    llLexical: type=LogLinearLexical model=2 init='uniform' sgd-steps=5 sgd-attempts=10 extractors=['categories','words']
    jump: type=VogelJump model=3


Note that the log-linear component uses two of the feature extractors declared in the first block.


## Iterations

Finally, the block `[iteration]` is used to specify the number of iterations of EM for each model in order.

Examples:


    [iterations]
    3
    5
    10

This means we start with the components in model 1 and run 3 iterations of EM.
Then, then we add components in model 2 and run 5 iterations of EM.
Finally, we add components in model 3 and run another 10 iterations of EM.

# Using config files

        
        python -m lola.aligner training/stdmodel.ini debug/std -f training/example.f -e training/example.e --save-entropy --save-parameters -v
        python -m lola.aligner training/llmodel.ini debug/LL -f training/example.f -e training/example.e --save-entropy --save-parameters -v
