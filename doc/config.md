

Lola's configuration file has 3 blocks.

# Extractors

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


# Components

The second block `[components]` is used to configure locally normalised components of several types.
Again, you should start by naming a component (e.g. lexical), then separate the name from the configuration with a colon ':'.
Then, choose the component's type (e.g. BrownLexical).
Finally, provide key-value pairs for the configuration of the component.

Examples of multinomial/categorical components:


    [components]
    lexical: type=BrownLexical
    udist: type=UniformAlignment
    jump: type=VogelJump


Examples of log-linear components:


    [components]
    llLexical: type=LogLinearLexical init='uniform' sgd-steps=5 sgd-attempts=10 extractors=['categories','words']


Note that the log-linear component uses two of the feature extractors declared in the first block.
Also, log-linear components are optimised by SGD, thus we note some SGD options being configured.


# Models

Finally, the block `[models]` is used to specify full models (made of sets of components).
First, name your model (e.g. ibm1). Then, to define this model, specify a list of unique components with they mandatory keyword 'components='.
Also, to decide for how long we should optimise this model with EM, specify the mandatory keyword 'iterations='.

Examples:


    [models]

    ibm1: iterations=5 components=['lexical','udist']
    ibm2: iterations=5 components=['lexical','jump']


Note that we can have multiple models. They are optimised in the given order (in this case `ibm1` first, then `ibm2`).
They may reuse certain components, in which case higher models benefit from previously optimised components from lower models.
In the example, `ibm2` reuses the lexical component previously trained with `ibm1`, but replaces IBM1's uniform distortion by jump-based distortion.
