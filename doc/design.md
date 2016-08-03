
Everything starts with [lola.model.GenerativeModel](../lola/model.pxd): 

* a generative model is made of a number of (locally normalised) generative components
* a [DefaultModel](../lola/model.pyx) simply maintains a vector of generative components that are combined multiplicatively

A [lola.component.GenerativeComponent](../lola/component.pxd):

* is defined over an event space
* is used to encapsulate the parametric form of a component
* offers methods for the computation of the E-step 
* offers methods for the M-step
 
[lola.event.EventSpace](../lola/event.pxd):

* an event is described by a context and a decision (these are 0-based integers)
* an event space simply converts the general information available (English sentence, French sentence, alignment assignment for a certain position being generated) into an event
 

Example: [BrownLexical](../lola/component.pyx) component is responsible for filling in a French word position conditioned on an English word in IBM model 1.
This component describes the probabilitiy `p(f|e)` where `f` is a word in the French vocabulary and `e` is word in the English vocabulary.
In IBM model 1, this is modelled by a [categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) per conditioning context (an English word).
Each distribution is defined over the entire French vocabulary, thus if `v_F` is the size of the French vocabuary and `v_E` is the size of the English vocabulary,
then we have as many as `v_F * v_E` parameters to be estimated.
[LexEventSpace](../lola/event.pyx) produces something of the kind `context=i` and `decision=j` from an alignment `e_snt, f_snt, i, j`.
[BrownLexical](../lola/component.pyx) then encapsulates the categorical distributions.


In [lola.config](../lola/config.py) we create model components and the specification of a generative model.
Model components that are feature-rich, e.g. [log-linear component](../lola/llcomp.pyx) rely of feature representation of events.
 These feature representations are produced by [feature extractors](../lola/ff.pxd). 
 At configuration time we also construct and load feature extractors.
 
EM is implemented in [lola.hmm0](../lola/hmm0.pyx). There you will see the main body of computation of the E-step (which assumes a 0th order model). 
 Computing the necessary quantities (e.g. likelihood and posterior) is delegated to each generative component.
 The same applies to the M-step, some components are categorical and require nothing more than renormalisation of expected counts, others are log-linear and require gradient-based optimisation.
 
In [lola.aligner](../lola/aligner.py) we have the `pipeline`

1. load extractors, components and model specifications
2. train each model in order
3. predictions

 