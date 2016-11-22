
* better integration between EventSpace, GenerativeComponent and FeatureMatrix
** FeatureMatrix could be aware of the EventSpace
*** EventSpace should know how to convert from event ids to surface events (e.g. ids to words, itds to jumps)
* prune event spaces by frequency (min-count and max-count)
* set an optional vocabulary size for EventSpace (rank events by frequency and keep top N)


Proposal:


        # declaring event spaces?
        [spaces]
        # max-contexts and max-decisions rank the space of contexts and decisions and keep only the most frequent ones
        lexical: type=LexEventSpace max-contexts=200 max-decisions=200
        # threshold pruning: min-context-count and min-decision-count prune by absolute frequency
        # this type of pruning applies before capping the sets 
        lexical2: type=LexEventSpace min-context-count=10 max-decisions=200
        jump: type=JumpEventSpace max-decisions=20
        
        [extractors]
        # a certain type of extractor can only support a certain type of EventSpace, and that's not configurable
        word_pairs: type=WholeWordFeatureExtractor 
        
        [components]
        # a certain component is defined over one single EventSpace, the one that is common to all extractors
        # is this a good design?!
        llLexical: type=LogLinearLexical init='uniform' event='lexical' extractors=['word_pairs']
        