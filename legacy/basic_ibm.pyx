from legacy.model cimport DefaultModel
from legacy.component cimport LexicalParameters, UniformAlignment, JumpParameters
from legacy.component cimport BrownDistortionParameters


cdef class IBM1(DefaultModel):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, LexicalParameters lex_parameters):
        super(IBM1, self).__init__([lex_parameters, UniformAlignment()])

    cpdef LexicalParameters lexical_parameters(self):
        return <LexicalParameters>self.component(0)


cdef class BrownIBM2(DefaultModel):
    """
    An IBM2 is a 0th-order model with lexical parameters and the original distortion parameters.
    """

    def __init__(self, LexicalParameters lex_parameters, BrownDistortionParameters dist_parameters):
        super(BrownIBM2, self).__init__([lex_parameters, dist_parameters])

    cpdef initialise(self, dict initialiser):
        if 'IBM1' in initialiser:
            # we are replacing our own lexical parameters, by those of an IBM1 which has already been optimised
            self._components[0] = initialiser['IBM1'].lexical_parameters()

    cpdef LexicalParameters lexical_parameters(self):
        return <LexicalParameters>self.component(0)

    cpdef BrownDistortionParameters distortion_parameters(self):
        return <BrownDistortionParameters>self.component(1)


cdef class VogelIBM2(DefaultModel):
    """
    An IBM2 is a 0th-order model with lexical parameters and Vogel's distortion parameters.
    """

    def __init__(self, LexicalParameters lex_parameters, JumpParameters dist_parameters):
        super(VogelIBM2, self).__init__([lex_parameters, dist_parameters])

    cpdef initialise(self, dict initialiser):
        if 'IBM1' in initialiser:
            # we are replacing our own lexical parameters, by those of an IBM1 which has already been optimised
            self._components[0] = initialiser['IBM1'].lexical_parameters()

    cpdef LexicalParameters lexical_parameters(self):
        return <LexicalParameters>self.component(0)

    cpdef JumpParameters distortion_parameters(self):
        return <JumpParameters>self.component(1)