
This project is about feature-rich unsupervised word alignment models.
At this point we are working on 0th order HMMs, that is, IBM2-like models.
We follow Berg-Kirkpatrick et al (2010) and reparameterise IBM2's categorical distributions using exponentiated linear functions (a log-linear parameterisation).

This code-base started as [Guido Linder's final project](https://esc.fnwi.uva.nl/thesis/centraal/files/f1886233032.pdf) towards his BSc.

# Build

Check our [build](doc/build.md) instructions.
        

# Usage

For a help message try:
 
```sh
$ python -m lola.aligner --help
```

Check some [examples](doc/example.md).
 
You can train different models with `lola`, check our [configuration file format](doc/config.md). 

