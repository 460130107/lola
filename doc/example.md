* `git clone` and build the code

* Download some [data](https://www.adrive.com/public/YVKseV/wa.tgz)

* Uncompress it with `tar -xzvf wa.tgz`


Throughout assume that my repository is under `~/workspace/github/lola/`.

# IBM2


```sh
$ python -m lola.aligner  ~/workspace/github/lola/test/data/ibm2.ini experiments/ibm2 -f wa/en-fr/training.en-fr.fr -e wa/en-fr/training.en-fr.en --test-f wa/en-fr/test.en-fr.fr --test-e wa/en-fr/test.en-fr.en --merge --naacl --skip-null --viterbi --save-entropy --save-parameters -v
```

We provide some required arguments:
* a configuration file `~/workspace/github/lola/test/data/ibm2.ini`
* a workspace `experiments/ibm2` where output files are saved
* training data (`-e` is the side we condition on, `-f` is the side we generate)

And also some optional arguments:
* test data (`--test-e` is the side we condition on, `--test-f` is the side we generate)
* `--merge` concatenates the training and the test data thus avoiding having to deal with unseen words at test time
 note that word alignment is an unsupervised task, thus we don't typically have training/test split unless we have manual data to evaluate on
* `--naacl` prints alignments in NAACL format (useful for comparison against manual alignments) in addition to the usual Moses format
* `--skip-null` omit NULL alignments from output files
* `--viterbi` saves the Viterbi alignments for training/test data
* `--save-entropy` saves a log of entropy after each iteration
* `--save-parameters` saves estimated parameters
* `-v` prints logging information on screen

```sh
$ perl wa/wa_eval_align.pl wa/en-fr/test.en-fr.naacl experiments/ibm2/ibm2.test.viterbi.naacl
```


## Log-linear IBM2


```sh
$ python -m lola.aligner  ~/workspace/github/lola/test/data/llibm2.ini experiments/llibm2 -f wa/en-fr/training.en-fr.fr -e wa/en-fr/training.en-fr.en --test-f wa/en-fr/test.en-fr.fr --test-e wa/en-fr/test.en-fr.en --merge --naacl --skip-null --viterbi --save-entropy --save-parameters --min-f-count 10 --min-e-count 10 -v
```

The only differences are:
* the configuration file
* `--min-f-count 10` which maps every French word that happens less than 10 times to a `f-unk` token
* `--min-e-count 10` which maps every English word that happens less than 10 times fo a `e-unk` token

Mapping rare words to special tokens leads to much faster training and for now this is the only way to deal with large vocabularies.

```sh
$ perl wa/wa_eval_align.pl wa/en-fr/test.en-fr.naacl experiments/llibm2/ibm2.test.viterbi.naacl
```


## Output files

```sh
$ ls experiments/ibm2
```