* `git clone` and build the code

* Download some [data](https://www.adrive.com/public/YVKseV/wa.tgz)

* Uncompress it with `tar -xzvf wa.tgz`


Throught assume that my repository is under `~/workspace/github/lola/`.

# IBM2

```bash
python -m lola.aligner  ~/workspace/github/lola/test/data/ibm2.ini experiments/ibm2 -f wa/en-fr/training.en-fr.fr -e wa/en-fr/training.en-fr.en --test-f wa/en-fr/test.en-fr.fr --test-e wa/en-fr/test.en-fr.en --merge --naacl --skip-null --viterbi --save-entropy --save-parameters -v
```

We provide some required arguments:
* a configuration file `~/workspace/github/lola/test/data/ibm2.ini`
* a workspace `experiments/ibm2` where output files are saved
* training data (`-e` is the side we condition on, `-f` is the side we generate)

And also some optional arguments:
* test data (`--test-e` is the side we condition on, `--test-f` is the side we generate)
* `--merge` concatenates the training and the test data thus avoiding having to deal with unseen words at test time
 note that word alignment is an unsupervised task, thus we don't typically have training/test split unless we have manual data to evaluate on
* `--naacl` prints alignments in NAACL format (useful for comparison against manual alignments)
* `--skip-null` omit NULL alignments from output files
* `--viterbi` saves the Viterbi alignments for training/test data
* `--save-entropy` saves a log of entropy after each iteration
* `--save-parameters` saves estimated parameters
* `-v` prints logging information on screen





        
