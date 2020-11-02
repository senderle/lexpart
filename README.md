# lexpart

Companion code for "Toward a Thermodynamics of Meaning," 
CHR 2020 (https://arxiv.org/abs/2009.11963)

This contains a simple reference implementation of a lingusitic partition
function as described in the paper, with some limited documentation. 
The repository is pip-installable:

    pip install git+https://github.com/senderle/lexpart#egg=lexpart

To train an embedding based on the included test dataset (enwiki8), run the
following commands:

    python -m lexpart vocab - testvocab
    python -m lexpart corpus testvocab.npz - testcorpus
    python -m lexpart embed testcorpus.npz testembed.tgz
    python -m lexpart wordsim testembed.tgz paris

This will print out a list of words in the corpus that are similar to "paris."

To train an embedding based on your own corpus, replace the "-" in the above
commands with the path to a folder containing plain text files.
