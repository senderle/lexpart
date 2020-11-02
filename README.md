# lexpart

Companion code for "Toward a Thermodynamics of Meaning," 
CHR 2020 (https://arxiv.org/abs/2009.11963)

This contains a simple reference implementation of a lingusitic partition
function as described in the paper, with some limited documentation.

### Installation

The repository is pip-installable:

    pip install git+https://github.com/senderle/lexpart#egg=lexpart

### Usage Example

To train an embedding based on the included test dataset (enwiki8), run the
following commands:

    python -m lexpart vocab vocab.npz -
    python -m lexpart corpus corpus.npz vocab.npz -
    python -m lexpart embed embed.npz corpus.npz
    python -m lexpart wordsim embed.npz paris

This will print out a list of words in the corpus that are similar to "paris."

To train an embedding based on your own corpus, replace the `-` in the above
commands with the path to a folder containing plain text files.

### Mathematical Fine Print

The model described in the paper is based on the grand canonical partition
function for multiple species in its standard form: 

Z = ∑<sub>i</sub> e<sup>β(µ[1]N[1,i] + µ[2]N[2,i] + ... + µ[k]N[k,i] − Ei)</sup>

For computational purposes, however, it's convenient to represent the
partition function in another form. Substituting u[k] for e<sup>βμ[k]</sup>, 
we can rewrite the above like so:

Z = ∑<sub>i</sub> u[1]<sup>N[1,i]</sup> u[2]<sup>N[2,i] ... u[k]<sup>N[k,i] e<sup>−βE[i]</sup>

If we cheat a bit by treating the energy term as a constant for all terms, 
we can treat the partition function as one huge polynomial. Each term in
the polynomial represents a sentence as a bag of words, where the exponent
is the word count. Since counts for sentences are sparse, and differentiation 
is a linear operator, we can calculate values for the jacobian and hessian 
very efficiently. The code that performs this calculation is in
`sparsehess.py`.
