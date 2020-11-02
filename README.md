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

Z = ∑<sub>i</sub> e<sup>β(µ<sub>1</sub>N<sub>1,i</sub> + µ<sub>2</sub>N<sub>2,i</sub> + ... + µ<sub>k</sub>N<sub>k,i</sub> − E<sub>i</sub>)</sup>

For computational purposes, however, it's convenient to represent the
partition function in another form. Substituting u<sub>k</sub> for e<sup>βμ<sub>k</sub></sup>, 
we can rewrite the above like so:

Z = ∑<sub>i</sub> u<sub>1</sub><sup>N<sub>1,i</sub></sup> u<sub>2</sub><sup>N<sub>2,i</sub></sup> ... u<sub>k</sub><sup>N<sub>k,i</sub></sup> e<sup>−βE<sub>i</sub></sup>

If we cheat a bit by treating the energy term (e<sup>−βE<sub>i</sub></sup>) 
as a constant for all i, we can treat the partition function as one huge 
polynomial. Each term in the polynomial represents a sentence as a bag of 
words, where the exponent is the word count. Since counts for sentences are 
sparse, and differentiation is a linear operator, we can calculate values 
for the jacobian and hessian very efficiently. The code that performs this 
calculation is in `sparsehess.py`.

There are some interesting connections between this way of thinking about
sentences and contexts in natural language and the way of thinking about
data types described in Conor McBride's 
"[The Derivative of a Regular Type is its Type of One-Hole Contexts](http://strictlypositive.org/diff.pdf)".
