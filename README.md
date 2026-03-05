## Syntax Meets Semantics: Understanding Scientific Formulae

A comprehensive understanding of scientific formulae requires modeling two fundamentally different formula modalities: 

**(1) structured syntax** (e.g., _symbols, operators, functions_) and **(2) semantic meaning**

We ask two research questions: 

(1) Do syntactic and semantic modalities of scientific formula _naturally_ align? 

(2) (_if misalignment exists_) Can a learned joint latent space better reconcile syntactic with semantic representation?

-----

### Dependencies
First, install all dependencies by running:

```
pip install -r requirements.txt
```

### Raw Alignment Analysis

Run the natural alignment of both modalities with:

```
python src/raw_alignment_analysis.py
```

This will produce all alignment score analyses as described in the paper.

### Learning Cross-Modal Alignment

Run the cross-modal alignment training with:

```
./run_all.sh  
```
This will run all methods described in the paper across the 5 cross-validation data splits. 

All required resources, including formula data, the 5-fold split files, and the structured and semantic modality vector data, are provided in the [`data`](./data/) directory. 
