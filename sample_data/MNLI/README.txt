This is the 1.0 distribution of the The Multi-genre NLI (MultiNLI) Corpus.

License information and a detailed description of the corpus are included in the accompanying PDF.

If you use this corpus, please cite the attached data description paper.

@InProceedings{williams2018broad,
  author    = {Williams, Adina and Nangia, Nikita and Bowman, Samuel R.},
  title     = {A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2018},
  publisher = {Association for Computational Linguistics},
}


Project page: https://www.nyu.edu/projects/bowman/multinli/


Release Notes
-------------

1.0:
- Replaces values in pairID and promptID fields. PromptID values are now shared across examples
  that were collected using the same prompt, as was originally intended, and pairID values are
  simply promptID values with an extra letter indicating the specific field in the prompt that was
  used. If you do not use these fields, this release is equivalent to 0.9.
