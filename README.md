#  Discourse-Enhanced Transformers for Propaganda Detection

This repo contains code for the paper: [Unleashing the Power of Discourse-Enhanced Transformers for Propaganda Detection](https://aclanthology.org/2024.eacl-long.87/)

### Code structure 
1) ```Discourse Analysis.ipynb``` contains data preparation, including feature construction and analysis of the correlation between discourse and propaganda classes.
2) ```Deberta Token Classification.ipynb``` presents steps for the span classification task (paragraph classification) for the DeBERTa model: data preparation; model training; inference and error analysis. Relevant code is placed in ```glue_deberta```.
3) ```Deberta Token Classification.ipynb``` presents steps for the token classification task (NER) for the DeBERTa model. Relevant code is placed in ```ner_deberta_multi```
4) ```XLM-RoBERTa Span Classification.ipynb``` presents steps for the span classification task for the xlm-RoBERTa model. Relevant code is placed in ```glue_xlmroberta```

Main code in folders:
1) ```dataset_construction.py``` - SemEval-based dataset preparation. Adds linguistic features as inputs.
2) ```modeling_...py``` - model architecture modification (see Fig. Architecture). The main idea is the concatenation of linguistic features and Transformer-based embeddings.
3) ```...configuration...py``` - extended model configuration using new parameters (e.g., *extra_feature_size*)
4) ```run_...py``` - run a training cycle for the modified architecture with possible class weights in the loss function.
5) ```inference.py``` - inference of the trained model


**Model Architecture**
The model architecture is developed for the two tasks: (a) token classification; (b) span classification (paragraph-level).
The trainable blocks of the model are indicated in a blue color.
![Architecture](https://github.com/alchernyavskiy/discourse_propagada_detection/blob/main/architecture.png?raw=true)


### Data
Data can be downloaded from the official competition website: [SemEval2023 Task3](https://propaganda.math.unipd.it/semeval2023task3/)
```full_parsed_result_train_matched.pkl``` contains an example of dataset_construction stage output.


### Citation
If you find this repository helpful, feel free to cite our publication:
```
@inproceedings{chernyavskiy-etal-2024-unleashing,
    title = "Unleashing the Power of Discourse-Enhanced Transformers for Propaganda Detection",
    author = "Chernyavskiy, Alexander  and
      Ilvovsky, Dmitry  and
      Nakov, Preslav",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.87",
    pages = "1452--1462",
    abstract = "The prevalence of information manipulation online has created a need for propaganda detection systems. Such systems have typically focused on the surface words, ignoring the linguistic structure. Here we aim to bridge this gap. In particular, we present the first attempt at using discourse analysis for the task. We consider both paragraph-level and token-level classification and we propose a discourse-aware Transformer architecture. Our experiments on English and Russian demonstrate sizeable performance gains compared to a number of baselines. Moreover, our ablation study emphasizes the importance of specific types of discourse features, and our in-depth analysis reveals a strong correlation between propaganda instances and discourse spans.",
}

```