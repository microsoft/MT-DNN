# Multi-Task Deep Neural Networks for Natural Language Understanding

MT-DNN, an open-source natural language understanding (NLU) toolkit that makes it easy for researchers and developers to train customized deep learning models. Built upon PyTorch and Transformers, MT-DNN is designed to facilitate rapid customization for a broad spectrum of NLU tasks, using a variety of objectives (classification, regression, structured prediction) and text encoders (e.g., RNNs, BERT, RoBERTa, UniLM).  

A unique feature of MT-DNN is its built-in support for robust and transferable learning using the adversarial multi-task learning paradigm. To enable efficient production deployment, MT-DNN supports multi-task knowledge distillation, which can substantially compress a deep neural model without significant performance drop.  We demonstrate the effectiveness of MT-DNN on a wide range of NLU applications across general and biomedical domains.  

This repository is a pip installable package that implements the Multi-Task Deep Neural Networks (MT-DNN) for Natural Language Understanding, as described in the following papers:

Xiaodong Liu\*, Pengcheng He\*, Weizhu Chen and Jianfeng Gao<br/>
Multi-Task Deep Neural Networks for Natural Language Understanding<br/>
[ACL 2019](https://aclweb.org/anthology/papers/P/P19/P19-1441/) <br/>
\*: Equal contribution <br/>

Xiaodong Liu, Pengcheng He, Weizhu Chen and Jianfeng Gao<br/>
Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/1904.09482) <br/>


Pengcheng He, Xiaodong Liu, Weizhu Chen and Jianfeng Gao<br/>
Hybrid Neural Network Model for Commonsense Reasoning <br/>
[arXiv version](https://arxiv.org/abs/1907.11983) <br/>


Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Jiawei Han <br/>
On the Variance of the Adaptive Learning Rate and Beyond <br/>
[arXiv version](https://arxiv.org/abs/1908.03265) <br/>

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Tuo Zhao <br/>
SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization <br/>
[arXiv version](https://arxiv.org/abs/1911.03437) <br/>

Xiaodong Liu, Yu Wang, Jianshu Ji, Hao Cheng, Xueyun Zhu, Emmanuel Awa, Pengcheng He, Weizhu Chen, Hoifung Poon, Guihong Cao, Jianfeng Gao<br/>
The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/2002.07972) <br/>

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon and Jianfeng Gao<br/>
Adversarial Training for Large Neural Language Models <br/>
[arXiv version](https://arxiv.org/abs/2004.08994) <br/>

## Pip install package
A [setup.py](./setup.py) file is provided in order to simplify the installation of this package. 

1. To install the package, please run the command below (from directory root)  

    ```Python
    pip install -e .
    ```

1. Running the command tells pip to install the `mt-dnn` package from source in development mode. This just means that any updates to `mt-dnn` source directory will immediately be reflected in the installed package without needing to reinstall; a very useful practice for a package with constant updates.

1. It is also possible to install directly from Github, which is the best way to utilize the package in external projects (while still reflecting updates to the source as it's installed as an editable '-e' package).

    ```Python
    pip install -e git+git@github.com:microsoft/mt-dnn.git@master#egg=mtdnn
    ```

1. Either command, from above, makes `mt-dnn` available in your conda virtual environment. You can verify it was properly installed by running:
 
    ```Python
    pip list | grep mtdnn
    ```  
> For Mixed Precision and Distributed Training, please install NVIDIA apex by following instructions [here](https://github.com/NVIDIA/apex#linux)  

## Run an example  
An example Jupyter [notebook](./examples/classification/tc_mnli.ipynb) is provided to show a runnable example using the MNLI dataset. The notebook reads and loads the MNLI data provided for your convenience [here](./sample_data).  This dataset is mainly used for natural language inference (NLI) tasks, where the inputs are sentence pairs and the labels are entailment indicators.  

> **NOTE:** The MNLI data is very large and would need [Git LFS](https://docs.github.com/en/github/managing-large-files/installing-git-large-file-storage) installed on your machine to pull it down.  

## How To Use  


1. Create a model configuration object, `MTDNNConfig`, with the necessary parameters to initialize the MT-DNN model. Initialization without any parameters will default to a similar configuration that initializes a BERT model. This configuration object can be initialized wit training and learning parameters like `batch_size` and `learning_rate`. Please consult the class implementation for all parameters.   

    ```Python
    BATCH_SIZE = 16
    MULTI_GPU_ON = True
    MAX_SEQ_LEN = 128
    NUM_EPOCHS = 5
    config = MTDNNConfig(batch_size=BATCH_SIZE, 
                        max_seq_len=MAX_SEQ_LEN, 
                        multi_gpu_on=MULTI_GPU_ON)
    ```

1. Define the task parameters to train for and initialize an `MTDNNTaskDefs` object. Definition can be a single or multiple tasks to train. MTDNNTaskDefs can take a python dict, yaml or json file with task(s) defintion. 

    ```Python
    DATA_DIR = "../../sample_data/"
    DATA_SOURCE_DIR = os.path.join(DATA_DIR, "MNLI")
    tasks_params = {
                    "mnli": {
                        "data_format": "PremiseAndOneHypothesis",
                        "encoder_type": "BERT",
                        "dropout_p": 0.3,
                        "enable_san": True,
                        "labels": ["contradiction", "neutral", "entailment"],
                        "metric_meta": ["ACC"],
                        "loss": "CeCriterion",
                        "kd_loss": "MseCriterion",
                        "n_class": 3,
                        "split_names": [
                            "train",
                            "dev_matched",
                            "dev_mismatched",
                            "test_matched",
                            "test_mismatched",
                        ],
                        "data_source_dir": DATA_SOURCE_DIR,
                        "data_process_opts": {"header": True, "is_train": True, "multi_snli": False,},
                        "task_type": "Classification",
                    },
                }

    # Define the tasks
    task_defs = MTDNNTaskDefs(tasks_params)
    ```

1. Create a data tokenizing object, `MTDNNTokenizer`. Based on the model initial checkpoint, it wraps around the model's Huggingface transformers library to encode the data to **MT-DNN** format. This becomes the input to the data building stage.  

    ```
    tokenizer = MTDNNTokenizer(do_lower_case=True)

    # Testing out the tokenizer  
    print(tokenizer.encode("What NLP toolkit do you recommend", "MT-DNN is a fantastic toolkit"))  
    
    # ([101, 2054, 17953, 2361, 6994, 23615, 2079, 2017, 16755, 102, 11047, 1011, 1040, 10695, 2003, 1037, 10392, 6994, 23615, 102], None, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    ```

1. Create a data preprocessing object, `MTDNNDataBuilder`. This class is responsible for converting the data into the MT-DNN format depending on the task. This object is responsible for creating the vectorized data for each task.   

    ```
    ## Load and build data
    data_builder = MTDNNDataBuilder(tokenizer=tokenizer,
                                    task_defs=task_defs,
                                    data_dir=DATA_SOURCE_DIR,
                                    canonical_data_suffix="canonical_data",
                                    dump_rows=True)

    ## Build data to MTDNN Format as an iterable of each specific task
    vectorized_data = data_builder.vectorize()
    ```
 
1. Create a data preprocessing object, `MTDNNDataProcess`. This creates the training, test and development PyTorch dataloaders needed for training and testing. We also need to retrieve the necessary training options required to initialize the model correctly, for all tasks.  

    ```Python
    data_processor = MTDNNDataProcess(config=config, 
                                    task_defs=task_defs, 
                                    vectorized_data=vectorized_data)

    # Retrieve the multi task train, dev and test dataloaders
    multitask_train_dataloader = data_processor.get_train_dataloader()
    dev_dataloaders_list = data_processor.get_dev_dataloaders()
    test_dataloaders_list = data_processor.get_test_dataloaders()

    # Get training options to initialize model
    decoder_opts = data_processor.get_decoder_options_list()
    task_types = data_processor.get_task_types_list()
    dropout_list = data_processor.get_tasks_dropout_prob_list()
    loss_types = data_processor.get_loss_types_list()
    kd_loss_types = data_processor.get_kd_loss_types_list()
    tasks_nclass_list = data_processor.get_task_nclass_list()
    num_all_batches = data_processor.get_num_all_batches()
    ```

1. Now we can create an `MTDNNModel`. 
    ```Python
    model = MTDNNModel(
        config,
        task_defs,
        pretrained_model_name="bert-base-uncased",
        num_train_step=num_all_batches,
        decoder_opts=decoder_opts,
        task_types=task_types,
        dropout_list=dropout_list,
        loss_types=loss_types,
        kd_loss_types=kd_loss_types,
        tasks_nclass_list=tasks_nclass_list,
        multitask_train_dataloader=multitask_train_dataloader,
        dev_dataloaders_list=dev_dataloaders_list,
        test_dataloaders_list=test_dataloaders_list,
    )
    ```
1. At this point the MT-DNN model allows us to fit to the model and create predictions. The fit takes an optional `epochs` parameter that overwrites the epochs set in the `MTDNNConfig` object. 

    ```Python
    model.fit(epochs=NUM_EPOCHS)
    ```


1. The predict function can take an optional checkpoint, `trained_model_chckpt`. This can be used for inference and running evaluations on an already trained PyTorch MT-DNN model.  
Optionally using a previously trained model as checkpoint.  

    ```Python
    # Predict using a PyTorch model checkpoint
    checkpt = "./checkpoint/model_4.pt"
    model.predict(trained_model_chckpt=checkpt)

    ```

## Pre-process your data in the correct format  
Depending on what `data_format` you have set in the configuration object `MTDNNConfig`, please follow the detailed data format below to prepare your data:

- `PremiseOnly` : single text, i.e. premise. Data format is "id" \t "label" \t "premise" .  

- `PremiseAndOneHypothesis` : two texts, i.e. one premise and one hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis".  

- `PremiseAndMultiHypothesis` : one text as premise and multiple candidates of texts as hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis_1" \t "hypothesis_2" \t ... \t "hypothesis_n".  

- `Sequence` : sequence tagging. Data format is "id" \t "label" \t "premise".


## FAQ

### Did you share the pretrained mt-dnn models?
Yes, we released the pretrained shared embedings via MTL which are aligned to BERT base/large models: ```mt_dnn_base.pt``` and ```mt_dnn_large.pt```. </br>

### How can we obtain the data and pre-trained models to test to try out?
Yes, we have provided a [download script](./scripts/download.sh) to assist with this.  

### Why SciTail/SNLI do not enable SAN?
For SciTail/SNLI tasks, the purpose is to test generalization of the learned embedding and how easy it is adapted to a new domain instead of complicated model structures for a direct comparison with BERT. Thus, we use a linear projection on the all **domain adaptation** settings.

### What is the difference between V1 and V2
The difference is in the QNLI dataset. Please refere to the GLUE official homepage for more details. If you want to formulate QNLI as pair-wise ranking task as our paper, make sure that you use the old QNLI data. </br>
Then run the prepro script with flags:   ```> sh experiments/glue/prepro.sh --old_glue``` </br>
If you have issues to access the old version of the data, please contact the GLUE team.

### Did you fine-tune single task for your GLUE leaderboard submission? 
We can use the multi-task refinement model to run the prediction and produce a reasonable result. But to achieve a better result, it requires a fine-tuneing on each task. It is worthing noting the paper in arxiv is a littled out-dated and on the old GLUE dataset. We will update the paper as we mentioned below. 


## Notes and Acknowledgments
BERT pytorch is from: https://github.com/huggingface/pytorch-pretrained-BERT <br/>
BERT: https://github.com/google-research/bert <br/>
We also used some code from: https://github.com/kevinduh/san_mrc <br/>

## Related Projects/Codebase
1. Pretrained UniLM: https://github.com/microsoft/unilm <br/>
2. Pretrained Response Generation Model: https://github.com/microsoft/DialoGPT <br/>
3. Internal MT-DNN repo: https://github.com/microsoft/mt-dnn <br/>

### How do I cite MT-DNN?

```
@inproceedings{liu2019mt-dnn,
    title = "Multi-Task Deep Neural Networks for Natural Language Understanding",
    author = "Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1441",
    pages = "4487--4496"
}


@article{liu2019mt-dnn-kd,
  title={Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1904.09482},
  year={2019}
}


@article{he2019hnn,
  title={A Hybrid Neural Network Model for Commonsense Reasoning},
  author={He, Pengcheng and Liu, Xiaodong and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1907.11983},
  year={2019}
}


@article{liu2019radam,
  title={On the Variance of the Adaptive Learning Rate and Beyond},
  author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
  journal={arXiv preprint arXiv:1908.03265},
  year={2019}
}


@article{jiang2019smart,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Zhao, Tuo},
  journal={arXiv preprint arXiv:1911.03437},
  year={2019}
}
```
### Contact Information

For help or issues using MT-DNN, please submit a GitHub issue.

For personal communication related to this package, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Yu Wang (`yuwan@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`), Jianshu Ji (`jianshuj@microsoft.com`), Emmanuel Awa (`Emmanuel.Awa@microsoft.com`) or Jianfeng Gao (`jfgao@microsoft.com`).


# Contributing

This project welcomes contributions and suggestions.  For more details please check the complete steps to contributing to this repo [here](./CONTRIBUTION.md).
