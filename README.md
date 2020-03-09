# Multi-Task Deep Neural Networks for Natural Language Understanding

This PyTorch package implements the Multi-Task Deep Neural Networks (MT-DNN) for Natural Language Understanding, as described in:

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


## FAQ

### Did you share the pretrained mt-dnn models?
Yes, we released the pretrained shared embedings via MTL which are aligned to BERT base/large models: ```mt_dnn_base.pt``` and ```mt_dnn_large.pt```. </br>
To obtain the similar models:
1. run the ```>sh scripts\run_mt_dnn.sh```, and then pick the best checkpoint based on the average dev preformance of MNLI/RTE. </br>
2. strip the task-specific layers via ```scritps\strip_model.py```. </br>

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