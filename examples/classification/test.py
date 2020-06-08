# # Copyright (c) Microsoft Corporation. All rights reserved.
# ### Licensed under the MIT License.

# ## Multi-Task Deep Neural Networks for Natural Language Understanding
#
#
# This PyTorch package implements the Multi-Task Deep Neural Networks (MT-DNN) for Natural Language Understanding.

import sys

# ### The data
#
# This notebook assumes you have data already pre-processed in the MT-DNN format and accessible in a local directory.
#
#
# For the purposes of this example we have added sample data that is already processed in MT-DNN format which can be found in the __sample_data__ folder.
import torch

from mtdnn.common.types import EncoderModelType
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.modeling_mtdnn import MTDNNModel
from mtdnn.process_mtdnn import MTDNNDataProcess
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.data_builder_mtdnn import MTDNNDataBuilder
from mtdnn.tokenizer_mtdnn import MTDNNTokenizer

# ## Define Configuration, Tasks and Model Objects
# DATA_DIR = "../../sample_data/bert_uncased_lower/mnli/"
DATA_DIR = "/home/useradmin/sources/mt-dnn-orig/data/canonical_data_2/bert_base_uncased"
BATCH_SIZE = 16


# ### Define a Configuration Object
# Create a model configuration object, `MTDNNConfig`, with the necessary parameters to initialize the MT-DNN model.
# Initialization without any parameters will default to a similar configuration that initializes a BERT model.
config = MTDNNConfig(batch_size=BATCH_SIZE)


# ### Create Task Definition Object
# Define the task parameters to train for and initialize an `MTDNNTaskDefs` object. Create a task parameter dictionary.
# Definition can be a single or multiple tasks to train.  `MTDNNTaskDefs` can take a python dict, yaml or json file with task(s) defintion.
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
        "data_source_dir": "/home/useradmin/sources/mt-dnn-orig/data/MNLI",
        "data_process_opts": {"header": True, "is_train": True, "multi_snli": False,},
        "task_type": "Classification",
    },
}

# Define the tasks
task_defs = MTDNNTaskDefs(tasks_params)

## Define an MTDNN Tokenizer
# Default values: model = bert-base-uncased, do_lower = false
tokenizer = MTDNNTokenizer()

print(tokenizer.encode("He is a boy", "what is he"))

## Load and build data
data_builder = MTDNNDataBuilder(
    tokenizer=tokenizer,
    task_defs=task_defs,
    data_dir="/home/useradmin/sources/mt-dnn-orig/data",
    canonical_data_suffix="canonical_data_2",
    dump_rows=True,
)


## Build data to MTDNN Format
## Iterable of each specific task and processed data
vectorized_data = data_builder.vectorize()


# ### Create the Data Processing Object
# Create a data preprocessing object, `MTDNNDataProcess`. This creates the training, test and development PyTorch dataloaders
# needed for training and testing. We also need to retrieve the necessary training options required to initialize the model correctly, for all tasks.
# Define a data process that handles creating the training, test and development PyTorch dataloaders
# Make the Data Preprocess step and update the config with training data updates
data_processor = MTDNNDataProcess(
    config=config, task_defs=task_defs, vectorized_data=vectorized_data
)

# Retrieve the processed batch multitask batch data loaders for training, development and test
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


# Let us update the batch steps
num_all_batches = data_processor.get_num_all_batches()


# ### Instantiate the MTDNN Model
# Now we can go ahead and create an `MTDNNModel` model
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


# ### Fit on one epoch and predict using the training and test
# At this point the MT-DNN model allows us to fit to the model
# and create predictions. The fit takes an optional `epochs`
# parameter that overwrites the epochs set in the `MTDNNConfig` object.
model.fit(epochs=1)
model.predict()


# ### Obtain predictions with a previously trained model checkpoint

# The predict function can take an optional checkpoint, `trained_model_chckpt`.
# This can be used for inference and running evaluations on an already trained PyTorch MT-DNN model.
# Optionally using a previously trained model as checkpoint.
#
# ```Python
# # Predict using a MT-DNN model checkpoint
# checkpt = "<path_to_existing_model_checkpoint>"
# model.predict(trained_model_chckpt=checkpt)
# ```
