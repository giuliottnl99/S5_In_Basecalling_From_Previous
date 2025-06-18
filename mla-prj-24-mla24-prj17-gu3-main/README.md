# MLA24-PRJ17-GU3-bioinspired-basecaller
This repository contain the code and scripts to train and evaluate nanopore basecalling neural networks Bonito based on the code base: https://github.com/marcpaga/nanopore_benchmark used for the paper Comprehensive benchmark and architectural analysis of deep learning models for nanopore sequencing basecalling available at: https://www.biorxiv.org/content/10.1101/2022.05.17.492272v2. 
Our repository give the possibility to replace N levels SLSTM with N levels SNN in the encoder of the architecture bonito. We built a fully spiking model with L2MU cells in the encoder. Details about L2MU can be found in the report present in this repo.

## Background
The project is a PyTorch-based Basecaller(the basecalling is the process of assigning nucleobases to electrical current changes resulting from nucleotides passing through a nanopore) where an existing repository( https://github.com/marcpaga/nanopore_benchmark ) is modified. The aim was to make the neural network more efficient while preserving accuracy as much as possible. To achieve this, Spiking neural networks were employed. In the 'bonitospikeconv' folder, you can find code to train and evaluate our proposed model.

## Installation
This code has been tested on python 3.9.16.
```

git clone https://github.com/MLinApp-polito/mla-prj-24-mla24-prj17-gu3.git
python3 -m venv gu3
source gu3/bin/activate
pip install --upgrade pip
pip install -r requirements_cluster.txt

```
## Getting started
### Data Download
To download the data check the bash scripts in ./download.
To download use this three scripts to download specific datasets: download_wick_train.sh and download_wick_test.sh

WARNING: This is a large data download, about 222GB in disk space. We used a reduced subset of the train dataset

WARNING: Many test dataset species currently lack correct data for testing. Please consider splitting train_data in subsets and test on one of them. To retrieve correct test data, please try to contact the original author in case of need 
## Data processing.
There is two main step for processing the data
#### Annotate the raw data with the reference/true sequence
For this step we used Tombo, which models the expected average raw signal level and aligns the expected signal with the measured signal. See their [documentation ](https://nanoporetech.github.io/tombo/resquiggle.html) for detailed info on how to use it.
After installation of tombo (you have to install it in a different environment, as it is not compatible with the training environment) you should be able to run the following.
```
source .bashrc
conda activate gu3.7
cd GU03-bioinspired-basecaller/sbonito
python ./apply_tombo.py --dataset-dir ./wick_to_be_resquiggled --processes 1

```
#### Chunk the raw signal and save it numpy arrays
In this step, we take the raw signal and splice it into segments of a fixed length so that they can be fed into the neural network.

This can be done by running the following script:
```
source .bashrc
conda activate gu3
python GU03-bioinspired-basecaller/sbonito/scripts/data_prepare_numpy.py \
--fast5-dir  mla24-prj17-gu3/sbonito/wick_to_be_resquiggled \
--output-dir  mla24-prj17-gu3/sbonito/new_train_numpy_after_resquiggle \
--total-files  4 \
--window-size 2000 \
--window-slide 0 \
--n-cores 4 \
--verbose
```
## Model Training
In this step we fed all the data we prepared (in numpy arrays), and train the model.

We can train four different types of model: bonito(classic architecture of Bonito), bonitosnn(with nlstm that indicate how many layer lstm you want in decoder) or bonitospikeconv (fully spiking model with leaky neurons in convolutional and l2mu cells in the encoder. l2mu parameter allows to specify desider depth of the encoder) :
```
source .bashrc
conda activate gu3
python /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/scripts/train_original.py \
--data-dir /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/new_train_numpy_after_resquiggle \
--output-dir /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/trained_bonito \
--model bonitospikeconv \
--window-size 2000 \
--batch-size 64 \
--starting-lr 0.001 \
--nl2mu 5

```
## Model Training with NNI
If you want to train the net using nni you must use the following script, model parameter can be bonitosnn to insert snn layer in the decoder, or bonitospikeconv  (Bonito with layer SNN in the feature extracture and in encoder) :
```
source .bashrc
conda activate gu3
python /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/experimentnni.py \
--data-dir ./new_wick2_train_numpy \
--output-dir ./test_nni_2 \
--model bonitosnn \
--nlstm 0 \
--train-file /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/scripts/train_originalnni.py \
--code-dir /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito \
--nni-dir /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito/nni-experiments \
--num-epochs 5
```
The tuning in bonitosnn works with the following hyperparameters:
```
 search_space = {
        'batch-size': {'_type': 'randint', '_value': [16, 128]},
        'starting-lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
        'slstm_threshold':{'_type': 'uniform', '_value': [0.01, 0.2]},
        }
```
The tuning in bonitospikeconv works with the following hyperparameters:
```
 search_space = {
        'batch-size': {'_type': 'randint', '_value': [16, 128]},
        'starting-lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
        'slstm_threshold':{'_type': 'uniform', '_value': [0.01, 0.2]},
        'conv_th':{'_type': 'uniform', '_value': [0.01, 0.2]},
        }
```
Bonitspikeconv tuning also performs tuning in leaky neurons present in feature extracture

## Basecalling
WARNING: all test species in the dataset currently lack fastA reference files to perform testing. 

Once a model has been trained, it can be used for basecalling. Here's an example command with the demo data:
```
source .bashrc
conda activate gu3
cd/home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito
python ./scripts/basecall_original.py \
--model bonitosnn \
--fast5-list ./inter_task_test_reads.txt \
--checkpoint ./trained/papermodels/inter_2000/checkpoint.pt \
--output-file ./trained/papermodels/inter_basecall_snn.fastq
```

## Evaluation
For the evaluation of the various model lunch this script:
```
source .bashrc
conda activate gu3
cd /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito
python3 evaluate.py --basecalls-path trained_bonito/inter_basecall_snn.fastq \
--references-path wick_to_be_resquiggled/all_references.fasta \
--output-file evaluations/bonito/fasta_bonito.csv \
--model-name bonitosnn \
--nlstm 2
```
WARNING: Not use underscore in model name

## Report
To create a report of the evaluation of your model based on the reference and basecalls do the following:
```
source .bashrc
conda activate gu3
cd /home/group_17//mla-prj-24-mla24-prj17-gu23/sbonito
python3 report.py \
--evaluation-file ./evaluations/bonito/fasta_bonito.csv \
--output-dir ./evaluations/bonito/reports \
--model-name bonitosnn
```
This will generate a bunch of report csv files in the output dir:

* absoultecounts: this contains the counts across all reads for different metrics (matched bases, mismatches bases, homopolymer correct bases, etc.)
* auc: this contains the values necessary to plot the AUC for the model.
* fraction: top fraction of best reads according to phredq.
* match_rate: match rate of reads in that fraction.
* phredq_mean: average PhredQ score of reads in that fraction.
* event rates: this contains the boxplot statistics for the main alignment events: match, mismatch, insertion and deletion.
* homopolymerrates: this containts the boxplot statistics for the homopolymer error rates per base or all together.
* phredq: this contains the boxplot statistics for the PhredQ scores of correctly and incorrectly basecalled bases.
* readoutcomes: this contains the number of reads that are successfully evaluated or that had some sort of error.
* signatures: this contains the rates and counts of different types of errors for each base in a 3-mer context. The 3-mer contexts are based on the basecalls, not the reference.
* singlevalues: this contains single summary values across all metrics based on the absolute counts, the read outcomes and the PhredQ scores distributions.

## Plots
For the plot of the various models lunch the following script(depth is how deep it will search between the folders):
```
conda activate gu3
python3 plot.py \
--reports evaluations \
--output-dir evaluations/plots \
--depth 5
```
