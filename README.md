This is the github repo for the code of the paper **Analysis of Privacy Leakage in Federated Large Language Models** [Accepted in AISTAT2024]

## Description
This repo not only contains the code for our AISTAT submission (LLMs folder) but also includes some extra theoretical results of our attacks (Theory folder) and experiments on other AI models (AMI and API folders).

## Dependencies
This codebase has been developed and tested only with python 3.8.10 and pytorch 1.7.0, on a linux 64-bit operation system.

### conda
We have prepared a file containing the same environment specifications that we use for this project. To reproduce this environment (only on a linux 64-bit OS), execute the following command:

```bash
$ conda create --name <name_env> --file spec-list.txt
```

- `name_env` is the name you want for your environment

Activate the created environment with the following command:

```bash
$ conda activate <name_env>
```

## Dataset and Preprocessing
For AISTAT 2024 experiments, the models and datasets are provided by [huggingface/datasets](https://huggingface.co/datasets). Follow [Link](https://huggingface.co/docs/transformers/en/installation) to install.
For AMI experiments on CIFAR10 and ImageNet datasets:

1. Follow the `README.md` file in `data/` to download the dataset
2. Run `$ python preprocessing.py` (The file is provided by https://github.com/trucndt/ami)

For API experiments, the Twitter and IMDB datasets are taken in [huggingface/datasets](https://huggingface.co/datasets). To install the datasets, run:

1. `conda install -c huggingface datasets transformers`

Our notebooks in the  `API/` folder demonstrate how to load the datasets. 

## Usage

Our code are distributed into three sub-directories:

- `Theory`: Experiments on synthetic One-hot, Spherical and Gaussian datasets
- `AMI`: Experiments of AMI on CIFAR10 and ImageNet datasets
- `API`: Experiments of API on IMDB and Twitter datasets

## Usage of code in `AMI/`
Simply run the two notebooks to reproduce the results in  (**Fig 7**)
Note that, the datasets need to be preprocessed as described above.

## Usage of code in `API/`

To reproduce the result of attention-based adversary on unprotected data (**Fig 4**), run:
`python api-att-imdb.py --beta [beta] --D [batchsize]`

  ***Example***:
> 	`python api-att-imdb.py --beta 0.1 --D 10`

***

We include two notebooks to reproduce the results of FC-based adversary in API (**Fig 8**):

- `api-fc-ldp-imdb.ipynb`: This notebook contains the experiments of the FC-based adversary on IMDB dataset protected by LDP mechanism. To change the LDP mechanisms (BitRand and OME), just uncomment the correspondings mechanisms in the `task_tpr` and `task_tnr` functions.
- `api-fc-ldp-twitter.ipynb`: This notebook contains the experiments of the FC-based adversary on Twitter dataset protected by LDP mechanism. To change the LDP mechanisms (BitRand and OME), just uncomment the correspondings mechanisms in the `task_tpr` and 

***

The results for Attention-based adversary (**Fig 8**) can be obtained with the following commands:

 1. For Twitter dataset:   `python api-att-ldp-twitter.py -e [privacy budget] -p [no.processes] -m [LDP-mech] --beta [beta] --times [no.games] --runs [no.runs] --D [batchsize] -o [result directory]`
 
 	 ***Example***:
 	> 	 `python api-att-ldp-twitter.py -e 5.0 -p 10 -m OME --beta
	 1.8 --times 100 --runs 10 --D 10 -o "twitter-res/"`
	    
	   The parameters to reproduce the reported results:

|LDP| Epsilon| Beta  | D
|--|--|--|--|
| BitRand|[0.5->7.0] |[0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1, 1,2, 6, 6, 11, 11, 17 ]*10^-1 | [10, 40]
|OME | [0.5->7.0] |1.8| [10, 40]


 3. For IMDB dataset:   `python api-att-ldp-imdb.py -e [privacy budget] -p [no.processes] -m [LDP-mech] --beta [beta] --times [no.games] --runs [no.runs] --D [batchsize] -o [result directory]`

	 ***Example***:
 	> 	 `python api-att-ldp-imdb.py -e 6.0 -p 1 -m BitRand --beta
	 0.05 --times 100 --runs 10 --D 10 -o "imdb-res/"`

    The parameters to reproduce the reported results:

|LDP| Epsilon| Beta  | D
|--|--|--|--|
| BitRand|[0.5->7.0] |[1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7]*10^-2 | [10, 40]
|OME | [0.5->7.0] |0.1| [10, 40]

## Usage of code in `Theory/`
Simply run the two notebooks to reproduce the results in  (**Fig 3**) and  (**Fig 6**).



## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
