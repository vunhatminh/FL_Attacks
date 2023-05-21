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
    | BitRand|[0.5->7.0] |[0.1 0.1, 0.1, 0.1, 0.2, 0.5, 1, 1,2, 6, 6, 11, 11, 17 ]*10^-1 | [10, 40]
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



