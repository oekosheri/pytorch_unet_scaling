## GPU acceleration of Unet using Pytorch native environment and Horovod-pytorch

On the root directory you will find the scripts to run UNet (used for image semantic segmentation) implemented in Pytorch using a GPU data parallel scheme in Horovod-pytorch. In the native Tensorflow directory you will find the scripts to run the same training jobs using Tensorflow native environment without Horovod. The goal is to compare the paralleisation performance of Horovod-pytoch vs native Pytorch for a UNet algorithm. The data used here is an open microscopy data for semantic segmentation: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7639190.svg)](https://doi.org/10.5281/zenodo.7639190). These calculations have all been done on the [RWTH high performance computing cluster](https://help.itc.rwth-aachen.de/), using Tesla V100 GPUs. 

### Virtual environment

To install Horovod for Pytorch a virtual environment was created. The same environment will be used for both Pytorch native and Horovod trainings so that the results are comparable. Read this [README](./environments/README.md) to see how to set-up the environment and look at this [script](./load_env_rpcky.sh) to see which softwares and environments need to be loaded before creating the vitual env and running the jobs. For both native Pytorch and Horovod we use NCCL as the backend for collective communications. We also use Intel MPI for spawning the parallel processes.

### Data parallel scheme

A data parallel scheme is often used for large internet dataset sizes. In scientific datasets we usually have smaller dataset sizes but higher resolution images which evetually lead to OOM errors when you increase the batch size. Therefore using a data parallel scheme can be helpful. In a data parallel scheme, the mini-batch size is fixed per GPU (worker) and is usually the batch size that maxes out the GPU memory. In our case here it was 16. By having more GPUs the effective batch size increases and therefore run time decreases. This is a very common method of deep learning parallelisation. The drawback maybe that it can eventually lead to poor convergence and therefore model metrics (in our case intersection over union (IOU)) deteriorate. 

To implement the data parallel scheme the following necessary steps have been taken:

Submission:

- For native Pytorch during the submission process, some SLURM environmental variables have been set up which will help us access the size and ranks of workers during training. For Horovod-pytorch no env variables are required to be set up.

Training:

- Process initiation: There are different ways to initiate the processes in Pytorch native environment. Here we use TCP initialisation by specifying the IP address of the rank zero worker. To achieve this, we give the name of the master node as input argument to the training script. In Horovod intialisation is a simple call to horovod init. 
  
- Learning rate: A linear scaling rule is applied to the learning rate: it means the learning rate is multiplied by the number of workers (GPUs). In addition an optional initial warm-up and a gradual scheduling might help the convergence. The warm-up is commented out in our case as it didn’t provide improvement.
  
- Datasets and data loaders: Data loaders require a distributed sampler that takes in the number of workers and the rank of workers as input arguments.
  
- Model and optimisation: In Pytorch native the model should be wrapped in DDP (distributed data parallel) in Pytorch horovod the optimiser is wrapped in Horovod distributed optimizer.

- Horovod-pytorch requires some other minor edits to the training script that you can read [here](https://horovod.readthedocs.io/en/latest/pytorch.html)

### Submission: 

- The submission.sh file submits all jobs in a loop. For our data 14 GPUs was the maximum number of GPUs which for our computing cluster correlates with 7 nodes. The submission file adapts the run_file.sh (contatining the python script and its input arguments) and submit_file.sh (containing the submission script) for each job.
  
- Parallel MPI jobs are spawned by the env variable $MPIEXE in submit_file.
  
- Log files containing training times and metrics are copied in the logs folder on the root directory.

### Notebook

This [notebook](./notebooks/Loss_curves.ipynb) has been used for post processing of log files. We use two metrics to judge the parallelisation performance. First, the deviation from an ideal linear speed-up which corresponds to increasing the computational cost. Second, the model metrics, here IOU, which might decrease in comparison with a 1 GPU scenario as the loss convergence might suffer in a data parallel scheme.

In the figure below we compare the GPU parallelisation of Unet for Pytorch native and Horovod-pytorch. 
The trend in Model metric (IOU) vs. GPU seems to remain similar. However the computational efficiency of Pytorch native environment seems to outperform Horovod. We have repeated these calculation with different seeds and the behaviours observed here seem to be consistent.  



