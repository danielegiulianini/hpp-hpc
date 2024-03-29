## Intro
This repository contains three parallel implementations in C of the Hardy–Pomeau–Pazzis ([HPP](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=hpp+cellular+automaton&btnG=)) cellular automaton, a simple model for fluid propagation from the Lattice Gas family (Lattice-Gas Cellular Automata or LGCA). The automaton involves a two-dimensional stencil computation problem.

In particular, the repository contains:

- A parallel version leveraging a shared memory model via OpenMP, achieved by incrementally applying different optimization techniques.
- A parallel version utilizing a distributed memory model with Open MPI, exploiting synchronous MPI_SendRecv routines for communication between processes.
- A parallel version based on a distributed memory model with Open MPI, replacing MPI_SendRecv calls with MPI_Isend and MPI_Irecv asynchronous routines to further reduce communication overhead and improve performance.

The versions are not reimplementations of the same solution strategy, but provide 3 different ways to tackle the problem.


## Techs
C, OpenMP, Open MPI, Docker.


## Documentation
The solution strategies adopted are explained in the comments along with the code.
Contact me for further documentation.


## How to deploy & use
To ease deployment and make it platform-independent, this repository includes a Dockerfile that isolates the executables in a container.


### Prerequisites
- Docker (tested with version 20.10.22)
- Git (tested with version 2.30.1)

### Steps
1. Clone the repo into the desired folder:

```bash
    git clone https://github.com/danielegiulianini/hpp-hpc
```

2. Move inside the downloaded folder:

```bash
    cd hpp-hpc
```

3. Build the image by running:

```bash
    docker build -t hpp-hpc .
```

4. Start the container from the created image (implying source compilation) giving it a custom name (replacing &lt;container-name&gt; with it):

```bash
    docker run -it --name <container-name> hpp-hpc
```



5. Run the parallel programs and generate the animations of the HPP cellular automaton.
**Optional: configure the simulation parameters:**
- n. of threads/processes executing (default: 2);
- Input image (default: cannon.in; alternative: box.in)
- image size (default: 256 pixels);
by passing the corresponding key-value pairs to make ("[" and "]" just stands for optionality): 

```bash
    make all-movies [EXECUTION_UNITS=2 MOVIE_SIZE=256 INPUT_FILE=cannon.in]
```

6. Now you can inspect the video generated by the programs to check for their correctness (files named: omp-hppmovie.avi, mpi-sendrecv-hppmovie.avi, and mpi-isend-hppmovie.avi for OpenMP, Open MPI with MPI_SendRecv and Open MPI with MPI_Isend, respectively).

```bash
    mpv --no-config --vo=tct omp-hppmovie.avi
```

7. You can run the executables with different configurations to analyze performance and outputs; just remember to clean the previous programs outputs before rerunning:

```bash
    make clean
    make all-movies [EXECUTION_UNITS=4 MOVIE_SIZE=512 INPUT_FILE=cannon.in]
```

8. Finally, free up the resources associated to the container:

```bash
    docker remove <container-name>
```

