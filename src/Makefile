############################################################################
##
## Makefile for HPP implementations
##
## Giulianini Daniele
##
############################################################################

## The targets defined by this makefile are:
##
## make         		compile all available sources.
## make clean   		delete temporary files and executables.
## make openmp  		compile the OpenMP version of the HPP model that 
##		produces a pgm format print of the final configuration.
## make mpi-sendrecv		compile the MPI version of the HPP model with 
##		MPI_Sendrecv for exchanging limits of local domains to individual 
##		processes, producing a pgm format print of the final configuration.
## make mpi-isend     		compile the MPI version of the HPP model with 
##		asynchronous MPI_Isend for exchanging limits of local domains to 
##		individual processes, producing a pgm format print of the final 
##		configuration.

## make openmp-dump-all 	compile the OpenMP version of the HPP model that 
##		produces a pgm format print of the configuration at each step.
## make mpi-sendrecv-dump-all	compile the MPI version with MPI_Sendrecv for 
##		exchanging local limits to individual processes in the domain of the 
##		HPP model, producing a pgm format print of the configuration at each 
##		step.
## make mpi-isend-dump-all	compile the MPI version with MPI_Isend for exchanging
##    	local limits to individual processes in the domain of the HPP model, 
##		producing a pgm format print of the configuration at each step.

## make openmp-movie     	compile the OpenMP version and execute it to generate
##	 	an animation in .movie format containing frames corresponding to 
##		individual steps of the automaton.
## make mpi-sendrecv-movie  compile the MPI version with MPI_Sendrecv and execute
##		it to generate an animation in .movie format containing frames 
##		corresponding to individual steps of the automaton.
## make mpi-isend-movie     	compile the MPI version with MPI_Isend and execute
##		it to generate an animation in .movie format containing frames 
##		corresponding to individual steps of the automaton.

## make all-movies     		compile all and execute all 3 versions (OpenMP, 
##		with MPI_Sendrecv, and with MPI_Isend) and generate the 3 corresponding 
##		animations in .movie format containing frames corresponding to 
##		individual steps of the automaton.


EXE_OMP=omp-hpp
EXE_MPI_SYNC=mpi-sendrecv-hpp
EXE_MPI_ASYNC=mpi-isend-hpp
DATAFILES:=
EXE:=$(EXE_OMP) $(EXE_MPI_SYNC) $(EXE_MPI_ASYNC)
EXE_DA:=$(EXE_OMP)-da $(EXE_MPI_SYNC)-da $(EXE_MPI_ASYNC)-da
CFLAGS+=-std=c99 -Wall -Wpedantic -O2
LDLIBS+=-lm

#configurable parameters:
EXECUTION_UNITS=2
MOVIE_SIZE=256
INPUT_FILE=cannon.in

.PHONY: clean openmp-dump-all mpi-sendrecv-dump-all mpi-isend-dump-all

ALL: $(EXE)

all-movies: openmp-movie mpi-sendrecv-movie mpi-isend-movie

openmp-movie: openmp-dump-all
	OMP_NUM_THREADS=$(EXECUTION_UNITS) ./$(EXE_OMP)-da $(MOVIESIZE) $(MOVIESIZE) $(INPUT_FILE)
	ffmpeg -y -i "hpp%05d.pgm" -vcodec mpeg4 $(EXE_OMP)movie.avi

mpi-sendrecv-movie: mpi-sendrecv-dump-all
	mpirun --allow-run-as-root -n $(EXECUTION_UNITS) ./$(EXE_MPI_SYNC)-da $(MOVIESIZE) $(MOVIESIZE) $(INPUT_FILE)
	ffmpeg -y -i "hpp%05d.pgm" -vcodec mpeg4 $(EXE_MPI_SYNC)movie.avi

mpi-isend-movie: mpi-isend-dump-all
	mpirun --allow-run-as-root -n $(EXECUTION_UNITS) ./$(EXE_MPI_ASYNC)-da $(MOVIESIZE) $(MOVIESIZE) $(INPUT_FILE)
	ffmpeg -y -i "hpp%05d.pgm" -vcodec mpeg4 $(EXE_MPI_ASYNC)movie.avi

openmp-dump-all: CFLAGS+=-fopenmp
openmp-dump-all:
	$(CC) $(CFLAGS) -DDUMP_ALL $(EXE_OMP).c -o $(EXE_OMP)-da $(LDLIBS)

mpi-sendrecv-dump-all:
	mpicc $(CFLAGS) -DDUMP_ALL $(EXE_MPI_SYNC).c -o $(EXE_MPI_SYNC)-da $(LDLIBS)

mpi-isend-dump-all:
	mpicc $(CFLAGS) -DDUMP_ALL $(EXE_MPI_ASYNC).c -o $(EXE_MPI_ASYNC)-da $(LDLIBS)

$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

$(EXE_MPI_SYNC): CC=mpicc
mpi-sendrecv: $(EXE_MPI_SYNC)

$(EXE_MPI_ASYNC): CC=mpicc
mpi-isend: $(EXE_MPI_ASYNC)

clean:
	\rm -f $(EXE) $(EXE_DA) hpp-movie *.o *~ *.pbm *.pgm *.avi
