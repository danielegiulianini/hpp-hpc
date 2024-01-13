/****************************************************************************
 *
 * mpi-sendrecv-hpp.c - MPI implementation of the HPP model with 
 *                      synchronous MPI_Sendrecv operation
 *
 * Giulianini Daniele
 *
 * --------------------------------------------------------------------------
*/

#include "hpc.h"
#include <stdlib.h>
#include <assert.h>
#include "mpi_commons.h"

/* 
 This program contains MPI implementation of HPP model leveraging MPI_Sendrecv
 for halo exchange between processes.
 No constraints on the input are assumed (other than the ones stated
 by HPP specification itself).
 
 To:
 1. avoid out-of-bound access to domain grid and
 2. reach best access performances in indexing boundary cells, by giving out 
    tipically expensive modulo operator,
 ghost cell pattern is exploited for local partitions, while global domain
 is not extended.

 Parallelization is reached by:
 1. vertically splitting input domain in rectangular blocks of rows, one for 
    process.
 2. extending each of them with a top-row containing the bottom-row of the
    top neighbour process, assuming cyclic boundary conditions.
 3. extending each of them with a left-column containing process own rightest 
    column, since cyclic boundary conditions are assumed.
 4. performing each step computation of the partition in parallel, 
    iteratively exchanging boundary cells at the end of it.

 Having:
 - N as the input side length of the actual domain
 - ext_rows as the count of the rows of the grid partition extended with 
    above-mentioned ghost cells
 - ext_n as the count of the columns of the grid partition extended with 
    above-mentioned ghost cells
 in the following there will be frequent use of these constants,
 referring to local memory layout:

    TOP_GHOST = row index of top row halo 
    LEFT_GHOST = column index of left column halo 
    TOP = row index of first row of actual domain
    BOTTOM = row index of last row of actual domain
    LEFT = columnn index of first column of actual domain
    RIGHT = column index of last column of actual domain


                     LEFT_GHOST=0            RIGHT=ext_n-1
                         |   LEFT=1               |
                         |    |                   |         
                         v    v                   v         
                        +---+---+---+---+---+---+---+        
        TOP_GHOST=0 ->  | G | G | G | G | G | G | G |
                        +---+---+---+---+---+---+---+        
              TOP=1 ->  | G |   |   |   |   |   |   |
                        +---+---+---+---+---+---+---+               
                        | G |   |   |   |   |   |   |      
                        +---+---+---+---+---+---+---+        
  BOTTOM=ext_rows-1 ->  | G |   |   |   |   |   |   |                   
                        +---+---+---+---+---+---+---+        
                              ^------ N ----------^
                          ^------- ext_n ---------^

   where G stands for ghost area.
*/

/**
 * @brief Procedure purposes: 
 * 1. Given the current state of the HPP CA for the partition made of ext_rows
 *  rows and ext_N columns in local_cur, computes the state of the CA resulting 
 *  from the given phase and writes it to local_next.
 * 2. Prepares memory layout for the next phase by exchanging boundary 
 *  cells with neighbour, leveraging MPI_Sendrecv.
 * Both partition grids passed incorporate top-row and left-column ghost cells
 * since ODD_PHASE (computation or set-up) would cause out-of-bound access, 
 * otherwise. 
 * 
 * @param local_cur pointer to the extended domain partition grid made up of 
 * ext_rows x ext_n cells of cell_t containing its current configuration along
 * with top-row and left-column ghost cells. Left untouched by the procedure.
 * @param local_next pointer to the extended domain partition grid made up of 
 * ext_rows x ext_n cells of cell_t that will contain the updated partition
 * resulting from the phase computation.
 * @param ext_rows count of the rows of the extended domain partition grid.
 * @param ext_N count of the columns of the extended domain partition grid, 
 * namely, original domain side length (N) + 1 (HALO).
 * @param phase phase to be computed, namely, one among EVEN_PHASE and 
 * ODD_PHASE.
 * @param rank_prev rank of the previous process in the chain, namely, the one
 * assigned with the domain rows preceding, or above, in the global grid, the 
 * ones of the caller process.
 * @param rank_next rank of the next process in the chain, namely the one 
 * assigned with the domain rows following, or below, in the global grid, the
 * ones of the caller process.
 * @param row_t MPI datatype modelling a row of cells of actual domain.
 */
void mpi_step(const cell_t *local_cur,
              cell_t *local_next,
              int ext_rows,
              int ext_n,
              phase_t phase,
              int rank_prev,
              int rank_next,
              MPI_Datatype row_t)
{
   const int BOTTOM = ext_rows - HALO;
   const int LEFT_GHOST = 0;
   const int TOP_GHOST = 0;
   const int LEFT = 1;

   /* Computes the updated state of the partition leveraging (if odd phase)
       or not (if even phase) ghost cells already set at previous stages. */
   step(local_cur, local_next, ext_rows, ext_n, phase);

   /* Copies updated boundary column (left in even phase, right in odd phase)
       into opposite column (right in even, left in odd) actually needed for
       next phase computation as respective neighbourhoods will focus on them.
       If next phase is odd, this copy fills ghost column to avoid 
       out-of-bound access without need of expensive modulo operator.
       Note that the choice of a (block, *) partitioning allows to spare a 
       communication call, since horizontal boundaries (right column after even
       phase, left column after odd) are already in memory.

        After EVEN_PHASE:             |               After ODD_PHASE:     
                                      |
     LEFT_GHOST=0     RIGHT=ext_n-1   |         LEFT_GHOST=0     RIGHT=ext_n-1  
     | LEFT=1             |           |           | LEFT=1             |   
     | |                  |           |           | |                  | 
     v v                  v           |           v v                  v
    +-+--------------------+          |           +-+--------------------+
    | |                    | <- TOP_GHOST=0  ->   |Y|                   Y|
    +-+--------------------+          |           +-+--------------------+
    |Y|                   Y| <-     TOP=1    ->   |Y|                   Y|
    |Y|                   Y|          |           |Y|                   Y| 
    |Y|<----------------- Y|          |           |Y|-----------------> Y|
    |Y|                   Y|          |           |Y|                   Y| 
    |Y|                   Y|<-BOTTOM=ext_rows-1 ->| |                    |
    +-+--------------------+          |           +-+--------------------+
       ^------ N ---------^           |              ^------ N ---------^
     ^------- ext_n ------^           |            ^------- ext_n ------^

    */
   if (phase == EVEN_PHASE)
   {
      /* Copies right (actual domain) column to (ghost) left column. */
      copy_right_to_left(local_next, ext_n, ext_rows);
   }
   else
   {
      /* Copies left (ghost) column to right (actual domain) column. */
      copy_left_to_right(local_next, ext_n, ext_rows);
   }

   /* Tracks source, i.e., who owns the updated boundary row, and destination, 
       i.e., who needs it for next phase computation, each depending on the 
       phase to compute. If in odd phase, as it works with top (ghost) row, 
       updated row is to be sent to previous process in the chain, if even,
       to the following one. */
   int dest = phase == EVEN_PHASE ? rank_next : rank_prev;
   int source = phase == EVEN_PHASE ? rank_prev : rank_next;

   /* Tracks the linear index of (the first element of) the updated boundary 
       row (last row after even phase and top (ghost) row after odd) to send 
       to neighbour that actually needs it for next phase computation 
       (previous in the chain after even and next after odd). */
   int send_idx = phase == EVEN_PHASE ? IDX(BOTTOM, LEFT_GHOST, ext_n)
                                      : IDX(TOP_GHOST, LEFT, ext_n);

   /* Tracks the linear index of (the first element of) the row to be filled 
       by the one updated by neighbour process (top ghost row after even phase 
       and bottom row after odd) and needed for next phase. */
   int recv_idx = phase == EVEN_PHASE ? IDX(TOP_GHOST, LEFT_GHOST, ext_n)
                                      : IDX(BOTTOM, LEFT, ext_n);

   /*
    A graphical representation of the communication pattern adopted is:

    After EVEN_PHASE:

 Process                                          \
                      LEFT_GHOST=0     RIGHT=ext_n-1
                         | LEFT=1             |     \
                         | |                  |      \  
                         v v                  v      _\|
                        +-+--------------------+        +-+--------------------+
        TOP_GHOST=0 ->  | |                    |        |Y|YYYYYYYYYYYYYYYYYYY |
                        +-+--------------------+        +-+--------------------+ 
  Pi          TOP=1 ->  | |                    |        | |                    |
                        | |                    |        | |                    |    
    BOTTOM=ext_rows-1-> |X|XXXXXXXXXXXXXXXXXXX | _      |X|XXXXXXXXXXXXXXXXXXX | 
                        +-+--------------------+  \     +-+--------------------+    
                               local_next          \           local_next
                                                    \
                                                    _\|
                                                        +-+--------------------+
                                                        |X|XXXXXXXXXXXXXXXXXXX |
                                                        +-+--------------------+
  Pi+1                                                  | |                    |
                                                        | |                    |
                                                        |Z|ZZZZZZZZZZZZZZZZZZZ |
                                                        +-+--------------------+
 ---------------------------------------------------------------------------------
        After ODD_PHASE:
                                                        +-+--------------------+
                                                        | |YYYYYYYYYYYYYYYYYYYY|
                                                        +-+--------------------+
  Pi                                                    | |                    |
                                                        | |                    |
                                                     _  | |XXXXXXXXXXXXXXXXXXXX|
                      LEFT_GHOST=0     RIGHT=ext_n-1 /| +-+--------------------+                                                  
                         LEFT=1               |     /          local_next
                         | |                  |    /  
                         v v                  v   /
                        +-+--------------------+ /      +-+--------------------+
        TOP_GHOST=0 ->  | |XXXXXXXXXXXXXXXXXXXX|        | |XXXXXXXXXXXXXXXXXXXX|
                        +-+--------------------+        +-+--------------------+ 
  Pi+1        TOP=1 ->  | |                    |        | |                    |
                        | |                    |        | |                    |    
    BOTTOM=ext_rows-1-> | |ZZZZZZZZZZZZZZZZZZZZ|     _  | |VVVVVVVVVVVVVVVVVVVV| 
                        +-+--------------------+     /| +-+--------------------+    
                               local_next           /          local_next
                                                   /
                                                  / 
 */

   /* Note that sendbuf and recvbuf are not overlapping, like required by
      MPI_Sendrecv. */
   MPI_Sendrecv(&local_next[send_idx], /* sendbuf      */
                1,                     /* sendcount    */
                row_t,                 /* datatype     */
                dest,                  /* dest         */
                0,                     /* sendtag      */
                &local_next[recv_idx], /* recvbuf      */
                1,                     /* recvcount    */
                row_t,                 /* datatype     */
                source,                /* source       */
                0,                     /* recvtag      */
                MPI_COMM_WORLD,        /* comm         */
                MPI_STATUS_IGNORE);    /* status       */
}

int main(int argc, char *argv[])
{
   int N, nsteps;
   FILE *filein;
   cell_t *cur = NULL;

   double tstart = -1;

   int my_rank, comm_sz;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   if ((0 == my_rank) && ((argc < 2) || (argc > 4)))
   {
      fprintf(stderr, "Usage: %s [N [S]] input\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   if (argc > 2)
   {
      N = atoi(argv[1]);
   }
   else
   {
      /* Default domain grid side length. */
      N = 512;
   }

   if (argc > 3)
   {
      nsteps = atoi(argv[2]);
   }
   else
   {
      /* Default steps count of HPP CA. */
      nsteps = 32;
   }

   if ((0 == my_rank) && (N % BLOCK_DIM != 0))
   {
      fprintf(stderr, "FATAL: the domain size N must be even\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   if (0 == my_rank)
   {
      if ((filein = fopen(argv[argc - 1], "r")) == NULL)
      {
         fprintf(stderr, "FATAL: can not open \"%s\" for reading\n",
                 argv[argc - 1]);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }

      srand(1234); /* Initializes PRNG deterministically. */

      /* Side length of the global grid. No ghost cell is required for global
         domain. */
      const size_t GRID_SIZE = (N * N) * sizeof(cell_t);
      cur = (cell_t *)malloc(GRID_SIZE);
      assert(cur != NULL);

      /* Domain inizialization read from file. */
      read_problem(filein, cur, N);
      tstart = hpc_gettime();
   }

   /* Partitioning:
       a static, coarse-grained, row block partitioning (block, *) of input 
       domain grid into local domains, each assigned to a process, 
       is performed.
       This choice is enforced with respect to omp-hpp because:
        1. like in omp-hpp: 
          a. coarse-grained p.: HPP step computation is a kind of uniform, easy
             to balance between processes, computation (as assessed by 
             empirical evaluation) without the need of either fine-grained 
             partitioning nor dynamic assignment.
          b. row p.: as C memory layout is row-major order, namely, the 
             consecutive elements of a row reside next to each other, a row 
             partitioning provides better memory access performances by 
             optimizing cache accesses with the respect to other choices, 
             like column partitioning.
       2. additionally to omp-hpp:
          a. coarse grained p.: communication, even in distributed memory 
             architectures, is expensive so it's important to reduce the number
             of messages exchanged, by increasing computation/communication 
             ratio with respect to shared memory architectures.
          b. row p.: since HPP assumes cyclic boundary conditions, row 
             partitioning allows to reduce interactions between processes, 
             allowing each process to already possess locally the boundary
             information of horizontal neighbours (left or right depending on 
             the phase), hence not requiring their retrieval from neighbour
             (w.r.t. a square block partitioning approach, for example), so 
             providing a further minimizing strategy for both the number of
             communications and the network bandwidth needed by the program. */

   /* Count of the columns of the partition of the grid assigned to each
       process extended with ghost cells. As row partitioning is implied,
       it matches with side length of global, extended domain.*/
   const int ext_n = N + HALO;

   /* sendcounts tracks how many row_t are to be sent to each process. */
   int *sendcounts = (int *)malloc(comm_sz * sizeof(*sendcounts));
   assert(sendcounts != NULL);
   /* displs tracks how many row_t are there from the begin of the 
       global domain to the first row_t of each process. */
   int *displs = (int *)malloc(comm_sz * sizeof(*displs));
   assert(displs != NULL);
   /* num_blocks_required tracks how many 2x2 (BLOCKDIM x BLOCKDIM) blocks
       are in a single row of the square domain grid. */
   const int num_blocks_required = N / BLOCK_DIM;

   for (int i = 0; i < comm_sz; i++)
   {
      /* Computes the starting and ending row index of the global domain
         to be assigned to a given process (row_iend is actually one row past 
         the ending row) by using a row block (block, *) partitioning approach.
         BLOCK_DIM multiplication ensures that minimum granularity in
         assignment to processes is that of a row of blocks of 2x2 cells 
         instead of that of a row of cells, so as to avoid that a process
         receives an odd count of rows. In doing that, after this step, 
         knowledge of a block is no more needed in communication primitives,
         but that of a row is used. */
      const int row_istart = num_blocks_required * i / comm_sz * BLOCK_DIM;
      const int row_iend = num_blocks_required * (i + 1) / comm_sz * BLOCK_DIM;
      const int blklen = row_iend - row_istart;
      sendcounts[i] = blklen;
      displs[i] = row_istart;
   }

   /* Computes the rank of the next and previous process on the
      chain. These will be used to exchange the boundaries. 
      Particular attention is needed for handling the edge cases in which
      processes (p) launched by user are MORE than partitions, like, for ex.,
      in a execution scenario with N=4 (so, 2 rows (i.e. partitions) of 
      BLOCKDIM x BLOCKDIM) and p=3, where the ones without them must not take
      part in communication. Instead of heaving the code by repeated if-elses,
      MPI_PROC_NULL is adopted. */
   const int rank_next = compute_rank_next(my_rank, comm_sz, sendcounts);
   const int rank_prev = compute_rank_prev(my_rank, comm_sz, sendcounts);

   /* local_rows tracks how many actual domain rows of cells a given process
      has to elaborate, while local_ext_rows tracks the ghost cells too. */
   const int local_rows = sendcounts[my_rank];

   /* If domain partition for this process is empty (as program was 
      launched with too many processes w.r.t. the global domain to split among 
      them so that there are not enough rows for all) then doesn't give him 
      halo neither. */
   const int local_ext_rows = local_rows != 0 ? local_rows + HALO : local_rows;

   /* Custom MPI datatypes of help for the program. */
   MPI_Datatype row_t, row_t_resized, ext_part_t;

   /* row_t models a row of cells of the actual, not extended domain grid. 
       Used to make code cleaner by incorporating the count of the elements
       making up a row so that they need not to be stated at every 
       communication primitive. */
   MPI_Type_contiguous(N, MPI_CHAR, &row_t);
   MPI_Type_commit(&row_t);

   /* row_t_resized models a row extended with a ghost cell. As reported in
      https://github.com/open-mpi/ompi/issues/9619, in the edge case in which 
      there are more processes than partitions MPI_Scatterv (in OpenMPI 4.0.3
      (the one used in isiraptor as of 01/03/22) and others) is affected by a
      bug that causes it to hang (for processes, other than the root, that 
      try to receive a type that has count zero) and which will be 
      resolved in OpenMPI 5. For that reason, in MPI_Scatterv call, this type 
      is used instead of directly leveraging ext_part_t. 
      The new extent set by MPI_Type_create_resized to N+1 allows to space rows
      into local, extended partitions when received by processes and to 
      compress them back when gathered at master only. */
   MPI_Type_create_resized(row_t,           /* oldtype     */
                           0,               /* lower bound */
                           ext_n,           /* extent      */
                           &row_t_resized); /* newtype     */
   MPI_Type_commit(&row_t_resized);

   /* ext_part_t models memory locations between the first actual cell of a
      partition and the last one, but inside the (extended) memory layout local
      to single processes. Together with row_t_resized, it incorporates spaces
      for left column halo, used to correctly align cells when exchanging 
      domains between global and local layouts, and in particular for:
       1. scattering MORE THAN ONE contiguous ROW of the global domain, which
          is NOT extended with ghost cells, to local memory partitions that ARE
          extended with ghost cells for computational reason, instead.
       2. gathering the local updated extended partitions, always consisting 
          of more THAN ONE contiguous ROW of cells, to the global, not 
          extended, domain.
       To do those safely, in communication primitives, row_t and ext_part_t 
       are to be used with count values such that the pair (count1, row_t) 
       and (count2, ext_part_t) have the same type signature (Typesig = { 
       type0 , ... , typen-1 }) while not having the same type map (Typemap
        = { (type0,disp0), ... , (typen-1, dispn-1)}, where typei are basic
        types, and dispi are displacements). */
   MPI_Type_contiguous(local_rows,    /* count   */
                       row_t_resized, /* oldtype */
                       &ext_part_t);  /* newtype */
   MPI_Type_commit(&ext_part_t);

   /*
                        +---+---+---+---+
        row_t           | X | X | X | X |
                        +---+---+---+---+
                        ^------ N ------^

                        +---+---+---+---+---+
        row_t_resized   | X | X | X | X |   |
                        +---+---+---+---+---+
                        ^------ N ------^   
                        ^------ ext_n ------^

                        +---+---+---+---+<
        ext_part_t      | X | X | X | X | |
                    +---+---+---+---+---+ |
                    |   | X | X | X | X | |
                    +---+---+---+---+---+ > local_rows
                    |   | X | X | X | X | |
                    +---+---+---+---+---+ |
                    |   | X | X | X | X | |
                    +---+---+---+---+---+<
                        ^------ N ------^
                    ^------ ext_n ------^
     */

   /* Each process (master included) reserves the space for the local
      partition as it will update it. */
   cell_t *local_cur = (cell_t *)malloc(
       (local_ext_rows * ext_n) * sizeof(*local_cur));
   assert(local_cur != NULL);
   cell_t *local_next = (cell_t *)malloc(
       (local_ext_rows * ext_n) * sizeof(*local_next));
   assert(local_next != NULL);

   /* For scattering domain partitions MPI_Scatterv primitive is used (instead
      of MPI_Scatter) because: given nproc as the arbitrary (as it depends on 
      program launch configuration) number of processes and nrows as the number
      of rows of 2x2 blocks to partition, nrows/nproc may not be an integer, 
      so a non-uniform partitioning of rows to processes is needed because 
      processes can't work on a fraction of a row of 2x2 blocks, but
      MPI_Scatter doesn't allow it. */
   MPI_Scatterv(cur,                                /* sendbuf      */
                sendcounts,                         /* sendcounts   */
                displs,                             /* displs       */
                row_t,                              /* sendtype     */
                &local_cur[IDX(HALO, HALO, ext_n)], /* recvbuf      */
                local_rows,                         /* recvcount    */
                row_t_resized,                      /* recvtype     */
                0,                                  /* root         */
                MPI_COMM_WORLD);                    /* comm         */

   /* s tracks current step. */
   int s;

   /* Evolution of CA along the steps. */
   for (s = 0; s < nsteps; s++)
   {

#ifdef DUMP_ALL
      if (my_rank == 0)
      {
         /* Only the master dumps the current state to the output image. */
         write_image(cur, N, s);
      }
#endif

      /* Even phase computation. */
      mpi_step(local_cur,
               local_next,
               local_ext_rows,
               ext_n,
               EVEN_PHASE,
               rank_prev,
               rank_next,
               row_t);

      /* Odd phase computation. */
      mpi_step(local_next,
               local_cur,
               local_ext_rows,
               ext_n,
               ODD_PHASE,
               rank_prev,
               rank_next,
               row_t);

#ifdef DUMP_ALL
      /* Gathering the updated local domains at the root to prepare for
           writing the updated domain image to file. MPI_Gatherv is needed for
           the same motivation of MPI_Scatterv.
           ext_part_t type allows to correctly spacing rows at sending side
           while row_t at receiving (root) side allows to overlook ghost cell 
           corresponding to first (after even phase) or last (after odd) column
           as global domain is not extended with ghost cells, like shown below. 
           Top ghost row of each local domains partition is explicitly ignored
           at sending side.

 process                        local_cur                         cur
                        +-+--------------------+                   
        TOP_GHOST=0 ->  | |                    |        
                        +-+--------------------+         +--------------------+ 
  Pi+1        TOP=1 ->  | |XXXXXXXXXXXXXXXXXXXX|         |XXXXXXXXXXXXXXXXXXXX|
                        | |XXXXXXXXXXXXXXXXXXXX|         |XXXXXXXXXXXXXXXXXXXX|    
     BOTTOM=ext_rows-1->| |XXXXXXXXXXXXXXXXXXXX|         |XXXXXXXXXXXXXXXXXXXX| 
                        +-+--------------------+   __\   +--------------------+
                                                  |   \  |YYYYYYYYYYYYYYYYYYYY|
                        +-+--------------------+  |__ /  |YYYYYYYYYYYYYYYYYYYY|
        TOP_GHOST=0 ->  | |                    |     /   |YYYYYYYYYYYYYYYYYYYY|
                        +-+--------------------+         +--------------------+ 
  Pi+1        TOP=1 ->  | |YYYYYYYYYYYYYYYYYYYY|        
                        | |YYYYYYYYYYYYYYYYYYYY|            
     BOTTOM=ext_rows-1->| |YYYYYYYYYYYYYYYYYYYY|         
                        +-+--------------------+        

        */
      MPI_Gatherv(&local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf       */
                  1,                                  /* sendcount     */
                  ext_part_t,                         /* sendtype      */
                  cur,                                /* recvbuf       */
                  sendcounts,                         /* recvcounts    */
                  displs,                             /* displs        */
                  row_t,                              /* recvtype      */
                  0,                                  /* root          */
                  MPI_COMM_WORLD);                    /* comm          */
#endif
   }

#ifdef DUMP_ALL
   /* Reverses all particles and goes back to the initial state
      by reverting the order of phases inside a step. */
   for (; s < 2 * nsteps; s++)
   {
      if (my_rank == 0)
      {
         /* Only the master dumps the current state to the output image. */
         write_image(cur, N, s);
      }

      /* Odd phase computation. */
      mpi_step(local_cur,
               local_next,
               local_ext_rows,
               ext_n, ODD_PHASE,
               rank_prev,
               rank_next,
               row_t);

      /* Even phase computation. Passing local_cur and local_next in
         different order w.r.t previous mpi_step call allows to avoid
         explicit swap between pointers. */
      mpi_step(local_next,
               local_cur,
               local_ext_rows,
               ext_n,
               EVEN_PHASE,
               rank_prev,
               rank_next,
               row_t);

      /* Gathers local, extended partitions into global, not extended domain
         at root, like before. */
      MPI_Gatherv(&local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf       */
                  1,                                  /* sendcount     */
                  ext_part_t,                         /* sendtype      */
                  cur,                                /* recvbuf       */
                  sendcounts,                         /* recvcount     */
                  displs,                             /* displs        */
                  row_t,                              /* recvtype      */
                  0,                                  /* root          */
                  MPI_COMM_WORLD);                    /* comm          */
   }
#endif

/* If compiling with DUMP_ALL global domain is already set for dumping here. */
#ifndef DUMP_ALL
   MPI_Gatherv(
       &local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf       */
       1,                                  /* sendcount     */
       ext_part_t,                         /* sendtype      */
       cur,                                /* recvbuf       */
       sendcounts,                         /* recvcounts    */
       displs,                             /* displs        */
       row_t,                              /* recvtype      */
       0,                                  /* root          */
       MPI_COMM_WORLD);                    /* comm          */

#endif

   if (my_rank == 0)
   {
      double elapsed = hpc_gettime() - tstart;
      fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

      /* Only the master dumps the current state to the output image. */
      write_image(cur, N, s);
   }

   /* All done, frees memory. */
   free(local_cur);
   free(local_next);
   free(cur);
   free(sendcounts);
   free(displs);

   /* Frees data types. */
   MPI_Type_free(&row_t);
   MPI_Type_free(&row_t_resized);
   MPI_Type_free(&ext_part_t);

   MPI_Finalize();

   return EXIT_SUCCESS;
}
