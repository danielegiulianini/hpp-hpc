/****************************************************************************
 *
 * mpi-isend-hpp.c - MPI implementation of the HPP model with 
 *                   asynchronous MPI_Isend operation
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
 This program contains MPI implementation of HPP model leveraging MPI_Isend 
 and MPI_Recv instead of MPI_Sendrecv as of mpi-sendrecv-hpp for halo exchange
 between processes.
 No constraints on the input are assumed (other than the ones stated
 by HPP specification itself).

 Taking the same local memory layout, the idea that differentiates this 
 version from mpi-sendrecv-hpp is that every process can actually compute all 
 but the last (during even phase) or first (during odd) row of blocks of its
 own partition while waiting for boundary information from neighbour process,
 instead of doing nothing in the meanwhile, like in mpi-sendrecv-hpp. 
 Indeed, information exchanged is only needed for the last (during even, 
 again) or first (during odd) row of 2x2 blocks. So, at least theoretically, 
 it's possible to reach best computation/communication ratio by decomposing:
 1. MPI_Sendrecv call in a MPI_Isend (and MPI_Wait) and MPI_Recv call 
 2. step invocation in two calls working on two different partitions of local
    domain, the first of which to be performed between MPI_Isend and MPI_Recv
    calls.
 */

/**
 * @brief Purposes of the procedure:
 * 1. Given the current state of the HPP CA for the partition made of `ext_rows` 
 *    rows and `ext_n` columns in `local_cur`, it computes the state of the CA 
 *    resulting from the given phase and writes it to `local_next`.
 * 2. handles the communication (with process assigned the neighbour partition)
 *    of boundary cells required for the given phase computation, by 
 *    leveraging MPI_Isend and MPI_Recv as a more efficient alternative to that
 *    of mpi-sendrecv-hpp, which exploits MPI_Sendrecv, instead.
 * Both partition grids passed incorporate top-row and left-column ghost cells
 * since ODD_PHASE computation, which assumes cyclic boundary conditions, would
 * cause out-of-bound access, or would need inefficient modulo operator,
 * otherwise.
 * 
 * @param local_cur pointer to the extended domain partition grid made of 
 * ext_rows x ext_n cells of cell_t containing its current configuration along
 * with top-row left-column ghost cells.
 * @param local_next pointer to the extended domain partition grid made of 
 * ext_rows x ext_n cells of cell_t that will contain the updated partition
 * resulting from the phase computation.
 * @param ext_rows count of the rows of the extended domain partition grid.
 * @param ext_n count of the columns of the extended domain partition grid.
 * namely, original domain side length (N) + 1.
 * @param phase phase to be computed, namely, one among EVEN_PHASE and 
 * ODD_PHASE.
 * @param rank_prev rank of the previous process in the chain, namely, the one
 * assigned with the domain rows preceding, or above, in the global grid, the 
 * ones of the caller process.
 * @param rank_next rank of the next process in the chain, namely, the one 
 * assigned with the domain rows following, or below, in the global grid, the
 * ones of the caller process.
 * @param row_t MPI datatype modelling a row of cells of actual (not extended) 
 * domain.
 */
void mpi_step(cell_t *local_cur,
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
   const int LEFT = HALO;

   /* With respect to mpi-sendrecv's mpi_step, a different order of execution of
       the operations (computation, copying columns, communication) is required:
       each mpi_step actually computes the phase after setting up memory and not
       before it. */

   /* Handle for knowing if MPI_Isend async operation has completed. */
   MPI_Request req;

   /* Copies boundary column resulting from previous phase (right after even 
       phase, left after odd) into opposite column (left after even, right 
       after odd), as it is actually needed for this phase computation as 
       respective neighbourhoods will focus on them.
       If next phase is odd this copy fills ghost column to avoid
       out-of-bound access without need of expensive modulo operator.
       Note that the choice of a (block, *) partitioning allows to spare a 
       communication call, since horizontal boundaries (right column after 
       even phase, left column after odd) are already in memory.

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
    |Y|                   Y|<-BOTTOM=ext_rows-1-> | |                    |
    +-+--------------------+          |           +-+--------------------+
       ^------ N ---------^           |              ^------ N ---------^
     ^------- ext_n ------^           |            ^------- ext_n ------^

    */
   if (phase == EVEN_PHASE)
   {
      /* Copies right (actual domain) column to (ghost) left column. */
      copy_left_to_right(local_cur, ext_n, ext_rows);
   }
   else
   {
      /* Copies left (ghost) column to right (actual domain) column. */
      copy_right_to_left(local_cur, ext_n, ext_rows);
   }

   /* Tracks source, i.e., who owns the boundary row updated at previous phase,
       and dest, i.e., who needs it for this phase computation, each depending on 
       the phase to compute. If in even phase, as previous odd worked with top 
       (ghost) row, updated row is to be sent to previous process in the chain,
       if odd, to the following one. */
   int dest = phase == ODD_PHASE ? rank_next : rank_prev;
   int source = phase == ODD_PHASE ? rank_prev : rank_next;

   /* Tracks the linear index of (the first element of) the boundary 
       row (last row after even phase and top (ghost) row after odd) updated
       at the previous phase, to send to neighbour that actually needs it for
       this phase computation (previous in the chain after odd and next after
       even). */
   int send_idx = phase == ODD_PHASE ? IDX(BOTTOM, LEFT_GHOST, ext_n)
                                     : IDX(TOP_GHOST, LEFT, ext_n);

   /* Asynchronously sends the boundary row updated during previous phase to 
       the process who needs it for current phase computation as it falls 
       inside the neighbourhoods of the cells of his partition. */
   MPI_Isend(&local_cur[send_idx], /* buf      */
             HALO,                 /* count    */
             row_t,                /* datatype */
             dest,                 /* dest     */
             0,                    /* tag      */
             MPI_COMM_WORLD,       /* comm     */
             &req);                /* request  */

   /* While waiting for the boundary row (updated by neighbour process 
       during previous phase) in transit through the network, instead of doing
       nothing like in mpi-sendrecv-hpp, process updates what already he can 
       without it, namely: all the partition but the row of blocks containing 
       that row (first ext_rows - BLOCK_DIM rows in even phase, last ext_rows 
       - BLOCK_DIM in odd). */
   int to_compute_soon_row_idx = phase == ODD_PHASE ? BLOCK_DIM : 0;

   /* If the process has been assigned an empty partition (in the edge case in
       which partitions are less than processes) or when it is made up of a 
       single row of blocks of BLOCK_DIM X BLOCK_DIM cells, then no row of 
       blocks is to be computed while the row of boundary cells is in transit. 
   */
   int rows_to_compute_soon = ext_rows > 0 ? ext_rows - HALO - BLOCK_DIM : 0;

   /* X denotes cells considered at this step call. 

               EVEN_PHASE:            |                   ODD_PHASE:     
                                      |
     LEFT_GHOST=0     RIGHT=ext_n-1   |          LEFT_GHOST=0     RIGHT=ext_n-1  
     | LEFT=1             |           |            | LEFT=1             |   
     | |                  |           |            | |                  | 
     v v                  v           |            v v                  v
    +-+--------------------+          |           +-+--------------------+ <
    | |                    | <-  TOP_GHOST=0   -> | |                    |  |
    +-+--------------------+          |           +-+--------------------+  > BLOCKDIM
    | |XXXXXXXXXXXXXXXXXXXX| <-     TOP=1      -> | |                    |  |
    +-+--------------------+          |           +-+--------------------+ <
    | |XXXXXXXXXXXXXXXXXXXX|          |           |X|XXXXXXXXXXXXXXXXXX| | 
    | |XXXXXXXXXXXXXXXXXXXX|          |           |X|XXXXXXXXXXXXXXXXXX| |
    | |XXXXXXXXXXXXXXXXXXXX|          |           |X|XXXXXXXXXXXXXXXXXX| |
    +-+--------------------+          |           +-+--------------------+
    | |                    |          |           |X|XXXXXXXXXXXXXXXXXX| | 
    +-+--------------------+          |           +-+--------------------+
    | |                    |<- BOTTOM=ext_rows-1->| |                  | |
    +-+--------------------+          |           +-+--------------------+
       ^------ N ---------^           |              ^------ N ---------^
     ^------- ext_n ------^           |            ^------- ext_n ------^

    */
   step(&local_cur[IDX(to_compute_soon_row_idx, 0, ext_n)],
        /* Sub-partition to_compute without boundary row for now. */
        &local_next[IDX(to_compute_soon_row_idx, 0, ext_n)],
        /* Only corresponding partition of local_next is updated for now. */
        rows_to_compute_soon + HALO,
        ext_n,
        phase);

   /* Tracks the linear index of (the first element of) the row to be filled 
      (top ghost row if now in odd phase and bottom row if now in even) by the
      one updated by neighbour process at previous phase and needed to 
      complete this phase. */
   int recv_idx = phase == EVEN_PHASE ? IDX(BOTTOM, LEFT, ext_n)
                                      : IDX(TOP_GHOST, LEFT_GHOST, ext_n);

   /* Process receives the boundary row updated during previous phase from the 
       process who updated it. */
   MPI_Recv(&local_cur[recv_idx], /* recvbuf   */
            HALO,                 /* recvcount */
            row_t,                /* datatype  */
            source,               /* source    */
            0,                    /* recvtag   */
            MPI_COMM_WORLD,       /* comm      */
            MPI_STATUS_IGNORE);   /* status    */

   /* remaining_row_idx tracks the row index of the first cell of the leftover row
       of BLOCKDIM x BLOCKDIM blocks to be computed to complete the phase (last
       but one for even, the first (with index 0) for odd). */
   int first_row_of_last_block_with_ghost = rows_to_compute_soon;
   int remaining_row_idx = phase == EVEN_PHASE ? first_row_of_last_block_with_ghost
                                               : 0;

   /* As the updated row from neighbour process is now available, process can 
       complete phase computation for its own partition with last (if even 
       phase) or first (if odd) remaining row of blocks of BLOCKDIM * BLOCKDIM 
       cells. 
       X denotes cells considered at this step call. 

               EVEN_PHASE:            |               ODD_PHASE:     
                                      |
     LEFT_GHOST=0     RIGHT=ext_n-1   |          LEFT_GHOST=0      RIGHT=ext_n-1  
     | LEFT=1             |           |            | LEFT=1             |   
     | |                  |           |            | |                  | 
     v v                  v           |            v v                  v
    +-+--------------------+          |           +-+--------------------+ <
    | |                    | <-  TOP_GHOST=0   -> |X|XXXXXXXXXXXXXXXXXX| |  |
    +-+--------------------+          |           +-+--------------------+  > BLOCKDIM
    | |                    | <-     TOP=1      -> |X|XXXXXXXXXXXXXXXXXX| |  |
    +-+--------------------+          |           +-+--------------------+ <
    | |                    |          |           | |                  | | 
    | |                    |          |           | |                  | | 
    | |                    |          |           | |                  | | 
    +-+--------------------+          |           +-+--------------------+
    | |XXXXXXXXXXXXXXXXXXXX|          |           | |                  | | 
    +-+--------------------+          |           +-+--------------------+
    | |XXXXXXXXXXXXXXXXXXXX|<-BOTTOM=ext_rows-1 ->| |                  | |
    +-+--------------------+          |           +-+--------------------+
       ^------ N ---------^           |              ^------ N ---------^
     ^------- ext_n ------^           |            ^------- ext_n ------^

    */

   step(&local_cur[IDX(remaining_row_idx, 0, ext_n)],
        &local_next[IDX(remaining_row_idx, 0, ext_n)],
        ext_rows - rows_to_compute_soon,
        ext_n,
        phase);

   /* Waiting for MPI_Isend to complete writing to MPI's buffer as first call
       to step in next phase would override memory locations pointed by 
       local_cur (so changing the content of communication message too, before 
       its delivery). If MPI_Wait is not called at all there's no warranty of 
       MPI_Isend completing until MPI_Finalize, which is not the case here, 
       since local_cur is reused during later phases.
       However, it's almost certain that the row has already been copied and 
       sent at this point, actually. */
   MPI_Wait(&req, MPI_STATUS_IGNORE);
}

/**
 * @brief Sends updated top (ghost) boundary row resulting from last odd phase 
 * to top neighbour's while filling bottom domain row with the one coming from
 * bottom neighbour's top one, which would otherwise hold outdated values.
 * Used after odd phase only, before gathering local domains at root into global
 * domain, since it would ignore updated top ghost row, otherwise.
 *  
 * @param local_cur local_cur points to the extended domain partition grid 
 * made of ext_rows x ext_n cells of cell_t containing its current 
 * configuration along with top-row and left-column ghost cells.
 * @param ext_rows count of the rows of the extended domain partition grid.
 * @param ext_n count of the columns of the extended domain partition grid. 
 * namely, original domain side length (N) + 1.
 * @param rank_prev rank of the previous process in the chain, namely the one
 * assigned with the domain rows preceding, or above, in the global grid, the 
 * ones of the caller process.
 * @param rank_next rank of the next process in the chain, namely the one 
 * assigned with the domain rows following, or below, in the global grid, the
 * ones of the caller process. 
 * @param row_t MPI datatype modelling a row of cells of actual (not extended)
 * domain.
 */
void sendrecv_first_row_from_prev(cell_t *local_cur,
                                  int ext_rows,
                                  int ext_n,
                                  int rank_prev,
                                  int rank_next,
                                  MPI_Datatype row_t)
{
   /* Usual memory layout. */
   const int BOTTOM = ext_rows - HALO;
   const int TOP_GHOST = 0;
   const int LEFT = 1;

   /* Process sends top row resulting from odd phase to process assigned with
       upper partition and fills its bottom row from the top row of the process
       assigned with the lower partition. 
       A graphical representation of the communication pattern adopted is:

       After ODD_PHASE:

                      LEFT_GHOST=0     RIGHT=ext_n-1 /|                                                   
                         LEFT=1               |     /         
                         | |                  |    /  
                         v v                  v   /
                        +-+--------------------+ /      +-+--------------------+
        TOP_GHOST=0 ->  | |XXXXXXXXXXXXXXXXXXXX|        | |XXXXXXXXXXXXXXXXXXXX|
                        +-+--------------------+        +-+--------------------+ 
  Pi          TOP=1 ->  | |                    |        | |                    |
                        | |                    |        | |                    |    
    BOTTOM=ext_rows-1-> | |UUUUUUUUUUUUUUUUUUUU|     _  | |XXXXXXXXXXXXXXXXXXXX| 
                        +-+--------------------+     /| +-+--------------------+                                                     
                                                    /          local_cur
                                                   /  
                                                  /
                        +-+--------------------+ /      +-+--------------------+
        TOP_GHOST=0 ->  | |XXXXXXXXXXXXXXXXXXXX|        | |XXXXXXXXXXXXXXXXXXXX|
                        +-+--------------------+        +-+--------------------+ 
  Pi+1        TOP=1 ->  | |                    |        | |                    |
                        | |                    |        | |                    |    
    BOTTOM=ext_rows-1-> | |ZZZZZZZZZZZZZZZZZZZZ|     _  | |VVVVVVVVVVVVVVVVVVVV| 
                        +-+--------------------+     /| +-+--------------------+    
                               local_cur            /          local_cur
                                                   /
                                                  / 
    */
   MPI_Sendrecv(&local_cur[IDX(TOP_GHOST, LEFT, ext_n)], /* sendbuf    */
                1,                                       /* sendcount  */
                row_t,                                   /* datatype   */
                rank_prev,                               /* dest       */
                0,                                       /* sendtag    */
                &local_cur[IDX(BOTTOM, LEFT, ext_n)],    /* recvbuf    */
                1,                                       /* recvcount  */
                row_t,                                   /* datatype   */
                rank_next,                               /* source     */
                0,                                       /* recvtag    */
                MPI_COMM_WORLD,                          /* comm       */
                MPI_STATUS_IGNORE);                      /* status     */
}

int main(int argc, char *argv[])
{
   int N, nsteps;
   FILE *filein;
   double tstart = -1;
   cell_t *cur = NULL;

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
      /* Default domain grid side length.*/
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
      const size_t GRID_SIZE = N * N * sizeof(cell_t);
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
       process, extended with ghost cells. As row partitioning is implied,
       it matches with side length of extended domain. */
   const int ext_n = N + HALO;

   int *sendcounts = (int *)malloc(comm_sz * sizeof(*sendcounts));
   assert(sendcounts != NULL);
   int *displs = (int *)malloc(comm_sz * sizeof(*displs));
   assert(displs != NULL);

   /* num_blocks_required tracks how many 2x2 (BLOCKDIM x BLOCKDIM) blocks
       are in a single row of the square domain grid. */
   const int num_blocks_required = N / BLOCK_DIM;

   for (int i = 0; i < comm_sz; i++)
   {
      /* Computes the starting and ending row index of the global domain
           to be assigned to a given process (row_iend is actually one row past 
           the ending row) by using a row block (block, *) partitioning 
           approach.
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
       chain. These will be used to exchange the boundaries. */
   const int rank_next = compute_rank_next(my_rank, comm_sz, sendcounts);
   const int rank_prev = compute_rank_prev(my_rank, comm_sz, sendcounts);

   /* local_rows tracks how many actual domain rows of cells a given process
       has to elaborate, while local_ext_rows tracks the ghost cells too. */
   const int local_rows = sendcounts[my_rank];
   const int local_ext_rows = local_rows != 0 ? local_rows + HALO : local_rows;

   /* Each process (master included) reserves the space for the local
       partition as it will update it. */
   cell_t *local_cur = (cell_t *)malloc(
       local_ext_rows * ext_n * sizeof(*local_cur));
   assert(local_cur != NULL);
   cell_t *local_next = (cell_t *)malloc(
       local_ext_rows * ext_n * sizeof(*local_next));
   assert(local_next != NULL);

   /* Custom MPI datatypes of help for the program. */
   MPI_Datatype row_t, row_t_resized, ext_part_t;

   /* row_t models a row of cells of the actual, not extended domain grid. 
       Used to make code cleaner by incorporating the count of the elements
       making up a row so that they need not to be stated at every 
       communication primitive. */
   MPI_Type_contiguous(N, MPI_CHAR, &row_t);
   MPI_Type_commit(&row_t);

   /* row_t_resized models a row extended with a ghost cell. Needed when
   processes are less than partitions. */
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

   /* For scattering domain partitions MPI_Scatterv primitive is used (instead
      of MPI_Scatter) because: given nproc as the arbitrary (as it depends on 
      program launch configuration) number of processes and nrows as the number
      of rows of 2x2 blocks to partition, nrows/nproc may not be an even 
      integer, so a non-uniform partitioning of rows to processes is needed 
      because processes can't work on a fraction of a row of 2x2 blocks, but
      MPI_Scatter doesn't allow it. */
   MPI_Scatterv(cur,                                /* sendbuf     */
                sendcounts,                         /* sendcounts  */
                displs,                             /* displs      */
                row_t,                              /* sendtype    */
                &local_cur[IDX(HALO, HALO, ext_n)], /* recvbuf     */
                local_rows,                         /* recvcount   */
                row_t_resized,                      /* recvtype    */
                0,                                  /* root        */
                MPI_COMM_WORLD);                    /* comm        */

#ifdef DUMP_ALL
   if (0 == my_rank)
   {
      /* Only the master dumps the current state to the output image. */
      write_image(cur, N, 0);
   }
#endif

   /* s tracks current step. Even phase of step 0 doesn't require preliminary 
       setup and communication operations because there's no previous odd phase 
       whose resulting boundary cells are to be copied. So, it's out of the CA 
       loop. */
   int s = 0;

   /* Even phase computation of step 0. */
   step(local_cur, local_next, local_ext_rows, ext_n, EVEN_PHASE);

   /* Odd phase computation of step 0. */
   mpi_step(local_next,
            local_cur,
            local_ext_rows,
            ext_n,
            ODD_PHASE,
            rank_prev,
            rank_next,
            row_t);

   /* First step is done. */
   s++;

   /* Evolution of CA along the steps. */
   for (; s < nsteps; s++)
   {

#ifdef DUMP_ALL
      /* Since gathering ignores top-row and left-column of ghost cells,
           but mpi_step for odd phase returns to caller without copying their
           updated values resulting from it into corresponding cells of actual domain 
           partition:
           1. a copy of left ghost column to right column is performed, along 
              with:
           2. overriding bottom row of domain partition with top ghost of 
              bottom-neighbour process row by sending it to top-neighbour one.
           This operation is needed whenever gathering local partitions in 
           global domain after odd phase, only, as for writing image to disk. 
           */
      copy_left_to_right(local_cur, ext_n, local_ext_rows);
      sendrecv_first_row_from_prev(local_cur,
                                   local_ext_rows,
                                   ext_n,
                                   rank_prev,
                                   rank_next,
                                   row_t);

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
    BOTTOM=ext_rows-1 ->| |XXXXXXXXXXXXXXXXXXXX|         |XXXXXXXXXXXXXXXXXXXX| 
                        +-+--------------------+   __\   +--------------------+
                                                  |   \  |YYYYYYYYYYYYYYYYYYYY|
                        +-+--------------------+  |__ /  |YYYYYYYYYYYYYYYYYYYY|
        TOP_GHOST=0 ->  | |                    |     /   |YYYYYYYYYYYYYYYYYYYY|
                        +-+--------------------+         +--------------------+ 
  Pi+1        TOP=1 ->  | |YYYYYYYYYYYYYYYYYYYY|        
                        | |YYYYYYYYYYYYYYYYYYYY|            
    BOTTOM=ext_rows-1 ->| |YYYYYYYYYYYYYYYYYYYY|         
                        +-+--------------------+        

        */
      MPI_Gatherv(&local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf   */
                  1,                                  /* sendcount */
                  ext_part_t,                         /* sendtype  */
                  cur,                                /* recvbuf   */
                  sendcounts,                         /* recvcounts*/
                  displs,                             /* displs    */
                  row_t,                              /* recvtype  */
                  0,                                  /* root      */
                  MPI_COMM_WORLD);                    /* comm      */

      if (0 == my_rank)
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
   }

#ifdef DUMP_ALL
   /* Like before, boundary (halo) row and column affected by odd phase are to 
       be copied to actual local domain since MPI_Gatherv considers it, not
       the extended one. */
   copy_left_to_right(local_cur, ext_n, local_ext_rows);
   sendrecv_first_row_from_prev(local_cur,
                                local_ext_rows,
                                ext_n,
                                rank_prev,
                                rank_next,
                                row_t);

   /* Reverses all particles and goes back to the initial state
       by reverting the order of phases inside a step. */
   for (; s < 2 * nsteps; s++)
   {
      MPI_Gatherv(&local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf   */
                  1,                                  /* sendcount */
                  ext_part_t,                         /* sendtype  */
                  cur,                                /* recvbuf   */
                  sendcounts,                         /* recvcount */
                  displs,                             /* displs    */
                  row_t,                              /* recvtype  */
                  0,                                  /* root      */
                  MPI_COMM_WORLD);                    /* comm      */

      if (my_rank == 0)
      {
         /* Only the master dumps the current state to the output image. */
         write_image(cur, N, s);
      }

      /* Odd phase computation. */
      mpi_step(local_cur,
               local_next,
               local_ext_rows,
               ext_n,
               ODD_PHASE,
               rank_prev,
               rank_next,
               row_t);

      /* Even phase computation. */
      mpi_step(local_next,
               local_cur,
               local_ext_rows,
               ext_n,
               EVEN_PHASE,
               rank_prev,
               rank_next,
               row_t);
   }

#endif

   /* If compiling with DUMP_ALL local domains are already set for gathering
       here. */
#ifndef DUMP_ALL
   copy_left_to_right(local_cur, ext_n, local_ext_rows);
   sendrecv_first_row_from_prev(local_cur,
                                local_ext_rows,
                                ext_n,
                                rank_prev,
                                rank_next,
                                row_t);
#endif

   MPI_Gatherv(&local_cur[IDX(HALO, HALO, ext_n)], /* sendbuf       */
               1,                                  /* sendcount     */
               ext_part_t,                         /* sendtype      */
               cur,                                /* recvbuf       */
               sendcounts,                         /* recvcounts    */
               displs,                             /* displs        */
               row_t,                              /* recvtype      */
               0,                                  /* root          */
               MPI_COMM_WORLD);                    /* comm          */

   if (0 == my_rank)
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

   /* Frees data types.*/
   MPI_Type_free(&row_t);
   MPI_Type_free(&row_t_resized);
   MPI_Type_free(&ext_part_t);

   MPI_Finalize();

   return EXIT_SUCCESS;
}
