/****************************************************************************
 *
 * omp-hpp.c - OpenMP implementation of the HPP model
 *
 * Giulianini Daniele
 *
 * --------------------------------------------------------------------------
*/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "commons.h"

/* 
 This program contains OpenMP implementation of HPP model.
 No constraints on the input are assumed (other than the ones stated
 by HPP specification itself) and minimal effort is required for
 handling rectangular (even-sized) domain grid.
 
 To:
 1. avoid out-of-bound access to domain grid and
 2. reach best access performances in indexing boundary cells, by giving out
    tipically expensive modulo operator,
 ghost cell pattern is exploited.

 Parallelization is reached, after extending global domain with top-row and 
 left-column, by scheduling loop iterations to different threads and running 
 them in parallel, both for:
    1. ghost cell filling/copying
    2. single phase step

 Having:
 - N as the input side length of the actual, global domain
 - ext_n as the side length of the grid extended with above-mentioned 
    ghost cells,
 in the following there will be frequent use of these constants,
 referring to memory layout:


                     LEFT_GHOST=0     RIGHT=ext_n-1
                         |   LEFT=1       |
                         |    |           |         
                         v    v           v         
                        +---+---+---+---+---+        
        TOP_GHOST=0 ->  | G | G | G | G | G |
                        +---+---+---+---+---+        
             TOP=1  ->  | G |   |   |   |   |
                        +---+---+---+---+---+               
                        | G |   |   |   |   |        
                        +---+---+---+---+---+        
                        | G |   |   |   |   |        
                        +---+---+---+---+---+        
    BOTTOM=ext_n - 1 -> | G |   |   |   |   |                   
                        +---+---+---+---+---+        
                            ^------ N ------^
                        ^------- ext_n -----^

*/

/**
 * @brief Fills top-row and left-column ghost cells before odd (after even)
 * phase of a step of the CA with the opposite domain cells.
 * Since HPP model specifies cyclic boundary conditions, this procedure allows
 * to avoid using the generally poor-performing modulo operator by working on
 * a domain extended with 1 row of cells at top and 1 column of cells at left, 
 * where to store bottom row and left column resulting by even phase, 
 * respectively. 
 * A "#pragma omp parallel" block wrapping the call is assumed.
 * 
 * @pre
 *                    LEFT_GHOST=0     RIGHT=ext_n-1
 *                        | LEFT=1    ____ | _________
 *                        | |        |     |         |
 *                        v v        V     v         |
 *                       +-+----------------+        |
 *       TOP_GHOST=0 ->  |Z|XXXXXXXXXXXXXXX |        |
 *                       +-+----------------+        |
 *            TOP=1  ->  |Y|               Y|        |
 *                       |Y|               Y|        |
 *                       |Y|<------------- Y|        |
 *                       |Y|               Y|        |
 *                       |Y|               Y|        |
 *  BOTTOM=ext_n - 1 ->  | |XXXXXXXXXXXXXXXZ|        |
 *                       +-+------|---------+        |
 *                                |__________________|
 * 
 *                         ^------ N -------^
 *                       ^------- ext_n ----^
 *   
 * @param grid pointer to a square grid of ext_n x ext_n cells of cell_t.
 * @param ext_n side length of the square grid pointed by grid (extended
 * with ghost cells at top and left sides), namely, original domain length
 * side N + 1.
 */
void fill_ghost_cells_after_even(cell_t *grid, int ext_n)
{
    const int TOP = 1;
    const int BOTTOM = ext_n - 1;
    const int TOP_GHOST = TOP - 1;
    const int LEFT = 1;
    const int RIGHT = ext_n - 1;
    const int LEFT_GHOST = LEFT - 1;

/* As they are disjoint (apart from top-left cell), row and column copies are
   wrapt by the same "#pragma omp for" so as to perform them in parallel 
   without a synchronization point between column and row copy and a second 
   iterations-assignment overhead, like instead implied by:
    #pragma omp for
    for (int i = TOP_GHOST; i < BOTTOM; i++)
        grid[IDX(i, LEFT_GHOST, ext_n)] = grid[IDX(i, RIGHT, ext_n)];
    #pragma omp for
    for (int i = LEFT; i <= RIGHT; i++)
        grid[IDX(TOP_GHOST, i, ext_n)] = grid[IDX(BOTTOM, i, ext_n)];
   Anyway, these different approaches are not expected to affect overall 
   performance, as this procedure, that dependes on N, is dominated by step 
   computation, that depends on N^2 instead.
   Since copying ghost cells implies uniform work, namely, each cell requires
   the same amount of time to be copied, a static block assignment provides the
   best performance with respect to fine-grained partitioning associated or not
   with dynamic scheduling.
   Copy overlooks last cell of first row and first cell of last row because 
   they are ignored by both even and odd phases, each working on all but one
   row and one column of the extended grid.
*/
#pragma omp for schedule(static)
    for (int i = 1; i < ext_n - 1; i++)
    {
        grid[IDX(i, LEFT_GHOST, ext_n)] = grid[IDX(i, RIGHT, ext_n)];
        grid[IDX(TOP_GHOST, i, ext_n)] = grid[IDX(BOTTOM, i, ext_n)];
    }
/* As here there are more active threads (due to the "omp parallel" directive 
   assumed at call side) who edit concurrently the top-left grid cell, removing
   this "omp single" could instead cause a data race, as of the definition of 
   the term given by C language specification, and, for that, could potentially 
   result in undefined behaviour, as stated by specification. The implicit 
   barrier of omp single is required for not letting other threads work with an
   outdated value for that cell (inside step). */
#pragma omp single
    grid[IDX(TOP_GHOST, LEFT_GHOST, ext_n)] = grid[IDX(BOTTOM, RIGHT, ext_n)];
}

/**
 * @brief Fills bottom-row and right-column of extended domain grid after odd 
 * and before even phase of a step of the CA (or before writing image to disk 
 * with write_image_removing_ghost_cells) by copying them from opposite ghost 
 * cells filled at previous even phase.
 * Since HPP model specifies cyclic boundary conditions, this procedure allows
 * to avoid the use of the generally poor-performing modulo operator by working
 * on a domain extended with 1 row of cells at top and 1 column of cells at 
 * left, where to store bottom-row and left-column resulting by even phase,
 * respectively.
 * A "#pragma omp parallel" block wrapping the call is assumed.
 *   
 * @pre
 *                    LEFT_GHOST=0     RIGHT=ext_n-1
 *                        | LEFT=1    ____ | ________
 *                        | |        |     |         |
 *                        v v        |     v         |
 *                       +-+----------------+        |
 *       TOP_GHOST=0 ->  |Z|XXXXXXXXXXXXXXX |        |
 *                       +-+----------------+        |
 *            TOP=1  ->  |Y|               Y|        |
 *                       |Y|               Y|        |
 *                       |Y|-------------> Y|        |
 *                       |Y|               Y|        |
 *                       |Y|               Y|        |
 *  BOTTOM=ext_n - 1 ->  | |XXXXXXXXXXXXXXXZ|        |
 *                       +-+------^---------+        |
 *                                |__________________|
 * 
 *                         ^------ N -------^
 *                       ^------- ext_n ----^
 *   
 * @param grid pointer to a square grid of ext_n x ext_n cells of cell_t.
 * @param ext_n side length of the square grid pointed by grid (extended
 * with ghost cells at top and left sides), namely, original domain length
 * side N + 1.
 */
void fill_domain_boundary_cells_after_odd(cell_t *grid, int ext_n)
{
    const int TOP = 1;
    const int BOTTOM = ext_n - 1;
    const int TOP_GHOST = TOP - 1;
    const int LEFT = 1;
    const int RIGHT = ext_n - 1;
    const int LEFT_GHOST = LEFT - 1;

    /* Considerations of fill_ghost_cells_after_even applies here too. */
#pragma omp for schedule(static)
    for (int i = 1; i < ext_n - 1; i++)
    {
        grid[IDX(i, RIGHT, ext_n)] = grid[IDX(i, LEFT_GHOST, ext_n)];
        grid[IDX(BOTTOM, i, ext_n)] = grid[IDX(TOP_GHOST, i, ext_n)];
    }
#pragma omp single
    grid[IDX(BOTTOM, RIGHT, ext_n)] = grid[IDX(TOP_GHOST, LEFT_GHOST, ext_n)];
}

/**
 * @brief Copies square sub-matrix of matrix to sub_matrix, possibly leaving
 * out matrix's boundary rows and/or columns.
 * Useful for extracting actual domain from grid extended with ghost cells when
 * they are not needed anymore, such as before writing it to file with 
 * write_image.
 * 
 * @pre
 *   sub_matrix_j_start
 *       |        
 *       v                                        sub_matrix
 * +---+---+---+---+---+---+                          |
 * |   |   |   |   |   |   |                          v
 * +---+---+---+---+---+---+                        +---+---+---+---+---+ <---
 * |   | X | X | X | X | X | <-sub_matrix_i_start   | X | X | X | X | X |     |
 * +---+---+---+---+---+---+                        +---+---+---+---+---+     |
 * |   | X | X | X | X | X |                        | X | X | X | X | X |     |
 * +---+---+---+---+---+---+                        +---+---+---+---+---+     |
 * |   | X | X | X | X | X |                        | X | X | X | X | X |  sub_matrix_nrows
 * +---+---+---+---+---+---+                        +---+---+---+---+---+     |
 * |   | X | X | X | X | X |                        | X | X | X | X | X |     |
 * +---+---+---+---+---+---+                        +---+---+---+---+---+     |
 * |   | X | X | X | X | X |                        | X | X | X | X | X |     |      
 * +---+---+---+---+---+---+                        +---+---+---+---+---+ <---
 *     ^-sub_matrix_ncols--^                        ^-sub_matrix_ncols--^
 * ^------- ext_n ---------^
 * 
 * 
 * @param matrix pointer to a square matrix of ext_n x ext_n cells of cell_t
 * whose elements are left untouched.
 * @param sub_matrix pointer to the matrix wherein to copy elements from 
 * matrix, made up of sub_matrix_nrows rows and sub_matrix_ncols columns
 * @param ext_n side length of the square matrix pointed by matrix.
 * @param sub_matrix_i_start row index, with respect to matrix, of the 
 * first element to copy to sub_matrix.
 * @param sub_matrix_j_start column index, with respect to matrix, of 
 * the first element to copy to sub_matrix.
 * @param sub_matrix_nrows rows count to be copied from input matrix to 
 * submatrix. Must be less than either ext_n or the actual columns allocated
 * pointed by sub_matrix to avoid out-of-bound access.
 * @param sub_matrix_ncols columns count to be copied from input matrix to 
 * submatrix. Must be less than either ext_n or the actual columns allocated 
 * pointed by sub_matrix to avoid out-of-bound access.
 */
void copy_submatrix(const cell_t *matrix,
                    cell_t *submatrix,
                    int ext_n,
                    int submatrix_i_start,
                    int submatrix_j_start,
                    int submatrix_nrows,
                    int submatrix_ncols)
{
    /* This for loop could be parallelized too, but optimizing I/O operations
       is not the focus of the program and this routine is called inside
       image dumping only. */
    int sm_i = 0, i, j;
    for (i = submatrix_i_start; i - submatrix_i_start < submatrix_nrows; i++)
    {
        for (j = submatrix_j_start; j - submatrix_j_start < submatrix_ncols; j++)
        {
            submatrix[sm_i++] = matrix[IDX(i, j, ext_n)];
        }
    }
}

/**
 * @brief Writes an image of `ext_grid` to a file in PGM (Portable Graymap)
 * format ignoring ghost cells at top-row and left-column.
 * 
 * @param ext_grid pointer to a square grid of ext_n x ext_n cells of cell_t 
 * kept untouched by the procedure.
 * @param N side length of the square domain grid without ghost cells.
 * @param ext_n side length of the square grid pointed by grid (extended with 
 * ghost cells at top and left sides), namely, original domain side length 
 * (N) + 1.
 * @param frameno time step number, used for labeling the output file.
 * @param not_extended_dom the pointer to the already allocated matrix that 
 * after the call will host the actual, not extended domain. Passed as 
 * parameter for performance reasons (avoiding to reallocate that matrix 
 * every time).
 */
void write_image_removing_ghost_cells(const cell_t *ext_grid,
                                      int N,
                                      int ext_n,
                                      int frameno,
                                      cell_t *not_extended_dom)
{
    copy_submatrix(ext_grid,
                   not_extended_dom,
                   ext_n,
                   HALO,
                   HALO,
                   N,
                   N);
    write_image(not_extended_dom, N, frameno);
}

/**
 * @brief Computes the state of the CA resulting from the given phase and 
 * writes it to next, given the current one in cur.
 * Both grids passed must incorporate top-row and left-column ghost cells,
 * since ODD_PHASE computation would cause out-of-bound access, otherwise.
 * A "#pragma omp parallel" block wrapping the call is assumed.
 * 
 * @pre
 *  phase == EVEN_PHASE                       phase == ODD_PHASE
 *
 * +---+---+---+---+---+                     +---+---+---+---+---+
 * |   |   |   |   |   | <- TOP_GHOST        | d | c | d | c |   | <- TOP_GHOST
 * +---+---+---+---+---+                     +---+---+---+---+---+ 
 * |   | a | b | a | b | <- TOP              | b | a | b | a |   | <- TOP
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | c | d | c | d |                     | d | c | d | c |   |     
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | a | b | a | b |                     | b | a | b | a |   |     
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | c | d | c | d | <- BOTTOM           |   |   |   |   |   | <- BOTTOM    
 * +---+---+---+---+---+                     +---+---+---+---+---+
 *       ^----LEFT                                 ^----LEFT
 *   ^----- LEFT_GHOST                         ^----- LEFT_GHOST
 * 
 * Note that ghost cells are needed for odd phase only.
 * 
 * 
 * @param cur pointer to a square grid of ext_n x ext_n cells of cell_t 
 * containing the current configuration along with top-row and left-column 
 * ghost cells. Left untouched by the procedure.
 * @param next pointer to a square grid of ext_n x ext_n cells of cell_t that 
 * will contain the updated domain resulting from the phase computation.
 * @param ext_n side length of the square grid pointed by grid (extended with 
 * ghost cells at top and left sides), namely, original domain side length 
 * (N) + 1.
 * @param phase phase to be computed, namely, one among EVEN_PHASE and 
 * ODD_PHASE.
 */
void step_square(const cell_t *cur, cell_t *next, int ext_n, phase_t phase)
{
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO;
    const int TOP = HALO;
    const int BOTTOM = ext_n - HALO;

    int i, j;

    /* This is the loop wherein the most CPU time is spent. Since the work to 
       be performed is uniform all along the domain grid and hence easy to 
       balance between threads, the best performances (like explained in the
       attached report) are reached with static, coarse-grained partitioning
       assigning approximately total iterations/thread count for each thread. 
       Static scheduling, indeed, limits runtime overhead, and load balancing 
       is not a issue. */
    /* Given the absence of loop carried dependencies inside single phase
       computation, collapse clause is exploited for giving more freedom to the 
       compiler to parallelize them by explicitly stating their absence. 
       collapse is used even if performance improvement may actually lack due
       to the extra logic required, because this chance of extra 
       parallelization could be taken by compilers smarter than gcc, or in 
       future versions of it. */
#pragma omp for collapse(2) schedule(static)
    /* Loop over all coordinates (i,j) s.t. both i and j are even. */
    for (i = TOP; i < BOTTOM; i += 2)
    {
        for (j = LEFT; j < RIGHT; j += 2)
        {
            step_block(cur, next, ext_n, phase, i, j);
        }
    }
}

/**
 * @brief Evolves the CA from its starting state pointed by cur to the 
 * nsteps-th step (each of which made of an even and an odd phase), 
 * writing its final state to next.
 * 
 * @param cur pointer to a square grid of ext_n x ext_n cells of cell_t 
 * containing the starting configuration along with uninitialized top-row 
 * and left-column ghost cells.
 * @param next pointer to a square grid of ext_n x ext_n cells of cell_t that 
 * will contain the domain updated after nsteps steps.
 * @param nsteps steps to be computed by the CA.
 * @param ext_n side length of the square grid pointed by grid (extended with 
 * ghost cells at top and left sides), namely, original domain side length 
 * (N) + 1.
 */
void evolve_hpp(cell_t *cur,
                cell_t *next,
                int nsteps,
                int ext_n,
                cell_t *not_extended_dom)
{
    int t;

    /* Thread pool recycle: the computational burden of forking and destroying 
       pool of threads iteratively at every phase of every steps by using many
       "#pragma omp parallel for" is solved by generating and destroying them
       only once, here and at the end of the parallel region, respectively.
       This allows to reduce pool generation and destruction overhead not only
       for step computation but even for ghost cells filling, which happens in 
       parallel too.
       Actually, some compilers could have still optimized the code by 
       generating and destroying the pool once, but this behaviour is 
       implementation-specific and up to it. Moving #pragma omp parallel up
       here makes decoupling threads generation from threads scheduling 
       explicit and platform-independent. */
    /* Conditional compilation is required for facing different handling of 
      constants scope in GCC<9 with respect to GCC >=9. */
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(cur, next, nsteps, ext_n, not_extended_dom) private(t)
#else
#pragma omp parallel default(none) shared(cur, next, nsteps, ext_n, not_extended_dom, HALO) private(t)
#endif
    {
        for (t = 0; t < nsteps; t++)
        {

#ifdef DUMP_ALL
#pragma omp master
            /* Only one thread (master) writes image to disk. Since call is 
               inside a parallel region because of thread pool recycle, 
               "#pragma omp master" directive is required. A "#pragma omp single
               nowait", letting a single thread (not necessarily the master)
               perform dumping, could be used here too. */
            write_image_removing_ghost_cells(cur,
                                             ext_n - 1,
                                             ext_n,
                                             t,
                                             not_extended_dom);
            /* Although inside a "omp parallel block" with many threads active,
               image dumping can be performed safely without the need of a
               barrier for preventing race conditions, because dumping 
               procedure only reads cur (as of its first const specifier) and 
               the next point where cur can be edited (concurrently) (as stated 
               by step_square's first parameter const specifier) is after 
               a(n implicit) synchronization point inside step_square and so,
               at runtime, it will necessarily be after this dumping. */
#endif
            /* Even phase computation. */
            step_square(cur, next, ext_n, EVEN_PHASE);

            /* Immediately after even phase, top-row and left-column ghost 
               cells are kept untouched. Since:
               1. odd phase implies a different neighbourhood that focuses
                  on them,
               2. HPP specification assumes cyclic boundary conditions,
               grid bottom-row and right-column must be copied into them before
               odd phase. */
            fill_ghost_cells_after_even(next, ext_n);

            /* Odd phase computation. */
            step_square(next, cur, ext_n, ODD_PHASE);

            /* Perfoming even phase (or writing configuration to disk) requires
               the values updated by odd phase that, at this moment, still 
               reside on ghost cells only, which are not considered by the even
               neighbourhood. A copy of top-row and left-column to bottom-row 
               and right-column is therefore required before proceding. */
            fill_domain_boundary_cells_after_odd(cur, ext_n);
        }

#ifdef DUMP_ALL
        /* Reverses all particles and goes back to the initial state
           by reverting the order of phases inside a step. */
        for (; t < 2 * nsteps; t++)
        {

#pragma omp master
            write_image_removing_ghost_cells(cur,
                                             ext_n - HALO,
                                             ext_n,
                                             t,
                                             not_extended_dom);

            /* At first iteration, copy to ghost cells is not needed
               because it's been computed by the last call of the
               previous for loop. */
            /* Odd phase computation. */
            step_square(cur, next, ext_n, ODD_PHASE);

            fill_domain_boundary_cells_after_odd(next, ext_n);

            /* Even phase computation. */
            step_square(next, cur, ext_n, EVEN_PHASE);

            fill_ghost_cells_after_even(cur, ext_n);
        }
#endif
    }
}

int main(int argc, char *argv[])
{
    int N,      /* Side length of the original domain grid of the CA. */
        nsteps; /* Steps of the HPP CA to be performed. */

    FILE *filein; /* Pointer to file containing starting scene description. */

    double tstart, elapsed;

    srand(1234); /* Initialize PRNG deterministically. */

    if ((argc < 2) || (argc > 4))
    {
        fprintf(stderr, "Usage: %s [N [S]] input\n", argv[0]);
        return EXIT_FAILURE;
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

    if (N % BLOCK_DIM != 0)
    {
        /* Side length of the domain grid must be even because HPP model works
           on blocks of size 2x2 (BLOCKDIM X BLOCKDIM). */
        fprintf(stderr, "FATAL: the domain size N must be even\n");
        return EXIT_FAILURE;
    }

    if ((filein = fopen(argv[argc - 1], "r")) == NULL)
    {
        fprintf(stderr, "FATAL: can not open \"%s\" for reading\n",
                argv[argc - 1]);
        return EXIT_FAILURE;
    }

    /* Side length of the global grid extended with ghost cells at top-row and
       left-column exploited to perform odd phases without any special treatment,
       as they would otherwise cross both top and left-boundaries of original 
       domain. */
    const int ext_n = N + HALO;

    /* Count of cells of global grid extended with ghost cells. */
    const size_t GRID_SIZE = ext_n * ext_n * sizeof(cell_t);

    /* cur points to the global grid extended with ghost cells containing 
       current configuration. It's not updated throughout each phase but only
       at the very end of it. Actually, since during a single phase of a step 
       every thread works on a region disjoint from any of the other ones, 
       a single grid would suffice, but next is kept here for clarity 
       by following classic stencil implementation implying two domains.*/
    cell_t *cur = (cell_t *)malloc(GRID_SIZE);
    assert(cur != NULL);

    /* next points to the extended grid resulting from every single phase
       of a step.*/
    cell_t *next = (cell_t *)malloc(GRID_SIZE);
    assert(next != NULL);

    /* not_extended_dom's only purpose is to point to the actual domain when
       performing image dumping, which must ignore ghost cells. It does not
       take part in phase-updating logic. */
    cell_t *not_extended_dom = (cell_t *)malloc(N * N);
    assert(not_extended_dom != NULL);

    /* Domain inizialization read from file. 
       Ghost cells in cur remain uninitialized after read_problem call. */
    read_problem_in_subgrid(filein, cur, N, ext_n, HALO, HALO);

    tstart = hpc_gettime();

    /* Evolution of CA along the steps. */
    evolve_hpp(cur, next, nsteps, ext_n, not_extended_dom);

    elapsed = hpc_gettime() - tstart;

    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

#ifdef DUMP_ALL
    /* DUMP_ALL execution implies reverting all the steps hence doubling 
       them to return to initial configuration. */
    nsteps *= 2;
#endif

    /* Call to write_image_removing_ghost_cells overlooks ghost cells of cur. 
    */
    write_image_removing_ghost_cells(cur, N, ext_n, nsteps, not_extended_dom);

    free(cur);
    free(next);
    free(not_extended_dom);

    fclose(filein);

    return EXIT_SUCCESS;
}
