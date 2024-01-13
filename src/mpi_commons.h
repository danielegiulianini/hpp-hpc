/****************************************************************************
 *
 * mpi_commons.h - Code common to mpi-sendrecv-hpp.c and mpi-isend-hpp.c
 *
 * Giulianini Daniele
 *
 * --------------------------------------------------------------------------
*/

#ifndef COMMONS_H
#include "commons.h"
#endif

#ifndef MPI_COMMONS_H /* Include guard */
#include <mpi.h>
#define MPI_COMMONS_H

/***************************************************************************
 ***************************   GENERAL USE CODE  ***************************
 **************************************************************************/

/**
 * @brief Copies a column of matrix to another column of matrix, not necessarily 
 * square, from the row_start-th row cell to the row_end-th one.
 * 
 * @param matrix pointer to matrix of cell_t elements.
 * @param ext_n count of the columns of the matrix whose column is to be copied.
 * @param row_start row index of the top (and first) element of the column to
 * be copied.
 * @param row_end row index of the bottom (and last) element of the column to
 * be copied.
 * @param from_column column index of the column to be copied.
 * @param to_column column index of the column to be overridden.
 */
void copy_column_into_column(cell_t *matrix,
                             int ext_n,
                             int row_start,
                             int row_end,
                             int from_column,
                             int to_column)
{
    for (int i = row_start; i <= row_end; i++)
    {
        matrix[IDX(i, to_column, ext_n)] = matrix[IDX(i, from_column, ext_n)];
    }
}

/***************************************************************************
 *************************** HPP COMPUTATION CODE **************************
 **************************************************************************/

/**
 * @brief Fills right-column of domain grid partition before even (after
 * odd) phase of a step of the CA by copying them from opposite ghost cells
 * filled at previous odd phase.
 * Since HPP model specifies cyclic boundary conditions, this procedure allows
 * to avoid, during step computation, the use of the generally poor-performing
 * modulo operator by working on a domain extended with 1 row of cells at top
 * and 1 column of cells at left, where to store bottom row and left column 
 * resulting by odd phase, respectively. 
 * 
 * @pre
 * 
 *                    LEFT_GHOST=0     RIGHT=ext_n-1
 *                        | LEFT=1             | 
 *                        | |                  |         
 *                        v v                  v         
 *                       +-+--------------------+        
 *       TOP_GHOST=0 ->  |Y|                   Y|        
 *                       +-+--------------------+        
 *             TOP=1 ->  |Y|                   Y|        
 *                       |Y|                   Y|        
 *                       |Y|-----------------> Y|        
 *                       |Y|                   Y|        
 * BOTTOM=ext_rows-1 ->  | |                    |        
 *                       +-+--------------------+        
 *                          ^------ N ---------^
 *                        ^------- ext_n ------^
 *   
 * 
 * @param grid pointer to the extended domain partition grid made of 
 * ext_rows x ext_n cells of cell_t.
 * @param ext_n count of the columns of the extended domain partition grid 
 * namely, original domain side length (N) + 1.
 * @param ext_rows count of the rows of the extended domain partition grid.
 */
void copy_left_to_right(cell_t *grid, int ext_n, int ext_rows)
{
    const int BOTTOM = ext_rows - HALO;
    const int LEFT_GHOST = 0;
    const int TOP_GHOST = 0;
    const int RIGHT = ext_n - HALO;

    copy_column_into_column(grid,
                            ext_n,
                            TOP_GHOST,
                            BOTTOM - 1,
                            LEFT_GHOST,
                            RIGHT);
}

/**
 * @brief Fills left-column ghost cells before odd (after even)
 * phase of a step of the CA with the opposite domain cells.
 * Since HPP model specifies cyclic boundary conditions, this procedure allows
 * to avoid, during step computation, the use of the generally poor-performing
 * modulo operator by working on a domain extended with 1 row of cells at top
 * and 1 column of cells at left, where to store bottom row and left column 
 * resulting by even phase, respectively.
 * 
 *                     LEFT_GHOST=0     RIGHT=ext_n-1
 *                        | LEFT=1             | 
 *                        | |                  |         
 *                        v v                  v         
 *                       +-+--------------------+        
 *       TOP_GHOST=0 ->  | |                    |        
 *                       +-+--------------------+        
 *             TOP=1 ->  |Y|                   Y|        
 *                       |Y|                   Y|        
 *                       |Y|<----------------- Y|        
 *                       |Y|                   Y|        
 * BOTTOM=ext_rows-1 ->  |Y|                   Y|        
 *                       +-+--------------------+        
 *                          ^------ N ---------^
 *                        ^------- ext_n ------^
 * 
 * @param grid pointer to the extended domain partition grid made of 
 *  ext_rows x ext_n cells of cell_t.
 * @param ext_n count of the columns of the extended domain partition grid 
 * namely, original domain side length (N) + 1.
 * @param ext_rows count of the rows of the extended domain partition grid.
 */
void copy_right_to_left(cell_t *grid, int ext_n, int ext_rows)
{
    const int TOP = HALO;
    const int BOTTOM = ext_rows - HALO;
    const int RIGHT = ext_n - HALO;
    const int LEFT_GHOST = 0;

    copy_column_into_column(grid, ext_n, TOP, BOTTOM, RIGHT, LEFT_GHOST);
}

/**
 * @brief Computes the state of the CA resulting from the given phase 
 * given the current one in cur and writes it to next.
 * Both grids passed must incorporate top-row and left-column ghost cells,
 * since ODD_PHASE computation would cause out-of-bound access, otherwise. 
 * Domain grid being square is not a requirement.
 * 
 * @pre
 *  phase == EVEN_PHASE                              phase == ODD_PHASE
 *
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+
 * |   |   |   |   |   |   |   | <- TOP_GHOST -->   | d | c | d | c | b | a |   |
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+ 
 * |   | a | b | a | b | a | b | <- TOP      -->    | b | a | b | a | b | a |   |
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+     
 * |   | c | d | c | b | a | d |                    | d | c | d | c | b | a |   |     
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+     
 * |   | a | b | a | b | a | b |                    | b | a | b | a | b | a |   |     
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+     
 * |   | c | d | c | b | a | d | <- BOTTOM  -->     |   |   |   |   |   |   |   |    
 * +---+---+---+---+---+---+---+                    +---+---+---+---+---+---+---+
 *       ^----LEFT                                         ^----LEFT
 *   ^----- LEFT_GHOST                                 ^----- LEFT_GHOST
 * 
 * Note that ghost cells are needed for odd phase only.
 * 
 * 
 * @param cur pointer to the extended domain grid made of ext_rows x ext_n 
 * cells of cell_t containing the current configuration along with top-row
 * and left-column ghost cells. Left untouched by the procedure.
 * @param next pointer to the extended grid of ext_rows x ext_n cells of
 * cell_t that will contain the updated domain resulting from the phase 
 * computation.
 * @param ext_rows count of the rows of the extended domain grid.
 * @param ext_n count of the columns of the extended domain grid, namely, 
 * original domain side length (N) + 1.
 * @param phase phase to be computed, namely, one among EVEN_PHASE and 
 * ODD_PHASE.
 */
void step(const cell_t *cur, cell_t *next, int ext_rows, int ext_n, phase_t phase)
{
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO;
    const int TOP = HALO;
    const int BOTTOM = ext_rows - HALO;

    int i, j;

    /* Loop over all coordinates (i,j) s.t. both i and j are even. */
    for (i = TOP; i < BOTTOM; i += 2)
    {
        for (j = LEFT; j < RIGHT; j += 2)
        {
            step_block(cur, next, ext_n, phase, i, j);
        }
    }
}

/***************************************************************************
 ************************* HPP INITIALIZATION CODE *************************
 **************************************************************************/

/**
 * @brief Initializes HPP domain grid according to file specification written
 * using the scene description language described in 
 * read_problem_in_subgrid. It works on original domain grid,
 * NOT extended with ghost cells.
 * 
 * @param filein pointer to file containing the starting scene description.
 * @param grid pointer to domain grid of N x N cells of cell_t.
 * @param N side length of the domain grid to initialize.
 */
void read_problem(FILE *filein, cell_t *grid, int N)
{
    read_problem_in_subgrid(filein, grid, N, N, 0, 0);
}

/**
 * @brief Computes the rank of the next process in the (wrapt-around) chain of 
 * processes assigned with row-block partitions of the domain, taking into
 * consideration that, as number of processes depends on user, there may be 
 * more processes than partitions. In particular,
 * 1. a process with a non-empty partition must communicate (hence, have (next)
 * neighbour) ONLY with processes assigned with non-empty partitions. 
 * 2. a process assigned with a empty partition should not communicate (hence
 * should not have (next) neighbour) at all, so MPI_PROC_NULL is implied.
 * 
 * @param my_rank rank of the calling process.
 * @param comm_sz size of the group of processes associated with the 
 * communicator.
 * @param sendcounts array with the count of the domain rows each process 
 * is assigned to.
 * @return int the rank of the next process in the chain assigned with a 
 * non-empty partition. If calling process has an empty partition,
 * MPI_PROC_NULL is returned as it does not have to exchange anything with
 * neighbours.
 */
int compute_rank_next(int my_rank, int comm_sz, int *sendcounts)
{
    int rank_next;
    /* Iterating to retrieve the first process assigned with a non-empty
       partition. */
    for (rank_next = (my_rank + 1) % comm_sz;
         rank_next != my_rank && sendcounts[rank_next] == 0;)
    {
        rank_next = (rank_next + 1) % comm_sz;
    }
    /* If the calling process is assigned a non-empty partition it doesn't
       have to communicate with any neighbours.*/
    return sendcounts[my_rank] == 0 ? MPI_PROC_NULL : rank_next;
}

/**
 * @brief Computes the rank of the previous process in the (wrapt-around) chain
 * of processes assigned with row-block partitions of the domain, taking into
 * consideration that, as number of processes depends on user, there may be 
 * more processes than partitions.
 * 
 * @param my_rank rank of the calling process.
 * @param comm_sz size of the group of processes associated with the 
 * communicator.
 * @param sendcounts array with the count of the domain rows each process 
 * is assigned to.
 * @return int the rank of the previous process in the chain assigned with a 
 * non-empty partition. If calling process has an empty partition,
 * MPI_PROC_NULL is returned as it does not have to exchange anything with
 * neighbours.
 */
int compute_rank_prev(int my_rank, int comm_sz, int *sendcounts)
{
    /* Same considerations of compute_rank_next apply here too, but for 
       the previous neighbour in the chain. */
    int rank_prev;
    for (rank_prev = (my_rank - 1 + comm_sz) % comm_sz;
         rank_prev != my_rank && sendcounts[rank_prev] == 0;)
    {
        rank_prev = (rank_prev - 1 + comm_sz) % comm_sz;
    }
    return sendcounts[my_rank] == 0 ? MPI_PROC_NULL : rank_prev;
}

#endif // MPI_COMMONS_H
