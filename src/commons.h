/******************************************************************************
 *
 * commons.h - Code common to omp-hpp.c, mpi-sendrecv-hpp.c and mpi-isend-hpp.c
 *
 * Giulianini Daniele
 *
 * ----------------------------------------------------------------------------
*/

#ifndef COMMONS_H /* Include guard */
#define COMMONS_H

#include <math.h>  /* for ceil() */
#include <stdio.h> /* for fopen(), fwrite()*/

/**
 * @brief Each of the possible states of a cell in HPP Cellular Automaton (CA).
 */
typedef enum
{
    WALL,
    GAS,
    EMPTY
} cell_value_t;

/**
 * @brief Phases that make up, alternatively, a single step.
 */
typedef enum
{
    ODD_PHASE = -1,
    EVEN_PHASE = 1
} phase_t;

/**
 * @brief The MPI datatype corresponding to "signed char" is MPI_CHAR.
 */
typedef signed char cell_t;

/**
 * @brief Number of ghost cells needed (both row-wise and column-wise).
 * */
const int HALO = 1;

/**
 * @brief Length, i.e., number of cells, of the side of the blocks that compose
 * HPP domain grid and make up a single neighborhood.
 */
const int BLOCK_DIM = 2;

/**
 * @brief Wraps the given index `i` around 0 or `N`, if needed.
 * 
 * @param i the index to be possibily wrapped.
 * @param N the upper bound the index has to be possibly wrapped around.
 * @return int the index wrapped around 0 or N, if needed.
 */
int wrap_index_around(int i, int N)
{
    return (i + N) % N;
}

/**
 * @brief Simplifies indexing on a `N`x`N` grid by returning the linear index
 * corresponding to row (`i`) and column(`j`) indexes in a square grid of size 
 * N while also wrapping them around 0 or N if negative or exceeding N, 
 * respectively.
 * Because of:
 * 1. the generally poor performance of modulo operator due to its absence 
 *    from native ISA support,
 * 2. the assumption of periodic boundary conditions implied by HPP 
 *    specification.
 * an indexing based on ghost cell that avoids use of modulo is a more 
 * efficient alternative to this, hence used for initialization code only.
 * 
 * @param i row index (possibly negative or exceeding N) of the cell to index
 * inside the square grid.
 * @param j column index (possibly negative or exceeding N) of the cell to 
 * index inside the square grid.
 * @param N side length of the square grid whose cell is to index.
 * @return int the linear index resulting by wrapping i and j.
 */
int IDX_mod(int i, int j, int N)
{
    i = wrap_index_around(i, N);
    j = wrap_index_around(j, N);
    return i * N + j;
}

/**
 * @brief Simplifies indexing on a `N`x`N` grid by returning the linear index 
 * corresponding row (`i`) and column(`j`) indexes in a square grid of size N.
 * No wrapping around grid boundary is performed so both i and j must be:
 * 1. non negative
 * 2. stricly less than N.
 * 
 * @param i row index (positive and less than N) of the cell to index inside 
 * the square grid.
 * @param j column index (positive and less than N) of the cell to index inside
 * the square grid.
 * @param N side length of the square grid whose cell is to index.
 * @return int the linear index corresponding to i and j, not wrapped.
 */
int IDX(int i, int j, int N)
{
    return i * N + j;
}

/***************************************************************************
 *************************** INITIALIZATION CODE ***************************
 **************************************************************************/

/**
 * @brief Maps a float value `v` from the range 0-1 to an int value in the
 * range 0-`N`.
 * 
 * @param v a float value in the range 0-1.
 * @param N the upper bound int value (lower is 0).
 * @return int the value after mapping to the range 0-N.
 */
int remap_to_N(float v, int N)
{
    /* ceil rounds up. */
    return ceil(v * N);
}

/**
 * @brief Returns an 1-D array index referencing the provided `ix` w.r.t.
 *`subarray_start`, for indexing a cell so that it lets subarray_start cells
 * untouched both at the beginning and, possibly, at the end. Cyclic boundary 
 * conditions are assumed, so wrap-around is performed. Useful for indexing
 * a cell in a 1-D array extended to contain ghost cells but the caller wants
 * to express cell position with respect to original, not extended array. 
 * Used for indexing a single coordinate (i or j) of the original 2-D domain 
 * inside grid extended with ghost cells, allowing to refactor initialization
 * code common to program versions working with actual domain (mpi-sendrecv-hpp
 * and mpi-isend-hpp) along with the one extended with ghost-cells (omp-hpp).
 * 
 * @pre
 *    subarray_start = 1
 *       |  ix = 1      
 *       |   |              
 *       |   |                
 *       v   v                
 * +---+---+---+---+---+---+
 * | X |   |   |   |   | X |
 * +---+---+-^-+---+---+---+
 *     ^-----|---N-----^
 *           |   
 *           |
 *          value returned = 2
 * 
 * In case of:
 * 1. negative indexes or 
 * 2. indexes exceeding actual original array (i.e., the subarray) length, a
 *    wrap-around that allows to keep room at caller side for halo
 *    (uninitialized at this stage) is performed, like shown:
 * 
 * ix = -1
 * subarray_start = 1
 *       |        
 *       |                
 *       |                  
 *       v                   
 * +---+---+---+---+---+---+
 * | X |   |   |   |   | X |
 * +---+---+-^-+---+-^-+---+
 *     ^------N = 4--|-^ 
 *                   |
 *       value returned = 1 + N-1 = 4
 * 
 * 
 * N.B. cells marked with X in the diagrams are left untouched, so that they can
 * be filled with ghost cells at a later stages.
 * 
 * @param ix index of the cell with respect to the not extended array whose 
 * index in the extended array is to get.
 * @param N length of the not extended 1-D array containing the cell to index
 * in the extended array.
 * @param subarray_start number of cells after whose to refer ix.
 * @return int index referring to the ixth cell but inside the extended array
 * so to keep subarray_start cells untouched.
 */
int get_ext_grid_coord_from_subgrid_coord(int ix, int N, int subarray_start)
{
    int ix_in_subgrid = wrap_index_around(ix, N);
    return ix_in_subgrid += subarray_start;
}

/**
 * @brief Gets linear index, corresponding to row and column indexes provided,
 * but with respect to the square subgrid with top-left cell at position 
 * (`subgrid_top_row_index`, `subgrid_left_column_index`) and size `N`, 
 * instead of (0, 0). Useful for working with same `ix` and `iy` for indexing 
 * cells both:
 * - in extended and
 * - in not-extended domain (with ghost cells), 
 * while moving the offsets to subgrid_top_row_index and 
 * subgrid_left_column_index parameters, so refactoring common inizialization
 * logic.
 * 
 * @param N side length of the smaller square grid; i.e., the original domain
 * grid wherein to write box.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param ix row index of the cell with respect to the top-left cell of the
 * not extended grid inside the extended one.
 * @param iy column index of the cell with respect to the top-left cell of 
 * the not extended grid inside the extended one.
 * @param subgrid_top_row_index row index, with respect to extended grid, of
 * the top-left cell of smaller square grid, so to keep subgrid_top_row_index
 * rows untouched at top and (possibly) bottom of the grid.
 * @param subgrid_left_column_index column index, with respect to the extended 
 * grid, of the top-left cell of the smaller square grid, so to keep 
 * subgrid_left_column_index columns untouched at the left and (possibly) 
 * right of the grid.
 * @return int linear index in extended domain grid referring row and column
 * indexes w.r.t. smaller grid instead of the extended one.
 */
int get_ext_grid_coords_from_subgrid_coords(int N,
                                            int ext_n,
                                            int ix,
                                            int iy,
                                            int subgrid_top_row_index,
                                            int subgrid_left_column_index)
{
    int ix_in_ext_grid = get_ext_grid_coord_from_subgrid_coord(ix,
                                                               N,
                                                               subgrid_top_row_index);
    int iy_in_ext_grid = get_ext_grid_coord_from_subgrid_coord(iy,
                                                               N,
                                                               subgrid_left_column_index);
    return IDX(iy_in_ext_grid, ix_in_ext_grid, ext_n);
}

/**
 * @brief Writes the given value `t` to the grid in the cell at position
 * (`ix`, `iy`) with respect to subgrid with top-left cell at position 
 * (`subgrid_top_row_index`, `subgrid_left_column_index`), instead of (0, 0).
 * Useful when indexing values in a grid extended to contain ghost cells but
 * application logic calling this prefers to express cell positions referred
 * to original, not extended domain.
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original domain 
 * grid wherein to write box.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param ix x coordinate, with respect to smaller grid, of the cell to write.
 * @param iy y coordinate, with respect to smaller grid, of the cell to write.
 * @param t value to assign to the cell.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void point_in_subgrid_i(cell_t *grid,
                        int N,
                        int ext_n,
                        int ix,
                        int iy,
                        cell_value_t t,
                        int subgrid_top_row_index,
                        int subgrid_left_column_index)
{
    grid[get_ext_grid_coords_from_subgrid_coords(N,
                                                 ext_n,
                                                 ix,
                                                 iy,
                                                 subgrid_top_row_index,
                                                 subgrid_left_column_index)] = t;
}

/**
 * @brief Draws a box with top-left cell at position (`ix1`, `iy1`) and 
 * bottom-right cell at position (`ix2`, `iy2`) with respect to the square 
 * subgrid with top-left cell at position (`subgrid_top_row_index`, 
 * `subgrid_left_column_index`) and side length `N`, instead of (0,0). 
 * 
 * 
 * @pre
 * 
 * If, for example, user wants to initialize domain like this:
 * 
 *     ix1 = 1      
 *       |     iy2 = 2         
 *       |       |         
 *       v       v         
 * +---+---+---+---+
 * | 0 | 0 | 0 | 0 |
 * +---+---+---+---+
 * | 0 | t | t | t |<- iy1 = 1
 * +---+---+---+---+
 * | 0 | t | t | t |<- iy2 = 2
 * +---+---+---+---+
 * | 0 | 0 | 0 | 0 |
 * +---+---+---+---+
 * 
 * while reserving room for halo, the function behaves like that:
 * 
 *   subgrid_top_row_index = 1
 *       |  ix1 = 1      
 *       |   |     iy2 = 2         
 *       |   |       |         
 *       v   v       v         
 * +---+---+---+---+---+---+
 * | X | X | X | X | X | X |
 * +---+---+---+---+---+---+
 * | X | 0 | 0 | 0 | 0 | X |<- subgrid_left_column_index = 1
 * +---+---+---+---+---+---+
 * | X | 0 | t | t | t | X |<- iy1 = 1
 * +---+---+---+---+---+---+
 * | X | 0 | t | t | t | X |<- iy2 = 2
 * +---+---+---+---+---+---+
 * | X | 0 | 0 | 0 | 0 | X |
 * +---+---+---+---+---+---+
 * | X | X | X | X | X | X |
 * +---+---+---+---+---+---+
 *     ^----- N -------^
 * ^------- ext_n ---------^
 * 
 * 
 * In case of:
 * 1. negative indexes 
 * 2. indexes exceeding actual domain (i.e., smaller grid) boundary, a 
 *    wrap-around that still keeps room for halo uninitialized is 
 *    performed, like shown:
 * 
 * 
 *   subgrid_top_row_index = 1
 *          |             
 * ix2 = 4  |      ix1 = 3      
 *       |__|        |         
 *       v           v        
 * +---+---+---+---+---+---+
 * | X | X | X | X | X | X |
 * +---+---+---+---+---+---+
 * | X | 0 | 0 | 0 | 0 | X |<- subgrid_left_column_index = 1
 * +---+---+---+---+---+---+
 * | X | t | 0 | 0 | t | X |<- iy1 = 1
 * +---+---+---+---+---+---+
 * | X | t | 0 | 0 | t | X |<- iy2 = 2
 * +---+---+---+---+---+---+
 * | X | 0 | 0 | 0 | 0 | X |
 * +---+---+---+---+---+---+
 * | X | X | X | X | X | X |
 * +---+---+---+---+---+---+
 *     ^----- n -------^
 * ^------- ext_n ---------^ 
 * 
 * N.B. indexing provided let cells marked with X in the diagrams untouched, so 
 *      that they can be filled with ghost cells at later stages.
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side of the smaller square grid; i.e., the original domain grid 
 * wherein to write box.
 * @param ext_n side of the square grid containing the smaller one.
 * @param ix1 column index, with respect to smaller grid, of top-left corner of 
 * the box. 
 * @param iy1 row index, with respect to smaller grid, of top-left corner of 
 * the box. 
 * @param ix2 column index, with respect to smaller grid, of the corner 
 * opposite to the one identified by x1 and y1.
 * @param iy2 row index, with respect to smaller grid, of the corner opposite 
 * to the one identified by previous arguments x1 and y1.
 * @param t value of the cells belonging to the box.
 * @param subgrid_top_row_index row index, with respect to extended grid, of the
 * top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended grid,
 * of the top-left cell of smaller square grid.
 */
void box_in_subgrid_i(cell_t *grid,
                      int N,
                      int ext_n,
                      int ix1,
                      int iy1,
                      int ix2,
                      int iy2,
                      cell_value_t t,
                      int subgrid_top_row_index,
                      int subgrid_left_column_index)
{
    int i, j;
    for (i = iy1; i <= iy2; i++)
    {
        for (j = ix1; j <= ix2; j++)
        {
            point_in_subgrid_i(grid,
                               N,
                               ext_n,
                               j,
                               i,
                               t,
                               subgrid_top_row_index,
                               subgrid_left_column_index);
        }
    }
}

/**
 * @brief Draws a box in a grid while leaving space for halo at boundaries.
 * 1. Maps each of the coordinates of the box with opposite corners (`x1`, 
 *    `y1`) and (`x2`, `y2`) w.r.t. the square domain `grid` with top-left 
 *    corner (0, 0) and bottom-right corner (1, 1) against the interval 0-`N`
 *    and 
 * 2. draws a box using the resulting coordinates against the square subgrid 
 *    with top-left cell at pos grid(`subgrid_top_row_index`, 
 *    `subgrid_left_column_index`) and side length N.
 * Indexes exceeding subgrid boundaries are wrapt-around, so cells between 
 * subgrid boundaries and outer grid are kept untouched anyway.
 * Useful for initializing domain while leaving room for ghost cells needed 
 * for later processing, that are kept here untouched and filled at a later
 * stage.
 * It refactors box-creating behaviour common to program versions working with
 * actual domain (mpi-sendrecv-hpp, mpi-isend-hpp) along with that using
 * ghost-cells extended one (omp-hpp), allowing to directly write to the 
 * extended grid instead of: (1)allocating and  (2)initializing exact-sized 
 * domain, (3)reallocating a bigger grid with room for ghost cells and 
 * (4)copying boundary values in four distinct moments.
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual
 * domain grid wherein to write box.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param x1 x coordinate, in the range 0-1, of a corner of the box. 
 * @param y1 y coordinate, in the range 0-1, of a corner of the box. 
 * @param x2 x coordinate, in the range 0-1, of the corner of the box opposite 
 * to the one identified by x1 and y1.
 * @param y2 y coordinate, in the range 0-1, of the corner of the box opposite
 * to the one identified by x1 and y1.
 * @param t value to assign to the cells belonging to the box.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void box_in_subgrid(cell_t *grid,
                    int N,
                    int ext_n,
                    float x1,
                    float y1,
                    float x2,
                    float y2,
                    cell_value_t t,
                    int subgrid_top_row_index,
                    int subgrid_left_column_index)
{
    const int ix1 = remap_to_N(fminf(x1, x2), N);
    const int ix2 = remap_to_N(fmaxf(x1, x2), N);
    const int iy1 = remap_to_N(fminf(y1, y2), N);
    const int iy2 = remap_to_N(fmaxf(y1, y2), N);

    box_in_subgrid_i(grid,
                     N,
                     ext_n,
                     ix1,
                     N - 1 - iy2,
                     ix2,
                     N - 1 - iy1,
                     t,
                     subgrid_top_row_index,
                     subgrid_left_column_index);
}

/**
 * @brief Draws a circle in a `grid` by filling with `t` the only cells whose
 * distance is less or equal to radius from the cell (centre) at position 
 * (`ix1`, `iy1`) with respect to the square subgrid with top-left cell at 
 * position (`subgrid_top_row_index`, `subgrid_left_column_index`).
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual
 * domain grid wherein to draw the circle.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param circle_center_x column index, with respect to smaller grid, of circle 
 * center. 
 * @param circle_center_y row index, with respect to smaller grid, of circle
 * center.
 * @param circle_radius circle radius, the maximum number of cells from the 
 * center to every other cell belonging to the circle (boundaries included)
 * @param t value to assign to the cells belonging to the circle area.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void circle_in_subgrid_i(cell_t *grid,
                         int N,
                         int ext_n,
                         int circle_center_x,
                         int circle_center_y,
                         int circle_radius,
                         cell_value_t t,
                         int subgrid_top_row_index,
                         int subgrid_left_column_index)
{
    int dx, dy;

    for (dy = -circle_radius; dy <= circle_radius; dy++)
    {
        for (dx = -circle_radius; dx <= circle_radius; dx++)
        {

            if (dx * dx + dy * dy <= circle_radius * circle_radius)
            {
                point_in_subgrid_i(grid,
                                   N,
                                   ext_n,
                                   circle_center_x + dx,
                                   N - 1 - circle_center_y - dy,
                                   t,
                                   subgrid_top_row_index,
                                   subgrid_left_column_index);
            }
        }
    }
}

/**
 * @brief Draw a circle in a `grid` while leaving space for halo at boundaries.
 * 1. Maps the coordinates of the circle with center in (`x`, `y`) and radius `r` 
 *    w.r.t. the square domain grid with top-left corner (0, 0) and 
 *    bottom-right corner (1, 1) against the interval 0-`N` and
 * 2. draws a circle into grid using the resulting coordinates against the 
 *    square subgrid with top-left cell at pos grid(`subgrid_top_row_index`, 
 *    `subgrid_left_column_index`) and side size N.
 * Indexes exceeding subgrid boundaries are wrapt-around, so cells between 
 * subgrid boundaries and outer grid are kept untouched, anyway.
 * Useful for initializing domain while leaving room for ghost cells needed
 * for later processing, that are kept here untouched and filled at a later
 * stage.
 * It refactors circle-creating behaviour common to program versions working 
 * with actual domain (mpi-sendrecv-hpp, mpi-isend-hpp) along with that using
 * ghost-cells extended one (omp-hpp), allowing to directly write to the 
 * extended grid instead of: (1)allocating and (2)initializing exact-sized 
 * domain, (3)reallocating a bigger grid with room for ghost cells and 
 * (4)copying boundary values in four distinct moments.
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual 
 * domain grid wherein to draw the circle.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param x x coordinate, in the range 0-1, of the center of the circle.
 * @param y y coordinate, in the range 0-1, of the center of the circle. 
 * @param r radius of the circle to draw. 
 * @param t value to assign to the cells belonging to the circle area.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void circle_in_subgrid(cell_t *grid,
                       int N,
                       int ext_n,
                       float x,
                       float y,
                       float r,
                       cell_value_t t,
                       int subgrid_top_row_index,
                       int subgrid_left_column_index)
{
    const int circle_center_x = remap_to_N(x, N);
    const int circle_center_y = remap_to_N(y, N);
    const int circle_radius = remap_to_N(r, N);

    circle_in_subgrid_i(grid,
                        N,
                        ext_n,
                        circle_center_x,
                        circle_center_y,
                        circle_radius,
                        t,
                        subgrid_top_row_index,
                        subgrid_left_column_index);
}

/**
 * @brief Fills the cells of the area with top-left cell at position (`ix1`,
 * `iy1`) and bottom-right cell at position (`ix2`, `iy2`) with respect to the
 * square subgrid with top-left cell at position (`subgrid_top_row_index`, 
 * `subgrid_left_column_index`), instead of (0,0), and side length `N`.
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual
 *  domain grid wherein to draw the circle.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param ix1 column index, with respect to smaller grid, of top-left corner of 
 * the area to be filled. 
 * @param iy1 row index, with respect to smaller grid, of a corner of the 
 * area to be filled. 
 * @param ix2 column index, with respect to smaller grid, of the corner of the 
 * area to be filled opposite to the one identified by x1 and y1.
 * @param iy2 row index, with respect to smaller grid, of the corner of the 
 * area to be filled opposite to the one identified by x1 and y1.
 * @param p the probability, for every cell involved, to be filled with a GAS
 * molecule.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void random_fill_in_subgrid_i(cell_t *grid,
                              int N,
                              int ext_n,
                              int ix1,
                              int iy1,
                              int ix2,
                              int iy2,
                              float p,
                              int subgrid_top_row_index,
                              int subgrid_left_column_index)
{
    int i, j;

    for (i = iy1; i <= iy2; i++)
    {
        for (j = ix1; j <= ix2; j++)
        {
            const int ij =
                get_ext_grid_coords_from_subgrid_coords(N,
                                                        ext_n,
                                                        j,
                                                        N - 1 - i,
                                                        subgrid_top_row_index,
                                                        subgrid_left_column_index);
            if (grid[ij] == EMPTY && ((float)rand()) / RAND_MAX < p)
            {
                point_in_subgrid_i(grid,
                                   N,
                                   ext_n, j,
                                   N - 1 - i,
                                   GAS,
                                   subgrid_top_row_index,
                                   subgrid_left_column_index);
            }
        }
    }
}

/**
 * @brief Random fills a grid leaving space for halo at boundaries.
 * 1. Maps each of the coordinates of the area to be filled with opposite 
 *    corners (`x1`, `y1`) and (`x2`, `y2`) w.r.t a square domain `grid` with
 *    top-left corner (0, 0) and bottom-right corner (1, 1) against the 
 *    interval 0-`N` and 
 * 2. fills the cells of the area using the resulting coordinates against the 
 *    square subgrid with top-left cell at pos grid(`subgrid_top_row_index`, 
 *    `subgrid_left_column_index`) and side length N.
 * Indexes exceeding subgrid boundaries are wrapt-around, so cells between
 * subgrid boundaries and outer grid are kept untouched anyway.
 * Useful for initializing domain while leaving room for ghost cells needed 
 * for later processing, that are kept here untouched and filled at a later 
 * stage.
 * It refactors filling behaviour common to program versions working with
 * actual domain (mpi-sendrecv-hpp, mpi-isend-hpp) along with that using 
 * ghost-cells extended one (omp-hpp), allowing to directly write to the 
 * extended grid instead of: (1)allocating and (2)initializing exact-sized 
 * domain, (3)reallocating a bigger grid with room for ghost cells and 
 * (4)copying boundary values in four distinct moments.
 * 
 * 
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual
 * domain grid wherein to draw the box.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param x1 x coordinate, in the range 0-1, of a corner of the area to be 
 * filled. 
 * @param y1 y coordinate, in the range 0-1, of a corner of the area to be 
 * filled.
 * @param x2 x coordinate, in the range 0-1, of the corner of the box opposite
 * to the one identified by x1 and y1.
 * @param y2 y coordinate, in the range 0-1, of the corner of the box 
 * identified by x1 and y1.
 * @param p the probability, for every cell involved, to be filled with a GAS
 * molecule.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void random_fill_in_subgrid(cell_t *grid,
                            int N,
                            int ext_n,
                            float x1,
                            float y1,
                            float x2,
                            float y2,
                            float p,
                            int subgrid_top_row_index,
                            int subgrid_left_col_index)
{
    const int ix1 = remap_to_N(fminf(x1, x2), N);
    const int ix2 = remap_to_N(fmaxf(x1, x2), N);
    const int iy1 = remap_to_N(fminf(y1, y1), N);
    const int iy2 = remap_to_N(fmaxf(y1, y2), N);

    random_fill_in_subgrid_i(grid,
                             N,
                             ext_n,
                             ix1,
                             iy1,
                             ix2,
                             iy2,
                             p,
                             subgrid_top_row_index,
                             subgrid_left_col_index);
}

/**
 * @brief Initializes HPP domain `grid` according to file specification written 
 * using the scene description language below, leaving halo untouched.
 * Used for initializing domain while leaving room for ghost cells needed for 
 * later processing, that are kept here untouched and filled at a later stage.

 * ## Scene description language
 *
 * All cells of the domain are initially EMPTY. All coordinates are
 * real numbers in [0, 1]; they:
 * 1. refer to cartesian coordinated system (y=0 is at the bottom)
 * 2. are automatically scaled to the resolution `N` used for the image.
 * 3. are mapped against the grid with top-left cell at position 
 *    (`subgrid_top_row_index`, and `subgrid_left_column_index`) for leaving 
 *    space for halo at boundaries.
 *
 * c x y r t
 *
 *   Draws a circle centered ad (x, y) with radius r filled with
 *   particles of type t (0=WALL, 1=GAS, 2=EMPTY)
 *
 *
 * b x1 y1 x2 y2 t
 *
 *   Draws a rectangle with opposite corners (x1,y1) and (x2,y2) filled
 *   with particles of type t (0=WALL, 1=GAS, 2=EMPTY)
 *
 *
 * r x1 y1 x2 y2 p
 *
 *   Fills the rectangle with opposite corners (x1,y1), (x2,y2) with
 *   GAS particles with probability p \in [0, 1]. Only EMPTY cells
 *   might be filled with gas particles, everything else is not
 *   modified.
 * 
 * @param filein pointer to file containing the starting scene description.
 * @param grid extended square grid containing the smaller square grid.
 * @param N side length of the smaller square grid; i.e., the original, actual 
 * domain grid to initialize.
 * @param ext_n side length of the square grid containing the smaller one.
 * @param subgrid_top_row_index row index, with respect to extended grid, of 
 * the top-left cell of smaller square grid.
 * @param subgrid_left_column_index column index, with respect to extended 
 * grid, of the top-left cell of smaller square grid.
 */
void read_problem_in_subgrid(FILE *filein,
                             cell_t *grid,
                             int N,
                             int ext_n,
                             int subgrid_top_row_index,
                             int subgrid_left_column_index)
{
    /* Since used at inizialization phase only, this procedure has not been
       parallelized. */
    int i, j;
    int nread;
    char op;

    const int TOP = subgrid_left_column_index;
    const int BOTTOM = subgrid_left_column_index + N - 1;
    const int LEFT = subgrid_top_row_index;
    const int RIGHT = subgrid_left_column_index + N - 1;

    for (i = TOP; i <= BOTTOM; i++)
    {
        for (j = LEFT; j <= RIGHT; j++)
        {
            grid[IDX_mod(i, j, ext_n)] = EMPTY;
        }
    }

    while ((nread = fscanf(filein, " %c", &op)) == 1)
    {
        int t;
        float x1, y1, x2, y2, r, p;
        int retval;

        switch (op)
        {
        case 'c': /* circle */
            retval = fscanf(filein, "%f %f %f %d", &x1, &y1, &r, &t);
            assert(retval == 4);
            circle_in_subgrid(grid,
                              N,
                              ext_n,
                              x1,
                              y1,
                              r,
                              t,
                              subgrid_top_row_index,
                              subgrid_left_column_index);
            break;
        case 'b': /* box */
            retval = fscanf(filein, "%f %f %f %f %d", &x1, &y1, &x2, &y2, &t);
            assert(retval == 5);
            box_in_subgrid(grid,
                           N,
                           ext_n,
                           x1,
                           y1,
                           x2,
                           y2,
                           t,
                           subgrid_top_row_index,
                           subgrid_left_column_index);
            break;
        case 'r': /* random_fill */
            retval = fscanf(filein, "%f %f %f %f %f", &x1, &y1, &x2, &y2, &p);
            assert(retval == 5);
            random_fill_in_subgrid(grid,
                                   N,
                                   ext_n,
                                   x1,
                                   y1,
                                   x2,
                                   y2,
                                   p,
                                   subgrid_top_row_index,
                                   subgrid_left_column_index);
            break;
        default:
            fprintf(stderr, "FATAL: Unrecognized command `%c`\n", op);
            exit(EXIT_FAILURE);
        }
    }
}

/***************************************************************************
 *************************** HPP COMPUTATION CODE **************************
 **************************************************************************/

/**
 * @brief Swaps the content of cells `a` and `b`, provided that neither is 
 * a WALL; otherwise, does nothing. 
 * 
 * @param a pointer to cell to be possibly swapped with that pointed by b
 * @param b pointer to cell to be possibly swapped with that pointed by a
 */
void swap_cells(cell_t *a, cell_t *b)
{
    if ((*a != WALL) && (*b != WALL))
    {
        const cell_t tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

/**
 * @brief Given the current configuration of the HPP cellular automaton (CA) in
 * `cur`, computes the updated values for the single block of size 
 * 2 x 2 starting at position cur[i, j] (actually: cur[`N` * i + j]), and 
 * writes them to `next`, keeping grid pointed by `cur` unchanged.
 * It contains domain-update logic for both odd and even phases of a step by 
 * focusing on two different neighboorhood at each of them, according to HPP 
 * specification, like:
 * 
 * @pre
 *  phase == EVEN_PHASE                       phase == ODD_PHASE
 *
 * +---+---+---+---+---+                     +---+---+---+---+---+
 * |   |   |   |   |   | <- TOP_GHOST        | d | c | d | c |   | <--- TOP_GHOST
 * +---+---+---+---+---+                     +---+---+---+---+---+ 
 * |   | a | b | a | b | <- TOP              | b | a | b | a |   | <--- TOP
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | c | d | c | d |                     | d | c | d | c |   |     
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | a | b | a | b |                     | b | a | b | a |   |     
 * +---+---+---+---+---+                     +---+---+---+---+---+     
 * |   | c | d | c | d | <- BOTTOM           |   |   |   |   |   | <-- BOTTOM    
 * +---+---+---+---+---+                     +---+---+---+---+---+
 *       ^----LEFT                                 ^----LEFT
 *   ^----- LEFT_GHOST                         ^----- LEFT_GHOST
 * 
 * Note that ghost cells are needed for odd phase only.
 * 
 * @param cur pointer to the grid containing the current CA configuration. 
 * Content of memory pointed by `cur` is not edited nor it is re-assigned 
 * to new location.
 * @param next pointer to the grid that will contain, at the end of the call,
 * the updated CA configuration (of a single block).
 * @param N side length of domain square grid. When function is called 
 * passing grid extended with ghost cells, N must state extended side 
 * length.
 * @param phase phase (odd or even) of the step to compute.
 * @param i row index of the top-left (in case of even phase) or bottom-right 
 * (in case of odd) cell of the 2x2 block to update.
 * @param j column index of the top-left (in case of even phase) or 
 * bottom-right (in case of odd) cell of the 2x2 block to update. 
 */
void step_block(const cell_t *cur,
                cell_t *next,
                int N,
                phase_t phase,
                int i,
                int j)
{
    /**
             * If phase==EVEN_PHASE:
             * ab
             * cd
             *
             * If phase==ODD_PHASE:
             * dc
             * ba
             */
    const int a = IDX(i, j, N);
    const int b = IDX(i, j + phase, N);
    const int c = IDX(i + phase, j, N);
    const int d = IDX(i + phase, j + phase, N);
    next[a] = cur[a];
    next[b] = cur[b];
    next[c] = cur[c];
    next[d] = cur[d];
    if ((((next[a] == EMPTY) != (next[b] == EMPTY)) &&
         ((next[c] == EMPTY) != (next[d] == EMPTY))) ||
        (next[a] == WALL) || (next[b] == WALL) ||
        (next[c] == WALL) || (next[d] == WALL))
    {
        /* swap_cells hides handling of walls too. */
        swap_cells(&next[a], &next[b]);
        swap_cells(&next[c], &next[d]);
    }
    else
    {
        swap_cells(&next[a], &next[d]);
        swap_cells(&next[b], &next[c]);
    }
}

/***************************************************************************
 ************************* OUTPUT-GENERATING CODE **************************
 **************************************************************************/

/**
 * @brief Writes an image of `grid` to a file in PGM (Portable Graymap)
   format.
 * 
 * @param grid grid to write to file. It has N x N cells.
 * @param N side length of grid.
 * @param frameno the time step number, used for labeling the output file. 
 */
void write_image(const cell_t *grid, int N, int frameno)
{
    FILE *f;
    char fname[128];

    snprintf(fname, sizeof(fname), "hpp%05d.pgm", frameno);
    if ((f = fopen(fname, "w")) == NULL)
    {
        printf("Cannot open \"%s\" for writing\n", fname);
        abort();
    }
    fprintf(f, "P5\n");
    fprintf(f, "# produced by hpp\n");
    fprintf(f, "%d %d\n", N, N);
    fprintf(f, "%d\n", EMPTY); /* highest shade of grey (0=black). */
    fwrite(grid, 1, N * N, f);
    fclose(f);
}

#endif // COMMONS_H
