#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdatomic.h>

// Configurable puzzle size (must be perfect square)
int SIZE = 9;  // Default 9x9
int BLOCK_SIZE = 3;  // sqrt(SIZE)
int FULL_MASK;  // Will be set based on SIZE

// Shared solution and flag
atomic_int solution_found = 0;
int **solution_grid;
omp_lock_t solution_lock;

// Performance counters
atomic_int tasks_created = 0;
atomic_int branches_explored = 0;
atomic_int dead_ends = 0;

// Difficulty levels
typedef enum {
    EASY,
    MEDIUM,
    HARD,
    EXPERT
} Difficulty;

// Allocate 2D array
int** allocate_grid() {
    int **grid = (int**)malloc(SIZE * sizeof(int*));
    for (int i = 0; i < SIZE; i++) {
        grid[i] = (int*)calloc(SIZE, sizeof(int));
    }
    return grid;
}

// Free 2D array
void free_grid(int **grid) {
    for (int i = 0; i < SIZE; i++) {
        free(grid[i]);
    }
    free(grid);
}

// Copy grid
void copy_grid(int **dest, int **src) {
    for (int i = 0; i < SIZE; i++) {
        memcpy(dest[i], src[i], SIZE * sizeof(int));
    }
}

// Print grid
void print_grid(int **grid) {
    for (int r = 0; r < SIZE; r++) {
        if (r % BLOCK_SIZE == 0 && r != 0) {
            for (int i = 0; i < SIZE + BLOCK_SIZE - 1; i++) {
                printf("-");
            }
            printf("\n");
        }
        for (int c = 0; c < SIZE; c++) {
            if (c % BLOCK_SIZE == 0 && c != 0) {
                printf("| ");
            }
            printf("%d ", grid[r][c]);
        }
        printf("\n");
    }
    printf("\n");
}

// Count empty cells
int count_empty_cells(int **grid) {
    int count = 0;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] == 0) count++;
        }
    }
    return count;
}

// Check if the entire grid is valid
int is_valid_solution(int **grid) {
    // Check if grid is completely filled
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] < 1 || grid[r][c] > SIZE) {
                return 0;
            }
        }
    }
    
    // Allocate seen array once
    int *seen = (int*)calloc(SIZE + 1, sizeof(int));
    
    // Check rows
    for (int r = 0; r < SIZE; r++) {
        // Clear seen array
        for (int i = 0; i <= SIZE; i++) seen[i] = 0;
        
        for (int c = 0; c < SIZE; c++) {
            int val = grid[r][c];
            if (seen[val]) {
                free(seen);
                return 0;
            }
            seen[val] = 1;
        }
    }
    
    // Check columns
    for (int c = 0; c < SIZE; c++) {
        // Clear seen array
        for (int i = 0; i <= SIZE; i++) seen[i] = 0;
        
        for (int r = 0; r < SIZE; r++) {
            int val = grid[r][c];
            if (seen[val]) {
                free(seen);
                return 0;
            }
            seen[val] = 1;
        }
    }
    
    // Check blocks
    for (int bi = 0; bi < SIZE; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < SIZE; bj += BLOCK_SIZE) {
            // Clear seen array
            for (int i = 0; i <= SIZE; i++) seen[i] = 0;
            
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    int val = grid[bi + i][bj + j];
                    if (seen[val]) {
                        free(seen);
                        return 0;
                    }
                    seen[val] = 1;
                }
            }
        }
    }
    
    free(seen);
    return 1;
}

// Parallel solver with fork-join strategy
void solve_parallel_forkjoin(int **grid, int *rowMask, int *colMask, int *blockMask, 
                            int depth, int max_parallel_depth) {
    if (atomic_load(&solution_found)) return;
    
    atomic_fetch_add(&branches_explored, 1);
    
    // Find the blank with fewest candidates
    int min_count = SIZE + 1, br = -1, bc = -1;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] == 0) {
                int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                int mask = rowMask[r] & colMask[c] & blockMask[bi];
                int count = __builtin_popcount(mask);
                if (count == 0) {
                    atomic_fetch_add(&dead_ends, 1);
                    return;  // dead end
                }
                if (count < min_count) {
                    min_count = count;
                    br = r; bc = c;
                }
            }
        }
    }
    
    // No blanks => we've got a solution
    if (br < 0) {
        omp_set_lock(&solution_lock);
        if (!atomic_load(&solution_found)) {
            copy_grid(solution_grid, grid);
            atomic_store(&solution_found, 1);
        }
        omp_unset_lock(&solution_lock);
        return;
    }
    
    int bi = (br/BLOCK_SIZE)*BLOCK_SIZE + (bc/BLOCK_SIZE);
    int mask = rowMask[br] & colMask[bc] & blockMask[bi];
    
    // Count candidates
    int candidates[SIZE];
    int num_candidates = 0;
    while (mask) {
        int bit = mask & -mask;
        mask -= bit;
        candidates[num_candidates++] = bit;
    }
    
    // Decide whether to parallelize based on depth and workload
    int should_parallelize = (depth < max_parallel_depth && 
                             num_candidates > 1 && 
                             !atomic_load(&solution_found));
    
    if (should_parallelize) {
        atomic_fetch_add(&tasks_created, num_candidates);
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_candidates; i++) {
            if (atomic_load(&solution_found)) continue;
            
            int bit = candidates[i];
            int num = __builtin_ctz(bit) + 1;
            
            // Create thread-local copies
            int **local_grid = allocate_grid();
            int *local_rowMask = (int*)malloc(SIZE * sizeof(int));
            int *local_colMask = (int*)malloc(SIZE * sizeof(int));
            int *local_blockMask = (int*)malloc(SIZE * sizeof(int));
            
            copy_grid(local_grid, grid);
            memcpy(local_rowMask, rowMask, SIZE * sizeof(int));
            memcpy(local_colMask, colMask, SIZE * sizeof(int));
            memcpy(local_blockMask, blockMask, SIZE * sizeof(int));
            
            // Place the number
            local_grid[br][bc] = num;
            local_rowMask[br] ^= bit;
            local_colMask[bc] ^= bit;
            local_blockMask[bi] ^= bit;
            
            // Recursive solve
            solve_parallel_forkjoin(local_grid, local_rowMask, local_colMask, 
                                   local_blockMask, depth + 1, max_parallel_depth);
            
            // Clean up
            free_grid(local_grid);
            free(local_rowMask);
            free(local_colMask);
            free(local_blockMask);
        }
    } else {
        // Sequential execution for deeper levels
        for (int i = 0; i < num_candidates; i++) {
            if (atomic_load(&solution_found)) return;
            
            int bit = candidates[i];
            int num = __builtin_ctz(bit) + 1;
            
            // Place
            grid[br][bc] = num;
            rowMask[br] ^= bit;
            colMask[bc] ^= bit;
            blockMask[bi] ^= bit;
            
            solve_parallel_forkjoin(grid, rowMask, colMask, blockMask, 
                                   depth + 1, max_parallel_depth);
            
            // Undo
            blockMask[bi] ^= bit;
            colMask[bc] ^= bit;
            rowMask[br] ^= bit;
            grid[br][bc] = 0;
        }
    }
}

// Main parallel solver with performance tuning
int solve_puzzle_parallel_optimized(int **puzzle, int **solution, int num_threads) {
    int **grid = allocate_grid();
    copy_grid(grid, puzzle);
    
    int *rowMask = (int*)malloc(SIZE * sizeof(int));
    int *colMask = (int*)malloc(SIZE * sizeof(int));
    int *blockMask = (int*)malloc(SIZE * sizeof(int));
    
    // Initialize masks
    for (int i = 0; i < SIZE; i++) {
        rowMask[i] = colMask[i] = blockMask[i] = FULL_MASK;
    }
    
    // Update masks based on existing numbers
    int empty_cells = 0;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] != 0) {
                int bit = 1 << (grid[r][c] - 1);
                int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                rowMask[r] &= ~bit;
                colMask[c] &= ~bit;
                blockMask[bi] &= ~bit;
            } else {
                empty_cells++;
            }
        }
    }
    
    // Reset counters
    atomic_store(&solution_found, 0);
    atomic_store(&tasks_created, 0);
    atomic_store(&branches_explored, 0);
    atomic_store(&dead_ends, 0);
    
    omp_init_lock(&solution_lock);
    
    // Adjust parallel depth based on problem size
    int max_parallel_depth = 2 + (empty_cells > 40 ? 1 : 0);
    
    printf("Starting parallel solve with %d threads, max parallel depth: %d\n", 
           num_threads, max_parallel_depth);
    printf("Empty cells: %d\n", empty_cells);
    
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    solve_parallel_forkjoin(grid, rowMask, colMask, blockMask, 0, max_parallel_depth);
    
    double end_time = omp_get_wtime();
    
    printf("\nPerformance statistics:\n");
    printf("Time: %.4f seconds\n", end_time - start_time);
    printf("Tasks created: %d\n", atomic_load(&tasks_created));
    printf("Branches explored: %d\n", atomic_load(&branches_explored));
    printf("Dead ends: %d\n", atomic_load(&dead_ends));
    
    omp_destroy_lock(&solution_lock);
    
    if (atomic_load(&solution_found)) {
        copy_grid(solution, solution_grid);
    }
    
    free_grid(grid);
    free(rowMask);
    free(colMask);
    free(blockMask);
    
    return atomic_load(&solution_found);
}

// Sequential solver for comparison
int solveSequential(int **grid, int *rowMask, int *colMask, int *blockMask) {
    // find the blank with fewest candidates
    int min_count = SIZE + 1, br = -1, bc = -1;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] == 0) {
                int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                int mask = rowMask[r] & colMask[c] & blockMask[bi];
                int count = __builtin_popcount(mask);
                if (count == 0) return 0;       // dead end
                if (count < min_count) {
                    min_count = count;
                    br = r; bc = c;
                }
            }
        }
    }

    // no blanks ⇒ we've got a solution
    if (br < 0) {
        return 1;
    }

    int bi = (br/BLOCK_SIZE)*BLOCK_SIZE + (bc/BLOCK_SIZE);
    int mask = rowMask[br] & colMask[bc] & blockMask[bi];
    while (mask) {
        int bit = mask & -mask;
        mask -= bit;
        int num = __builtin_ctz(bit) + 1;

        // place
        grid[br][bc] = num;
        rowMask[br]   ^= bit;
        colMask[bc]   ^= bit;
        blockMask[bi] ^= bit;

        if (solveSequential(grid, rowMask, colMask, blockMask))
            return 1;

        // undo
        blockMask[bi] ^= bit;
        colMask[bc]   ^= bit;
        rowMask[br]   ^= bit;
        grid[br][bc]  = 0;
    }
    return 0;
}

// Generate a complete valid sudoku grid
void generate_complete_grid(int **grid) {
    // Clear the grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            grid[i][j] = 0;
        }
    }
    
    // Fill diagonal blocks first (they don't conflict with each other)
    for (int i = 0; i < SIZE; i += BLOCK_SIZE) {
        // Create array of numbers 1 to SIZE
        int *nums = (int*)malloc(SIZE * sizeof(int));
        for (int k = 0; k < SIZE; k++) {
            nums[k] = k + 1;
        }
        
        // Shuffle the numbers
        for (int k = SIZE - 1; k > 0; k--) {
            int j = rand() % (k + 1);
            int temp = nums[k];
            nums[k] = nums[j];
            nums[j] = temp;
        }
        
        // Fill the diagonal block
        int idx = 0;
        for (int row = i; row < i + BLOCK_SIZE; row++) {
            for (int col = i; col < i + BLOCK_SIZE; col++) {
                grid[row][col] = nums[idx++];
            }
        }
        
        free(nums);
    }
    
    // Now solve the rest using our solver
    int *rowMask = (int*)malloc(SIZE * sizeof(int));
    int *colMask = (int*)malloc(SIZE * sizeof(int));
    int *blockMask = (int*)malloc(SIZE * sizeof(int));
    
    // Initialize masks
    for (int i = 0; i < SIZE; i++) {
        rowMask[i] = colMask[i] = blockMask[i] = FULL_MASK;
    }
    
    // Update masks based on the diagonal blocks we filled
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] != 0) {
                int bit = 1 << (grid[r][c] - 1);
                int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                rowMask[r] &= ~bit;
                colMask[c] &= ~bit;
                blockMask[bi] &= ~bit;
            }
        }
    }
    
    solveSequential(grid, rowMask, colMask, blockMask);
    
    free(rowMask);
    free(colMask);
    free(blockMask);
}

// Generate puzzle by removing numbers
void generate_puzzle(int **grid, Difficulty difficulty) {
    int cells_to_remove;
    
    // Determine how many cells to remove based on difficulty
    switch (difficulty) {
        case EASY:
            cells_to_remove = SIZE * SIZE * 0.4;
            break;
        case MEDIUM:
            cells_to_remove = SIZE * SIZE * 0.5;
            break;
        case HARD:
            cells_to_remove = SIZE * SIZE * 0.6;
            break;
        case EXPERT:
            cells_to_remove = SIZE * SIZE * 0.7;
            break;
    }
    
    int removed = 0;
    int attempts = 0;
    int max_attempts = cells_to_remove * 3;
    
    while (removed < cells_to_remove && attempts < max_attempts) {
        int row = rand() % SIZE;
        int col = rand() % SIZE;
        
        if (grid[row][col] != 0) {
            int backup = grid[row][col];
            grid[row][col] = 0;
            
            // Create a copy to test if puzzle has unique solution
            int **test_grid = allocate_grid();
            copy_grid(test_grid, grid);
            
            // Try to solve the puzzle
            int *rowMask = (int*)malloc(SIZE * sizeof(int));
            int *colMask = (int*)malloc(SIZE * sizeof(int));
            int *blockMask = (int*)malloc(SIZE * sizeof(int));
            
            for (int i = 0; i < SIZE; i++) {
                rowMask[i] = colMask[i] = blockMask[i] = FULL_MASK;
            }
            
            for (int r = 0; r < SIZE; r++) {
                for (int c = 0; c < SIZE; c++) {
                    if (test_grid[r][c] != 0) {
                        int bit = 1 << (test_grid[r][c] - 1);
                        int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                        rowMask[r] &= ~bit;
                        colMask[c] &= ~bit;
                        blockMask[bi] &= ~bit;
                    }
                }
            }
            
            if (solveSequential(test_grid, rowMask, colMask, blockMask)) {
                removed++;
            } else {
                // Restore the number if no unique solution
                grid[row][col] = backup;
            }
            
            free_grid(test_grid);
            free(rowMask);
            free(colMask);
            free(blockMask);
        }
        
        attempts++;
    }
}

// Sequential solver for timing comparison
double solve_sequential_timed(int **puzzle) {
    int **grid = allocate_grid();
    copy_grid(grid, puzzle);
    
    int *rowMask = (int*)malloc(SIZE * sizeof(int));
    int *colMask = (int*)malloc(SIZE * sizeof(int));
    int *blockMask = (int*)malloc(SIZE * sizeof(int));
    
    // Initialize masks
    for (int i = 0; i < SIZE; i++) {
        rowMask[i] = colMask[i] = blockMask[i] = FULL_MASK;
    }
    
    // Update masks based on existing numbers
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (grid[r][c] != 0) {
                int bit = 1 << (grid[r][c] - 1);
                int bi = (r/BLOCK_SIZE)*BLOCK_SIZE + (c/BLOCK_SIZE);
                rowMask[r] &= ~bit;
                colMask[c] &= ~bit;
                blockMask[bi] &= ~bit;
            }
        }
    }
    
    double start_time = omp_get_wtime();
    solveSequential(grid, rowMask, colMask, blockMask);
    double end_time = omp_get_wtime();
    
    free_grid(grid);
    free(rowMask);
    free(colMask);
    free(blockMask);
    
    return end_time - start_time;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s generate <size> <difficulty> <num_threads>\n", argv[0]);
        printf("  %s solve <size> <puzzle_file> <num_threads>\n", argv[0]);
        printf("\nSize must be a perfect square (4, 9, 16, 25...)\n");
        printf("Difficulty: 0=Easy, 1=Medium, 2=Hard, 3=Expert\n");
        return EXIT_FAILURE;
    }
    
    if (strcmp(argv[1], "generate") == 0) {
        if (argc != 5) {
            printf("Usage: %s generate <size> <difficulty> <num_threads>\n", argv[0]);
            return EXIT_FAILURE;
        }
        
        SIZE = atoi(argv[2]);
        BLOCK_SIZE = (int)sqrt(SIZE);
        
        if (BLOCK_SIZE * BLOCK_SIZE != SIZE) {
            fprintf(stderr, "Error: Size must be a perfect square\n");
            return EXIT_FAILURE;
        }
        
        FULL_MASK = (1 << SIZE) - 1;
        Difficulty diff = atoi(argv[3]);
        int num_threads = atoi(argv[4]);
        
        // Allocate grids
        int **complete_grid = allocate_grid();
        int **puzzle_grid = allocate_grid();
        solution_grid = allocate_grid();
        
        printf("Generating %dx%d sudoku puzzle...\n", SIZE, SIZE);
        
        // Generate complete grid
        generate_complete_grid(complete_grid);
        
        // Create puzzle from complete grid
        copy_grid(puzzle_grid, complete_grid);
        generate_puzzle(puzzle_grid, diff);
        
        printf("\nPuzzle:\n");
        print_grid(puzzle_grid);
        
        // First, solve sequentially for comparison
        printf("\nSolving sequentially for comparison...\n");
        double seq_time = solve_sequential_timed(puzzle_grid);
        printf("Sequential time: %.4f seconds\n", seq_time);
        
        // Solve the puzzle with parallel solver
        printf("\nSolving with parallel algorithm...\n");
        int **solved_grid = allocate_grid();
        
        int solved = solve_puzzle_parallel_optimized(puzzle_grid, solved_grid, num_threads);
        
        if (solved) {
            printf("\nSolution:\n");
            print_grid(solved_grid);
            
            // Verify solution
            if (is_valid_solution(solved_grid)) {
                printf("Solution is VALID!\n");
            } else {
                printf("Solution is INVALID!\n");
            }
        } else {
            printf("Failed to solve puzzle\n");
        }
        
        // Clean up
        free_grid(complete_grid);
        free_grid(puzzle_grid);
        free_grid(solved_grid);
        free_grid(solution_grid);
        
    } else if (strcmp(argv[1], "solve") == 0) {
        if (argc != 5) {
            printf("Usage: %s solve <size> <puzzle_file> <num_threads>\n", argv[0]);
            return EXIT_FAILURE;
        }
        
        SIZE = atoi(argv[2]);
        BLOCK_SIZE = (int)sqrt(SIZE);
        
        if (BLOCK_SIZE * BLOCK_SIZE != SIZE) {
            fprintf(stderr, "Error: Size must be a perfect square\n");
            return EXIT_FAILURE;
        }
        
        FULL_MASK = (1 << SIZE) - 1;
        char *filename = argv[3];
        int num_threads = atoi(argv[4]);
        
        FILE *file = fopen(filename, "r");
        if (!file) {
            perror("Error opening file");
            return EXIT_FAILURE;
        }
        
        int **puzzle = allocate_grid();
        
        // Read puzzle from file
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                if (fscanf(file, "%d", &puzzle[r][c]) != 1) {
                    fprintf(stderr, "Error reading puzzle from file\n");
                    fclose(file);
                    free_grid(puzzle);
                    return EXIT_FAILURE;
                }
            }
        }
        fclose(file);
        
        printf("Input puzzle:\n");
        print_grid(puzzle);
        
        // First, solve sequentially for comparison
        printf("\nSolving sequentially for comparison...\n");
        double seq_time = solve_sequential_timed(puzzle);
        printf("Sequential time: %.4f seconds\n", seq_time);
        
        // Solve the puzzle
        int **solution = allocate_grid();
        solution_grid = allocate_grid();
        
        int solved = solve_puzzle_parallel_optimized(puzzle, solution, num_threads);
        
        if (solved) {
            printf("\nSolution:\n");
            print_grid(solution);
            
            // Verify solution
            if (is_valid_solution(solution)) {
                printf("Solution is VALID!\n");
            } else {
                printf("Solution is INVALID!\n");
            }
        } else {
            printf("Failed to solve puzzle\n");
        }
        
        // Clean up
        free_grid(puzzle);
        free_grid(solution);
        free_grid(solution_grid);
    } else {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
