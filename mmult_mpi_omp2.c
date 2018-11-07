#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#define min(x, y) ((x)<(y)?(x):(y))

double* gen_matrix(int n, int m);
int mmult(double *c, double *a, int aRows, int aCols, double *b, int bRows, int bCols);
void compare_matrices(double *a, double *b, int nRows, int nCols);

/**
 Program to multiply a matrix times a matrix using both
 mpi to distribute the computation among nodes and omp
 to distribute the computation among threads.
 */

int main(int argc, char* argv[])
{
    int nrows, ncols;
    double *aa;    /* the A matrix */
    double *bb;    /* the B matrix */
    double *cc1;    /* A x B computed using the omp-mpi code you write */
    double *cc2;    /* A x B computed using the conventional algorithm */
    static double *buffer;
    int myid, master, numprocs;
    double starttime, endtime, ans;
    MPI_Status status;
    /* insert other global variables here */
    
    MPI_Request send_request,recv_request; //Fixing deadlocks
    
    int i, j, k, numsent, sender;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int anstype, row;
    if (argc > 1) {
        nrows = atoi(argv[1]);
        ncols = nrows;
        
        // Master Code goes here
        aa = (double*)malloc(sizeof(double) * nrows * ncols);
        bb = (double*)malloc(sizeof(double) * ncols * nrows );
        cc1 = (double*)malloc(sizeof(double) * nrows * nrows);
        cc2 = (double*)malloc(sizeof(double) * nrows * nrows );
        buffer = (double*)malloc(sizeof(double) * ncols * nrows * 2);
        
        aa = gen_matrix(nrows, ncols);
        bb = gen_matrix(ncols, nrows);
        
        
        for (int r = 0; r < nrows; ++r){
            printf("|");
            for ( int c = 0; c< ncols; ++c){
                printf(" %f", aa[r*ncols + c]);
            }
            printf("|\n");
        }
        
        
        master = 0;
        if (myid == master) {

            
            starttime = MPI_Wtime();
            /* Insert your master code here to store the product into cc1 */
            numsent = 0;
            
            
            //
            for (i = 0; i < nrows; i++) { // for each row of aa
                for (j = 0; j < ncols; j++){
                    buffer[j] = aa[i*ncols + j];
                }
                MPI_Bcast(buffer, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD); // SEND aa[row i] to all
                
                for (j = 0; j < numprocs; j++) { // for each column of bb
                    for (k = 0; k < nrows; k++){
                        buffer[k] = bb[j + k*nrows];
                    }
                    printf("Master is past before DL1 at %i %i\n", i, j);
                    MPI_Isend(buffer, nrows, MPI_DOUBLE, i*ncols + j, (i % numprocs)+1, MPI_COMM_WORLD, &send_request); //SEND bb[col j] to one slave (which? does this iterate past number of slaves?)
                    // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK 1^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    printf("Master is past DL1 at %i %i\n", i, j);
                    numsent++;
                }
                
                
                // recieve here? -- waits for all of row i to complete before moving to row i++
                //for (int r = 0; r < nrows; r++) { // this loop might not be necessary
                printf("Master has reached DL2 at i = %i\n", i);
                    MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                             MPI_COMM_WORLD, &status); // RECEIVE ans from slave
                    // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK 2^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    printf("Master is past DL2 at i = %i\n", i);
                    sender = status.MPI_SOURCE;
                    anstype = status.MPI_TAG;
                    cc1[anstype-1 + i*nrows] = ans; // cc1[r][anstype-1]
                    
                    if (numsent < nrows) {
                        for (j = 0; j < ncols; j++) {
                            buffer[j] = bb[numsent*nrows + j];
                        }
                        MPI_Send(buffer, ncols, MPI_DOUBLE, sender, numsent+1,
                                 MPI_COMM_WORLD);
                        numsent++;
                    } else {
                        MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD); // breaks inner slave loop
                    }
                //}
                
                
            }//
            
            
            
            endtime = MPI_Wtime();
            printf("%f\n",(endtime - starttime));
            cc2  = malloc(sizeof(double) * nrows * nrows);
            mmult(cc2, aa, nrows, ncols, bb, ncols, nrows);
            compare_matrices(cc2, cc1, nrows, nrows);
        } else {
            int brecs = 0;
            do{
                // Slave Code goes here
                //printf("Shogun's Decapitator + 132 Lords' heads =");
                MPI_Bcast(bb, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD); // RECIEVE aa[row c]
                ++brecs;
                
                while(1) { //inner slave loop
                    MPI_Recv(buffer, ncols, MPI_DOUBLE, master, MPI_ANY_TAG, // is this bb[col c]
                             MPI_COMM_WORLD, &status);
                    if (status.MPI_TAG == 0){
                        break; // breaks inner slave loop
                    }
                    row = status.MPI_TAG;
                    ans = 0.0;
                    for (j = 0; j < ncols; j++) {
                        //calculate answer (e.i. aa[c][j] . bb[j][c]) OR a[j] *b[j]
                        ans += buffer[j] * bb[j];
                    }
                    
                    printf("Process %i reaching DL3 at row = %i\n", myid, row);
                    MPI_Send(&ans, 1, MPI_DOUBLE, master, row, MPI_COMM_WORLD); // SEND ans to Master
                    // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK 3^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    //https://stackoverflow.com/questions/49598174/how-does-mpi-blocking-send-and-receive-work
                    printf("Process %i past DL3 at row = %i\n", myid, row);
                }
                
            } while(brecs < nrows);// broadcasts recieved is less than number rows
            
        }
    } else {
        fprintf(stderr, "Usage matrix_times_vector <size>\n");
    }
    MPI_Finalize();
    return 0;
}

/*temporarily storing methods here in attempt to resolve compiler error*/
/*int mmult(double *c,
          double *a, int aRows, int aCols,
          double *b, int bRows, int bCols) {
    int i, j, k;
    for (i = 0; i < aRows; i++) {
        for (j = 0; j < bCols; j++) {
            c[i*bCols + j] = 0;
        }
        for (k = 0; k < aCols; k++) {
            for (j = 0; j < bCols; j++) {
                c[i*bCols + j] += a[i*aCols + k] * b[k*bCols + j];
            }
        }
    }
    return 0;
}

double* gen_matrix(int n, int m) {
    int i, j;
    double *a = malloc(sizeof(double) * n * m);
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            a[i*m + j] = (double)rand()/RAND_MAX;
        }
    }
    return a;
}

void compare_matrices(double* a, double* b, int nRows, int nCols) {
    int n = nRows * nCols;
    int i, j, k;
    for (k = 0; k < n; ++k) {
        if (fabs(a[k]-b[k])/fabs(a[k]) > 1e-12) {
            i = k/nCols;
            j = k%nCols;
            printf("a[%d][%d] == %.12g\nb[%d][%d] == %.12g\ndelta == %.12g\nrelerr == %.12g\n",
                   i, j, a[k], i, j, b[k], fabs(a[k]-b[k]), fabs(a[k]-b[k])/fabs(a[k]));
            return;
        }
    }
    printf("Matrices are the same\n");
}
*/

