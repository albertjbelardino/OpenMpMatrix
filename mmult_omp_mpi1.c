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
  double *aa;	/* the A matrix */
  double *bb;	/* the B matrix */
  double *cc1;	/* A x B computed using the omp-mpi code you write */
  double *cc2;	/* A x B computed using the conventional algorithm */
  double *buffer;
  int myid, master, numprocs;
  double starttime, endtime, ans;
  MPI_Status status;
  /* insert other global variables here */
  int i, j, k, numsent, sender;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int anstype, row;
  if (argc > 1) {
    nrows = atoi(argv[1]);
    ncols = nrows;
    master = 0;
    if (myid == master) {
      // Master Code goes here
      aa = gen_matrix(nrows, ncols);
      bb = gen_matrix(ncols, nrows);
      cc1 = malloc(sizeof(double) * nrows * nrows); 
      buffer = malloc(sizeof(double) * ncols);
      starttime = MPI_Wtime();
      /* Insert your master code here to store the product into cc1 */
      numsent = 0;
        
      //
      for (i = 0; i < nrows; i++) { // for each row of aa
        for (j = 0; j < ncols; j++)
          buffer[j] = aa[i*ncols + j];
        MPI_Bcast(buffer, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD); // SEND aa[row i] to all
          
        for (j = 0; j < ncols; j++) { // for each column of bb
          for (k = 0; k < nrows; k++)
            buffer[k] = bb[j + k*nrows];
          BPI_Send(buffer, nrows, MPI_DOUBLE, i*ncols + j, i*ncols + j, MPI_COMM_WORLD); //SEND bb[col j] to one slave (which? does this iterate past number of slaves?)
            // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          numsent++;
        }
          
          
        // recieve here? -- waits for all of row i to complete before moving to row i++
        for (r = 0; r < nrows; r++) { // this loop might not be necessary
          MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                   MPI_COMM_WORLD, &status); // RECEIVE ans from slave
          // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          sender = status.MPI_SOURCE;
          anstype = status.MPI_TAG;
          cc1[r][anstype-1] = ans;
          
          if (numsent < nrows) {
              for (j = 0; j < ncols; j++) {
                  buffer[j] = aa[numsent*ncols + j];
              }
              MPI_Send(buffer, ncols, MPI_DOUBLE, sender, numsent+1,
                       MPI_COMM_WORLD);
              numsent++;
          } else {
              MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD); // breaks inner slave loop
          }
        }
      

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
                    MPI_Send(&ans, 1, MPI_DOUBLE, master, row, MPI_COMM_WORLD); // SEND ans to Master
                    
                    // ^^^^^^^^^^^^^^^^^^^^^ MAY DEAD LOCK ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    //https://stackoverflow.com/questions/49598174/how-does-mpi-blocking-send-and-receive-work
                }
            
        } while(brecs < nrows);// broadcasts recieved is less than number rows

    }
  } else {
    fprintf(stderr, "Usage matrix_times_vector <size>\n");
  }
  MPI_Finalize();
  return 0;
}

