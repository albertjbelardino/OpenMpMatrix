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
  int myid, numprocs;
  double starttime, endtime;
  MPI_Status status;
  /* insert other global variables here */
  double *buffer, ans;
  double *times;
  double total_times;
  int run_index;
  int nruns;
  int myid, numprocs; //master,
  int i, j, numsent, sender;
  int anstype, row;
  srand(time(0));
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc > 1) {
    nrows = atoi(argv[1]);
    ncols = nrows;
    aa = (double*)malloc(sizeof(double) * nrows * ncols);
    bb = (double*)malloc(sizeof(double) * nrows * ncols);
    cc1 = (double*)malloc(sizeof(double) * nrows * ncols);
    cc2 = (double*)malloc(sizeof(double) * nrows * ncols);

    buffer = (double*)malloc(sizeof(double) * ncols);
    //master = 0; // redundant with next line
    if (myid == 0) {
      // Master Code goes here
		/*
			first part
      //start from here
for each row in A
   //this following is the matrix times vector program
   broadcast row a to every slave
   for each column vector in B
       send column vector b to some slave process 
		*/
      MPI_Bcast(b, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
      for (i = 0; i < min(numprocs-1, nrows); i++) {
        for (j = 0; j < ncols; j++) {
          buffer[j] = aa[i * ncols + j];
        }
        MPI_Send(buffer, ncols, MPI_DOUBLE, i+1, i+1, MPI_COMM_WORLD);
        numsent++;
      }


      aa = gen_matrix(nrows, ncols);
      bb = gen_matrix(ncols, nrows);
      cc1 = malloc(sizeof(double) * nrows * nrows);


     /*
      * Generate Random Matrices aa, bb, cc1
      * Note: This needs to be adapted to accept various dimension matrices
      * including test to see if aa works with bb
     */
      for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++) {
          aa[i*ncols + j] = (double)rand()/RAND_MAX;
        }
      }
      for (i = 0; i < ncols; i++) {
        for (j = 0; j < nrows; j++) {
          bb[i*ncols + j] = (double)rand()/RAND_MAX;
        }
      }
      for (i = 0; i < ncols; i++) { //must be [m,p] if aa=[m,n] bb=[n,p]
        for (j = 0; j < ncols; j++) {
          cc1[i*ncols + j] = (double)rand()/RAND_MAX;
        }
      }


      // starttime = MPI_Wtime();// moved below

/*    //start from here
         for each row in A
         //this following is the matrix times vector program
         for each column vector in B
             send row a to every slave
             send column vector b to some slave process 
*/

      starttime = MPI_Wtime();
      numsent = 0;
      MPI_Bcast(bb, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
      for (i = 0; i < min(numprocs-1, nrows); i++) {
        for (j = 0; j < ncols; j++) {
          buffer[j] = aa[i * ncols + j];
        }
        MPI_Send(buffer, ncols, MPI_DOUBLE, i+1, i+1, MPI_COMM_WORLD);
        numsent++;
      }

      for (i = 0; i < nrows; i++) {
        MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        sender = status.MPI_SOURCE;
        anstype = status.MPI_TAG;
        c[anstype-1] = ans;
        if (numsent < nrows) {
          for (j = 0; j < ncols; j++) {
            buffer[j] = aa[numsent*ncols + j];
          }
          MPI_Send(buffer, ncols, MPI_DOUBLE, sender, numsent+1,
                   MPI_COMM_WORLD);
          numsent++;
        } else {
          MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
        }
      }


      endtime = MPI_Wtime();
      printf("%f\n",(endtime - starttime));
      cc2  = malloc(sizeof(double) * nrows * nrows);
      mmult(cc2, aa, nrows, ncols, bb, ncols, nrows);
      compare_matrices(cc2, cc1, nrows, nrows);
    } else {
      // Slave Code goes here
	/*
		third part
while(1)
  receive row and column
  if stop
    break
  compute dot product
  send back to master
	*/
    }
  } else {
    fprintf(stderr, "Usage matrix_times_vector <size>\n");
  }
  MPI_Finalize();
  return 0;
}
