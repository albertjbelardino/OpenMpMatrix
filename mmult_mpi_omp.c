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
  double *a;
  double *buffer, ans;
  int myid, master, numprocs;
  double starttime, endtime;
  MPI_Status status;
  /* insert other global variables here */
  int i, j, k, numsent, sender;
  int anstype, entry;
  int cur_row, cur_col;
  int prev_row, cur_pos;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc > 1) {
    nrows = atoi(argv[1]);
    ncols = nrows;
    buffer = malloc(sizeof(double) * ncols);
    master = 0;
    if (myid == master) {
      // Master Code goes here
      aa = gen_matrix(nrows, ncols);
      bb = gen_matrix(ncols, nrows);
      cc1 = malloc(sizeof(double) * nrows * nrows); 
      starttime = MPI_Wtime();
      /* Insert your master code here to store the product into cc1 */
      numsent = 0;
      cur_row = (numprocs-2) / ncols;
      cur_col = (numprocs-2) % ncols;
      for (i = 0; i < cur_row; i++) {
        for (j = 0; j < ncols; j++)
          buffer[j] = aa[i*ncols + j];
        MPI_Bcast(buffer, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
        for (j = 0; j < ncols; j++) {
          for (k = 0; k < nrows; k++)
            buffer[k] = bb[j + k*nrows];
          MPI_Send(buffer, nrows, MPI_DOUBLE, i*ncols + j + 1, i*ncols + j + 1, MPI_COMM_WORLD);
          numsent++;
        }
      }
      for (i = 0; i < ncols; i++)
        buffer[i] = aa[cur_row*ncols + i];
      MPI_Bcast(buffer, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
      for (i = 0; i < cur_col + 1; i++) {
        for (j = 0; j < nrows; j++)
          buffer[j] = bb[i + j*nrows];
        MPI_Send(buffer, nrows, MPI_DOUBLE, cur_row*ncols + i + 1, cur_row*ncols + i + 1, MPI_COMM_WORLD);
        numsent++;
      }
      for (i = 0; i < nrows * ncols; i++) {
        MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        sender = status.MPI_SOURCE;
        anstype = status.MPI_TAG;
        cc1[anstype - 1] = ans;
        if (numsent < nrows * ncols) {
          if (numsent % ncols == 0) {
            cur_row++;
            cur_col = 0;
            for (j = 0; j < ncols; j++)
              buffer[j] = aa[cur_row*ncols + j];
            MPI_Bcast(buffer, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
          } else {
            cur_col++;
          }
          for (j = 0; j < nrows; j++)
            buffer[j] = bb[cur_col + j*nrows];
          MPI_Send(buffer, nrows, MPI_DOUBLE, sender, numsent + 1, MPI_COMM_WORLD);
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
      a = malloc(sizeof(double) * ncols);
      prev_row = 0;
      MPI_Bcast(a, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
      if (myid <= nrows * ncols) {
        while (1) {
          MPI_Recv(buffer, nrows, MPI_DOUBLE, master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          if (status.MPI_TAG == 0)
            break;
          cur_pos = status.MPI_TAG; 
          cur_row = (cur_pos-1) / ncols;
          if (cur_row != prev_row) {
            MPI_Bcast(a, ncols, MPI_DOUBLE, master, MPI_COMM_WORLD);
            prev_row = cur_row;
          }
          ans = 0.0;
          for (i = 0; i < ncols; i++)
            ans += a[i] * buffer[i];
          MPI_Send(&ans, 1, MPI_DOUBLE, master, cur_pos, MPI_COMM_WORLD);
        }
      }
    }
  } else {
    fprintf(stderr, "Usage matrix_times_vector <size>\n");
  }
  MPI_Finalize();
  return 0;
}
