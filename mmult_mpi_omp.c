#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#define min(x, y) ((x)<(y)?(x):(y))

int verify_dimensions(int aRows, int aCols, int bRows, int bCols);
int mmult(double *c, double *a, int aRows, int aCols, double *b, int bRows, int bCols);
void compare_matrices(double *a, double *b, int nRows, int nCols);

/** 
    Program to multiply a matrix times a matrix using both
    mpi to distribute the computation among nodes and omp
    to distribute the computation among threads.
*/

int main(int argc, char* argv[])
{
  int nrows1, ncols1;
  int nrows2, ncols2;
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
  int anstype;
  int cur_row, cur_col;
  int prev_row, cur_pos;
  FILE *fp;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc > 2) {
    master = 0;
    if (myid == master) {
      // Master Code goes here
      if ((fp = fopen(argv[1], "r")) == NULL) {
        printf("File %s does not exist!\n", argv[1]);
        return 1;
      }
      if (fscanf(fp, "rows(%d) cols(%d)", &nrows1, &ncols1) != 2) {
        printf("Invalid file format!\n");
        return 1;
      }
      fgetc(fp);
      aa = malloc(sizeof(double) * nrows1 * ncols1);
      for (i = 0; i < nrows1 * ncols1; i++) {
        if (fscanf(fp, "%lf", &aa[i]) != 1) {
          printf("Invalid file format!\n");
          return 1;
        }
      }
      if ((fp = fopen(argv[2], "r")) == NULL) {
        printf("File %s does not exist!\n", argv[2]);
        return 1;
      }
      if (fscanf(fp, "rows(%d) cols(%d)", &nrows2, &ncols2) != 2) {
        printf("Invalid file format!\n");
        return 1;
      }
      fgetc(fp);
      bb = malloc(sizeof(double) * nrows2 * ncols2);
      for (i = 0; i < nrows2 * ncols2; i++) {
        if (fscanf(fp, "%lf", &bb[i]) != 1) {
          printf("Invalid file format!\n");
          return 1;
        }
      }
      if (!verify_dimensions(nrows1, ncols1, nrows2, ncols2)) {
          printf("Can't multiply matrices of dimensions %dx%d and %dx%d!\n", nrows1, ncols1, nrows2, ncols2);
          return 1;
      }
      cc1 = malloc(sizeof(double) * nrows1 * ncols2); 
      buffer = malloc(sizeof(double) * ncols1);
      starttime = MPI_Wtime();
      /* Insert your master code here to store the product into cc1 */
      MPI_Bcast(&nrows1, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&ncols1, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&nrows2, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&ncols2, 1, MPI_INT, master, MPI_COMM_WORLD);
      numsent = 0;
      cur_row = min((numprocs-2) / ncols2, nrows1 - 1);
      if (numprocs - 1 > nrows1 * ncols2)
        cur_col = ncols2 - 1;
      else
        cur_col = (numprocs-2) % ncols2;
      for (i = 0; i < cur_row; i++) {
        for (j = 0; j < ncols1; j++)
          buffer[j] = aa[i*ncols1 + j];
        MPI_Bcast(buffer, ncols1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        for (j = 0; j < ncols2; j++) {
          for (k = 0; k < nrows2; k++)
            buffer[k] = bb[j + k*ncols2];
          MPI_Send(buffer, nrows2, MPI_DOUBLE, i*ncols2 + j + 1, i*ncols2 + j + 1, MPI_COMM_WORLD);
          numsent++;
        }
      }
      for (i = 0; i < ncols1; i++)
        buffer[i] = aa[cur_row*ncols1 + i];
      MPI_Bcast(buffer, ncols1, MPI_DOUBLE, master, MPI_COMM_WORLD);
      for (i = 0; i < cur_col + 1; i++) {
        for (j = 0; j < nrows2; j++)
          buffer[j] = bb[i + j*ncols2];
        MPI_Send(buffer, nrows2, MPI_DOUBLE, cur_row*ncols2 + i + 1, cur_row*ncols2 + i + 1, MPI_COMM_WORLD);
        numsent++;
      }
      for (i = 0; i < nrows1 * ncols2; i++) {
        MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        sender = status.MPI_SOURCE;
        anstype = status.MPI_TAG;
        cc1[anstype - 1] = ans;
        if (numsent < nrows1 * ncols2) {
          if (numsent % ncols2 == 0) {
            cur_row++;
            cur_col = 0;
            for (j = 0; j < ncols1; j++)
              buffer[j] = aa[cur_row*ncols1 + j];
            MPI_Bcast(buffer, ncols1, MPI_DOUBLE, master, MPI_COMM_WORLD);
          } else {
            cur_col++;
          }
          for (j = 0; j < nrows2; j++)
            buffer[j] = bb[cur_col + j*ncols2];
          MPI_Send(buffer, nrows2, MPI_DOUBLE, sender, numsent + 1, MPI_COMM_WORLD);
          numsent++;
        } else {
          MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
        }
      }
      endtime = MPI_Wtime();
      printf("%f\n",(endtime - starttime));
      fp = fopen("result.txt", "w+");
      fprintf(fp, "rows(%d) cols(%d)\n", nrows1, ncols2);
      for (i = 0; i < nrows1; i++) {
        for (j = 0; j < ncols2; j++)
          fprintf(fp, "%f ", cc1[i*ncols2 + j]);
        fprintf(fp, "\n");
      }
      fclose(fp);
      cc2  = malloc(sizeof(double) * nrows1 * ncols2);
      mmult(cc2, aa, nrows1, ncols1, bb, nrows2, ncols2);
      compare_matrices(cc2, cc1, nrows1, ncols2);
      free(aa);
      free(bb);
      free(cc1);
      free(cc2);
    } else {
      // Slave Code goes here
      MPI_Bcast(&nrows1, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&ncols1, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&nrows2, 1, MPI_INT, master, MPI_COMM_WORLD);
      MPI_Bcast(&ncols2, 1, MPI_INT, master, MPI_COMM_WORLD);
      buffer = malloc(sizeof(double) * nrows2);
      a = malloc(sizeof(double) * ncols1);
      prev_row = -1;
      if (myid <= nrows1 * ncols2) {
        while (1) {
          MPI_Recv(buffer, nrows2, MPI_DOUBLE, master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          if (status.MPI_TAG == 0) {
            while (prev_row < nrows1 - 1) {
              MPI_Bcast(a, ncols1, MPI_DOUBLE, master, MPI_COMM_WORLD);
              prev_row++;
            }
            break;
          }
          cur_pos = status.MPI_TAG; 
          cur_row = (cur_pos-1) / ncols2;
          while (prev_row < cur_row) {
            MPI_Bcast(a, ncols1, MPI_DOUBLE, master, MPI_COMM_WORLD);
            prev_row++;
          }
          ans = 0.0;
          for (i = 0; i < ncols1; i++)
            ans += a[i] * buffer[i];
          MPI_Send(&ans, 1, MPI_DOUBLE, master, cur_pos, MPI_COMM_WORLD);
        }
      }
      free(a);
    }
    free(buffer);
  } else {
    fprintf(stderr, "Usage mmult_mpi_omp <matrix_a> <matrix_b>\n");
  }
  MPI_Finalize();
  return 0;
}

int verify_dimensions(int aRows, int aCols, int bRows, int bCols) {
  if (aCols == bRows)
    return 1;
  return 0;
}

