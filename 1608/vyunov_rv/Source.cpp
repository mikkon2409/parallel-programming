#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
const double eps = 1e-4;
int proc_num = 0;
int proc_rank = 0;

int checkPrecision(double x, double y, double prec) {
	return abs(x - y) < prec;
}

int checkVectorPrecision(double * x, double * y, double prec, int N) {
	int isOk = 1;
	for (int i = 0; i < N; i++) {
		isOk = isOk && checkPrecision(x[i], y[i], prec);
	}
	return isOk;
}

void fillSquareMatrix(double * mat, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i * N + j] = (i == j) ? 0.125f : 0;
		}
	}
}

void fillVector(double * vec, int N) {
	for (int i = 0; i < N; i++) {
		vec[i] = 1;
	}
}

int main(int argc, char* argv[]) {
	int N = atoi(argv[1]);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	double* A = nullptr;
	double* B = nullptr;
	double* X_old = new double[N];
	double* X_new = new double[N];

	double* X_old_lin = new double[N];
	double* X_new_lin = new double[N];
	double start = 0, end = 0;

	int begin_row = proc_rank * N / proc_num;
	int end_row = (proc_rank + 1) * N / proc_num;
	int end_calc = 0;
	// i - rows j - cols
	if (proc_rank == 0) {
		A = new double[N * N];
		fillSquareMatrix(A, N);
		B = new double[N];
		fillVector(B, N);
		memset(X_old_lin, 0, sizeof(double) * N);
		memset(X_new_lin, 0, sizeof(double) * N);
		start = MPI_Wtime();
		do {
			double * tmp = X_old_lin;
			X_old_lin = X_new_lin;
			X_new_lin = tmp;
			for (int i = 0; i < N; i++) {
				double div = A[i * N + i];
				X_new_lin[i] = 0;
				for (int j = 0; j < N; j++) {
					if (j != i)
						X_new_lin[i] -= X_old_lin[j] * A[i * N + j];
				}
				X_new_lin[i] += B[i];
				X_new_lin[i] /= A[i * N + i];
			}
		} while (checkVectorPrecision(X_old_lin, X_new_lin, eps, N));
		end = MPI_Wtime();
		std::cout << "Linear version time: " << end - start << std::endl;
	}
	if (proc_rank == 0) {
		start = MPI_Wtime();
		for (int i = 1; i < proc_num; i++) {
			int begin_row1 = i * N / proc_num;
			int end_row1 = (i + 1) * N / proc_num;
			MPI_Send(&A[begin_row1 * N], (end_row1 - begin_row1) * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&B[begin_row1], (end_row1 - begin_row1), MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}
	}
	memset(X_old, 0, sizeof(double) * N);
	memset(X_new, 0, sizeof(double) * N);
	if (proc_rank != 0) {
		A = new double[(end_row - begin_row) * N];
		B = new double[end_row - begin_row];
		MPI_Status stat;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(A, (end_row - begin_row) * N, MPI_DOUBLE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
		MPI_Probe(0, 1, MPI_COMM_WORLD, &stat);
		MPI_Recv(B, end_row - begin_row, MPI_DOUBLE, stat.MPI_SOURCE, 1, MPI_COMM_WORLD, &stat);
	}
	MPI_Bcast(X_old, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(X_new, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	do {
		if (proc_rank != 0) {
			MPI_Send(&X_new[begin_row], end_row - begin_row, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
		}
		if (proc_rank == 0) {
			for (int i = begin_row; i < end_row; i++) {
				X_old[i] = X_new[i];
			}
			for (int i = 1; i < proc_num; i++) {
				MPI_Status stat;
				MPI_Probe(MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &stat);
				int begin_row1 = stat.MPI_SOURCE * N / proc_num;
				int end_row1 = (stat.MPI_SOURCE + 1) * N / proc_num;
				MPI_Recv(&X_old[begin_row1], end_row1 - begin_row1, MPI_DOUBLE, stat.MPI_SOURCE, 10, MPI_COMM_WORLD, &stat);
			}
		}
		MPI_Bcast(X_old, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int i = begin_row; i < end_row; i++) {
			X_new[i] = 0;
			for (int j = 0; j < N; j++) {
				if (j != i)
					X_new[i] -= X_old[j] * A[(i - begin_row) * N + j];
			}
			X_new[i] += B[i - begin_row];
			X_new[i] /= A[(i - begin_row) * N + i];
		}
		int res = checkVectorPrecision(&X_old[begin_row], &X_new[begin_row], eps, end_row - begin_row);
		MPI_Allreduce(&res, &end_calc, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
	} while (!end_calc);
	if (proc_rank != 0) {
		MPI_Send(&X_new[begin_row], end_row - begin_row, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
	}
	if (proc_rank == 0) {
		for (int i = 1; i < proc_num; i++) {
			MPI_Status stat;
			int begin_row1 = i * N / proc_num;
			int end_row1 = (i + 1) * N / proc_num;
			MPI_Probe(MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &stat);
			MPI_Recv(&X_new[i * N / proc_num], end_row1 - begin_row1, MPI_DOUBLE, stat.MPI_SOURCE, 10, MPI_COMM_WORLD, &stat);
		}
		end = MPI_Wtime();
		std::cout << "Parallel version style: " << end - start << std::endl;
		bool isOk = true;
		for (int i = 0; i < N; i++) {
			isOk = isOk && (X_new[i] == X_new_lin[i]);
		}
		if (isOk)
			std::cout << "TEST SUCCESS" << std::endl;
		else
			std::cout << "TEST FAILED" << std::endl;
	}
	MPI_Finalize();
	return 0;
}