#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char * argv[])
{
    int n = atoi(argv[1]);
    char symbol = argv[2][0];
    char *arr = new char[n];
    int par_count = 0, lin_count = 0;
    double starttime = 0.0f, endtime = 0.0f, linstarttime = 0.0f, linendtime = 0.0f;
    int i1 = 0, i2 = 0;
    int proc_count = 0;

    for (int i = 0; i < n; i++)
        arr[i] = rand() % 26 + 97;

    int ProcNum = 0, ProcRank = 0;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    if (ProcRank == 0)
    {
        linstarttime = MPI_Wtime();
        for (int i = 0; i < n; i++)
            if (arr[i] == symbol) lin_count++;
        linendtime = MPI_Wtime();
		if(n < 80)
			printf("%s\n", arr);
        printf("frequency of %c = %f\n", symbol, (float)lin_count / (float)n);
        printf("work time of linuar algoritm = %f\n", linendtime - linstarttime);
        starttime = MPI_Wtime();
		for (int i = 1; i < ProcNum; i++)
		{
			i1 = i * n / ProcNum;
			i2 = (i + 1) * n / ProcNum;
			MPI_Send(&arr[i1], i2 - i1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
		}
    }

    i1 = ProcRank * n / ProcNum;
    i2 = (ProcRank + 1) * n / ProcNum;

	if (ProcRank > 0)
	{
		MPI_Status stat;
		char* buf = new char[i2 - i1];
		MPI_Recv(buf, i2 - i1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
		for (int j = 0; j < i2 - i1; j++)
			if (buf[j] == symbol)
				proc_count++;
		delete buf;
	}
	else
	{
		for (int j = i1; j < i2; j++)
			if (arr[j] == symbol)
				proc_count++;
	}

    MPI_Reduce(&proc_count, &par_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ProcRank == 0)
    {
        endtime = MPI_Wtime();
        printf("frequency of %c = %f\n", symbol, (float)par_count / (float)n);
        printf("work time of parallel algoritm = %f\n", endtime - starttime);
        if (par_count == lin_count)
            printf("All correct! Linear algorithm absolutely converges with parallel");
    }

    MPI_Finalize();
    
    delete arr;
    return 0;
}
