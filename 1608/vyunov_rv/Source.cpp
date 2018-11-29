#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <stdlib.h>
#include <iostream>

using namespace cv;

unsigned char clamp(float val) {
    if (val < 0)
        val = 0;
    else if(val > UCHAR_MAX)
        val = UCHAR_MAX;
    return (unsigned char)val;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << " Usage: display_image ImageToLoadAndDisplay" << std::endl;
        return -1;
    }
    Mat *image = nullptr;
    Mat *lin_image = nullptr;
    Mat *res_image = nullptr;
    int img_width = 0;
    int img_height = 0;
    int img_size = 0;
    
    unsigned char *buf  = nullptr;
    int size = 0;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int proc_num = 0;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
        image = new Mat();
        *image = imread(argv[1], IMREAD_GRAYSCALE);
        if (!image->data) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        img_width = image->cols;
        img_height = image->rows;
        img_size = img_height * img_width;
        lin_image = new Mat(img_height, img_width, CV_8UC1);
        res_image = new Mat(img_height, img_width, CV_8UC1);

        unsigned char max = 0;
        unsigned char min = UCHAR_MAX;
        start_time = MPI_Wtime();
        for (int i = 0; i < img_size; i++) {
            if (image->data[i] < min)
                min = image->data[i];
            if (image->data[i] > max)
                max = image->data[i];
        }
        for (int i = 0; i < img_size; i++) {
            lin_image->data[i] = clamp(((float)(image->data[i] - min) * ((float)UCHAR_MAX / (max - min))));
        }
        end_time = MPI_Wtime();
        std::cout << "Line time: " << end_time - start_time << std::endl;
        int first_end = img_height / proc_num;
        size = first_end * img_width;
        
        start_time = MPI_Wtime();
        for (int i = 1; i < proc_num; i++) {
            int begin = i * img_height / proc_num;
            int end = (i + 1) * img_height / proc_num;
            MPI_Send(&image->data[begin * img_width], (end - begin) * img_width, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    int gen_min = 0;
    int gen_max = 0;
    int min = UCHAR_MAX;
    int max = 0;

    if (proc_rank > 0){
        MPI_Status stat;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &stat);
        MPI_Get_count(&stat, MPI_CHAR, &size);
        buf = new unsigned char[size];
        MPI_Recv(buf, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < size; i++) {
            if (buf[i] < min)
                min = buf[i];
            if (buf[i] > max)
                max = buf[i];
        }
    }
    else {
        for (int i = 0; i < size; i++) {
            if (image->data[i] < min)
                min = image->data[i];
            if (image->data[i] > max)
                max = image->data[i];
        }
    }
    MPI_Allreduce(&max, &gen_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min, &gen_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (proc_rank > 0) {
        for (int i = 0; i < size; i++) {
            buf[i] = clamp(((float)(buf[i] - gen_min) * ((float)UCHAR_MAX / (gen_max - gen_min))));
        }
        MPI_Send(buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    else {
        for (int i = 0; i < size; i++) {
            res_image->data[i] = clamp(((float)(image->data[i] - gen_min) * ((float)UCHAR_MAX / (gen_max - gen_min))));
        }
    }

    if (proc_rank == 0) {
        for (int i = 1; i < proc_num; i++) {
            MPI_Status stat;
            int recv_size = 0;
            MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stat);
            MPI_Get_count(&stat, MPI_CHAR, &recv_size);
            int begin = stat.MPI_SOURCE * img_height / proc_num;
            MPI_Recv(&res_image->data[begin * img_width], recv_size, MPI_CHAR, stat.MPI_SOURCE, 1, MPI_COMM_WORLD, &stat);
        }
        end_time = MPI_Wtime();
        std::cout << "Parallel time: " << end_time - start_time << std::endl;

        bool ok = true;
        for (int i = 0; i < img_size; i++) {
            if (res_image->data[i] != lin_image->data[i]) {
                ok = false;
            }
        }
        if (ok) {
            std::cout << "TEST PASSED!" << std::endl;
        }
        else {
            std::cout << "TEST FAILED!" << std::endl;
        }
        namedWindow("RAW IMAGE", WINDOW_KEEPRATIO);
        imshow("RAW IMAGE", *image);
        namedWindow("LINE CALC", WINDOW_KEEPRATIO);
        imshow("LINE CALC", *lin_image);
        namedWindow("PARALLEL CALC", WINDOW_KEEPRATIO);
        imshow("PARALLEL CALC", *res_image);

        waitKey(0);
    }
    MPI_Finalize();
    return 0;
}
