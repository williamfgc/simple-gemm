#include <algorithm>
#include <execution>
#include <future>
#include <iostream>
#include <vector>

#include <gemm-dense-common.h>

namespace {

class int_iterator
{
public:
    using difference_type = std::ptrdiff_t;
    using value_type = int;
    using pointer = int*;
    using reference = int;
    using iterator_category = std::random_access_iterator_tag;

    int_iterator(int i) : _i(i)
    {
    }

    int_iterator&
    operator++()
    {
        ++_i;
        return *this;
    }

    int
    operator*() const
    {
        return _i;
    }

    pointer
    operator->()
    {
        return &_i;
    }

    int
    operator[](difference_type n)
    {
        return _i + n;
    }

private:
    friend int_iterator
    operator+(int_iterator a, int_iterator b)
    {
        return {a._i + b._i};
    }

    friend int
    operator-(int_iterator a, int_iterator b)
    {
        return a._i - b._i;
    }

    friend bool
    operator<(int_iterator a, int_iterator b)
    {
        return a._i < b._i;
    }

    friend bool
    operator==(int_iterator a, int_iterator b)
    {
        return a._i == b._i;
    }

    friend bool
    operator!=(int_iterator a, int_iterator b)
    {
        return !(a == b);
    }

private:
    int _i;
};

void gemm(float *A, float *B, float *C, const int A_rows, const int A_cols,
          const int B_rows) {
    std::for_each(std::execution::par,
        int_iterator(0), int_iterator(A_rows),
        [=](int i) {
            for (int k = 0; k < A_cols; k++) {
                auto temp = A[i * A_cols + k];
                for (int j = 0; j < B_rows; j++) {
                    C[i * B_rows + j] += temp * B[k * B_rows + j];
                }
            }
        });
}

// static void print_matrix(float *A, const int A_rows, const int A_cols) {

//   std::cout << "[ ";
//   for (int i = 0; i < A_rows; ++i) {
//     for (int j = 0; j < A_cols; ++j) {
//       std::cout << A[i * A_cols + j] << " ";
//     }
//     std::cout << ";\n";
//   }
//   std::cout << "]\n";
// }

}  // namespace

int main(int argc, char *argv[]) {
    // Set seed once
    std::mt19937 e(std::chrono::duration_cast<std::chrono::hours>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count());

    int A_rows, A_cols, B_rows, B_cols;

    if (argc != 4) {
        std::cout << "Usage: 3 arguments: matrix A rows, matrix A cols and "
                     "matrix B cols"
                  << std::endl;
        return 1;
    } else {
        A_rows = atoi(argv[1]);
        A_cols = atoi(argv[2]);
        B_rows = atoi(argv[2]);
        B_cols = atoi(argv[3]);
    }

    const int nthreads = std::stoi(std::getenv("OMP_NUM_THREADS"));

    const auto start = std::chrono::high_resolution_clock::now();
    float *A = new float[A_rows * A_cols];
    auto tmp = gd::print_dtime(start, "allocate A");
    float *B = new float[B_rows * B_cols];
    tmp = gd::print_dtime(tmp, "allocate B");
    float *C = new float[A_rows * B_cols]();  // value-init to zero
    tmp = gd::print_dtime(tmp, "initialize C");

    gd::fill_random(A, A_rows, A_cols, e);
    tmp = gd::print_dtime(tmp, "fill A");
    gd::fill_random(B, B_rows, B_cols, e);
    tmp = gd::print_dtime(tmp, "fill B");

    gemm(A, B, C, A_rows, A_cols, B_cols);
    tmp = gd::print_dtime(tmp, "simple gemm");
    tmp = gd::print_dtime(start, "total time");

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
