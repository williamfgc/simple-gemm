#include <iostream>

#include <gemm-dense-common.h>

namespace gd
{
void fill_random(float *A, const int n, const int m, std::mt19937 &r) {

  std::uniform_real_distribution<float> f(0, 1);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[i * m + j] = f(r);
    }
  }
}

std::chrono::system_clock::time_point
print_dtime(const std::chrono::system_clock::time_point &start,
            const std::string &hint) {

  const auto end = std::chrono::high_resolution_clock::now();
  const double dtime =
      std::chrono::duration<double, std::milli>(end - start).count() / 1E3;
  std::cout << "Time to " << hint << " " << dtime << " s\n";
  return end;
}

}
