#include <chrono>
#include <random>
#include <string>

namespace gd
{
void fill_random(float *A, const int n, const int m, std::mt19937 &r);

std::chrono::system_clock::time_point
print_dtime(const std::chrono::system_clock::time_point &start,
            const std::string &hint);
}
