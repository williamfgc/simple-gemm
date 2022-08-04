#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <time.h>

#define NITR 1

#define GEMM_FMULS( m_, n_, k_ ) ( (m_) * (n_) * (k_)) 
#define GEMM_FADDS( m_, n_, k_ ) ( (m_) * (n_) * (k_)) 
#define GEMM_FLOPS( m_, n_, k_ ) GEMM_FMULS( (double)(m_), (double)(n_), (double)(k_) ) \
	+ GEMM_FADDS( (double)(m_), (double)(n_), (double)(k_) )

static struct timespec print_dtime(struct timespec start, const char *process) {
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  const float dtime = ((float)((end.tv_sec - start.tv_sec) * 1000000 +
                               (end.tv_nsec - start.tv_nsec) / 1000)) /
                      1E6;

  printf("Time to %s = %f s\n", process, dtime);
  return end;
}

int main( int argc, char* argv[] )
{

  struct timeval start, end;
  double time; 
	  
  int M;
  int N;
  int K;

  Kokkos::initialize( argc, argv );
  {
  
  for (int i = 4096; i <= 16384; i += 1024 )
  {
  
  M = i;
  N = i;
  K = i;
 
  Kokkos::View<double**> A ( "A", M, K );
  Kokkos::View<double**> B ( "B", K, N );
  Kokkos::View<double**> C ( "C", M, N );

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  Kokkos::parallel_for( "gemm_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    C(m, n) = 0.0;
  });

  Kokkos::parallel_for( "gemm_init", mdrange_policy( {0, 0}, {M, K}), KOKKOS_LAMBDA ( int m, int k )
  {
    A(m, k) = 4.0;
  });

  Kokkos::parallel_for( "gemm_init", mdrange_policy( {0, 0}, {K, N}), KOKKOS_LAMBDA ( int k, int n )
  {
    B(k, n) = 2.0;
  });

  Kokkos::fence();

  // Warming
  Kokkos::parallel_for( "gemm", mdrange_policy( {0, 0}, {M, K}), KOKKOS_LAMBDA ( int i, int k )
  {
    float tmp = 0.0;  
    for ( int j = 0; j < N; j++ )
    {
      C(i,j) += A(i, k) * B(k, j);
    }
  });
  
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < NITR; i++ )
  { 
  Kokkos::parallel_for( "gemm", mdrange_policy( {0, 0}, {M, K}), KOKKOS_LAMBDA ( int i, int k )
  {
    float tmp;  
    tmp = A(i, k);
    for ( int j = 0; j < N; j++ )
    {
      C(i,j) += tmp * B(k, j);
    }
  });
  
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( float ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / NITR;

  double flops = GEMM_FLOPS(M, N, K);

  printf( "GEMM = %d Time = %e s GFLOPS = %e\n", M, time, ( flops / time ) / 1e9 );

  //Kokkos::kokkos_free<>(A);
  //Kokkos::kokkos_free<>(B);
  //Kokkos::kokkos_free<>(C);

  }
  
  }

  Kokkos::finalize();
  
  return 0;
}
