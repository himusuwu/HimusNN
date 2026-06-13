#ifndef UTIL_HPP
#define UTIL_HPP

#include "Matrix.hpp"

#include <thread>
#include <vector>

Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix outer(const std::vector<float>& a, const std::vector<float>& b);

template <typename Index, typename Callable> void paraller_for(Index start, Index end, Callable func)
{
    Index total = end - start;

    if (total < 20000)
    {
        for (Index i = start; i < end; i++)
        {
            func(i);
        }

        return;
    }

    static unsigned int num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    Index chunk_size = total / num_threads;

    for (unsigned int t = 0; t < num_threads; t++)
    {
        Index t_start = start + t * chunk_size;
        Index t_end = (t == num_threads - 1) ? end : t_start + chunk_size;

        threads.emplace_back(
            [t_start, t_end, &func]()
            {
                for (Index i = t_start; i < t_end; i++)
                {
                    func(i);
                }
            }
        );
    }

    for (auto& thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

#endif
