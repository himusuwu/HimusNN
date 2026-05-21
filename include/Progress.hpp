#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <chrono>
#include <ratio>

class Progress
{
	public:
	  Progress(size_t bar_width, bool use_color);

	  void update(int epoch, int epochs, double mse, double acc, std::chrono::duration<double> eta_s, std::chrono::duration<double, std::milli> epoch_ms);

	  void finish();

	private:
	  const char* CLR_RED = "\033[31m";
      const char* CLR_YEL = "\033[33m";
      const char* CLR_GRN = "\033[32m";
      const char* CLR_RST = "\033[0m";

	  size_t bar_width;
	  bool use_color;
};

#endif