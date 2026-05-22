#include "Progress.hpp"
#include <iomanip>
#include <iostream>

Progress::Progress(size_t bar_width, bool use_color) : bar_width(bar_width), use_color(use_color) {}

void Progress::update(int epoch, int epochs, double last_mse, double last_acc, std::chrono::duration<double> last_eta_s, std::chrono::duration<double, std::milli> last_epoch_ms)
{
	double progress = static_cast<double>((epoch + 1)) / epochs;
	double filled = static_cast<int>(progress * bar_width);

	std::string bar = "[";
	bar.append(filled, '=');
	bar += (filled < bar_width ? '>' : '=');
	bar.append(bar_width - filled - 1, ' ');
	bar += "]";

	if(use_color)
	{
		const char* acc_color =
		(last_acc < 0.7) ? CLR_RED :
		(last_acc < 0.9) ? CLR_YEL : CLR_GRN;

		const char* mse_color =
		(last_mse < 1e-3) ? CLR_GRN :
		(last_mse < 1e-2) ? CLR_YEL : CLR_RED;

		std::cout << std::fixed << std::setprecision(6)
			<< "\r" << bar << " "
			<< "mse=" << mse_color << last_mse << CLR_RST << " "
			<< std::setprecision(3)
			<< "acc=" << acc_color << last_acc << CLR_RST << " "
			<< std::setprecision(2)
			<< "eta=" << last_eta_s.count() << "s "
			<< "ep=" << last_epoch_ms.count() << "ms"
		<< std::flush;
	}
	else
	{
		std::cout << std::fixed << std::setprecision(6)
			<< "\r" << bar << " "
			<< "mse=" << last_mse << " "
			<< std::setprecision(3)
			<< "acc=" << last_acc << " "
			<< std::setprecision(2)
			<< "eta=" << last_eta_s.count() << "s "
			<< "ep=" << last_epoch_ms.count() << "ms"
		<< std::flush;
	}
}

void Progress::finish()
{
	std::cout << "\n\nTraining ended!\n" << std::endl;
}