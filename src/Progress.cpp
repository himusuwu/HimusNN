#include "Progress.hpp"

#include <iomanip>
#include <iostream>

Progress::Progress(size_t bar_width, bool use_color) : bar_width(bar_width), use_color(use_color) {}

void Progress::update(
    int epoch,
    int epochs,
    float last_mse,
    float last_acc,
    float val_mse,
    float val_acc,
    std::chrono::duration<float> last_eta_s,
    std::chrono::duration<float, std::milli> last_epoch_ms
)
{
    float progress = static_cast<float>((epoch + 1)) / epochs;
    float filled = static_cast<int>(progress * bar_width);

    std::string bar = "[";
    bar.append(filled, '=');
    bar += (filled < bar_width ? '>' : '=');
    bar.append(bar_width - filled - 1, ' ');
    bar += "]";

    if (use_color)
    {
        const char* acc_color = (last_acc < 0.7f) ? CLR_RED : (last_acc < 0.9f) ? CLR_YEL : CLR_GRN;

        const char* mse_color = (last_mse < 1e-3f) ? CLR_GRN : (last_mse < 1e-2f) ? CLR_YEL : CLR_RED;

        const char* val_acc_color = (val_acc < 0.7f) ? CLR_RED : (val_acc < 0.9f) ? CLR_YEL : CLR_GRN;

        const char* val_mse_color = (val_mse < 1e-3f) ? CLR_GRN : (val_mse < 1e-2f) ? CLR_YEL : CLR_RED;

        std::cout << std::fixed << std::setprecision(6) << "\r" << bar << " "
                  << "mse=" << mse_color << last_mse << CLR_RST << " " << std::setprecision(3) << "acc=" << acc_color
                  << last_acc << CLR_RST << " "
                  << "val_mse=" << val_mse_color << val_mse << CLR_RST << " " << std::setprecision(3)
                  << "val_acc=" << val_acc_color << val_acc << CLR_RST << " " << std::setprecision(2)
                  << "eta=" << last_eta_s.count() << "s "
                  << "ep=" << last_epoch_ms.count() << "ms" << std::flush;
    }
    else
    {
        std::cout << std::fixed << std::setprecision(6) << "\r" << bar << " "
                  << "mse=" << last_mse << " " << std::setprecision(3) << "acc=" << last_acc << " "
                  << std::setprecision(2) << "val_mse=" << val_mse << " " << std::setprecision(3)
                  << "val_acc=" << val_acc << " " << std::setprecision(2) << "eta=" << last_eta_s.count() << "s "
                  << "ep=" << last_epoch_ms.count() << "ms" << std::flush;
    }
}

void Progress::finish()
{
    std::cout << "\n\nTraining ended!\n" << std::endl;
}
