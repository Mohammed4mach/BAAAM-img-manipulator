#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "imgmanip.h"
#include "QFileDialog"
#include "unordered_map"

using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

QPixmap MainWindow::changePixmapScale(QPixmap pixmap)
{
    return pixmap.scaled(335, 303, Qt::KeepAspectRatioByExpanding);
}

void MainWindow::on_import_image_clicked()
{
    QString img_path = QFileDialog::getOpenFileName(this,
                                                   tr("Select Image"),
                                                   "",
                                                   tr("Images (*.png *.xpm *.jpg)")
    );

    if(img_path.isNull())
        return;

    QPixmap pixmap(img_path);

    pixmap = this->changePixmapScale(pixmap);

    this->img_in_path = img_path.toStdString();
    this->img_in = pixmap;

    // Set image
    this->ui->image_in->setPixmap(this->img_in);

    // Set hisogram
    QPixmap histogram = ImgManip::get_histogram(this->img_in_path);

    this->ui->histogram_in->setPixmap(histogram);
}

void MainWindow::on_segmentation_ok_clicked()
{
    int threshold_value = this->ui->segmentation_input->value();

    QPixmap pixmap = ImgManip::segmentation(this->img_in_path, threshold_value);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}

void MainWindow::on_histogram_ok_clicked()
{
    QPixmap pixmap = ImgManip::equalization(this->img_in_path);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_smoothing_linear_ok_clicked()
{
    int kernel_size = (int) this->ui->kernel_size_input->currentText().at(0).unicode() - '0';
    // int kernel_size = this->ui->kernel_size_input->itemText()

    QPixmap pixmap = ImgManip::box_filter(this->img_in_path, kernel_size);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_smoothing_nonlinear_ok_clicked()
{
    int kernel_size = (int) this->ui->kernel_size_input->currentText().at(0).unicode() - '0';

    QPixmap pixmap = ImgManip::min_filter(this->img_in_path, kernel_size);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_sharpening_ok_clicked()
{
    bool enhanced = this->ui->laplacian_enhaced_check->isChecked();

    QPixmap pixmap = ImgManip::laplacian(this->img_in_path, enhanced);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_detection_ok_clicked()
{
    string detection_alg = this->ui->detection_filter_input->currentText().toStdString();

    unordered_map<string, QPixmap (*)(string)> alg_func = {
        {"Sobel Filter", & ImgManip::sobel_filter},
        {"Canny Filter", & ImgManip::canny_threshold}
    };

    QPixmap pixmap = (* alg_func[detection_alg])(this->img_in_path);
    // QPixmap pixmap = ImgManip::sobel_filter(this->img_in_path);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}

double MainWindow::calc_trans_time(int baud_rate, int multiplier = 1)
{
    double time = ImgManip::transmission_time(this->img_in_path, baud_rate);

    return time * multiplier;
}

void MainWindow::on_baud_input_valueChanged(int value)
{
    unordered_map<string, int> channel_multiplier = {
        {"Grayscale", 1},
        {"RGB", 3}
    };

    int baud_rate = this->ui->baud_input->value();

    string channel = this->ui->trans_time_channel_input->currentText().toStdString();
    int multiplier = channel_multiplier[channel];

    double time = this->calc_trans_time(baud_rate, multiplier);

    this->ui->transmission_time_output->display(QString::number(time));
}

void MainWindow::on_trans_time_channel_input_currentTextChanged(const QString &arg1)
{
    on_baud_input_valueChanged(1);
}
