#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "imgmanip.h"
#include "QFileDialog"

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
    int kernel_size = (int) this->ui->kernel_size_input->currentText().at(0).unicode();
    // int kernel_size = this->ui->kernel_size_input->itemText()

    QPixmap pixmap = ImgManip::box_filter(this->img_in_path, kernel_size);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_smoothing_nonlinear_ok_clicked()
{
    int kernel_size = (int) this->ui->kernel_size_input->currentText().at(0).unicode();

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
    QPixmap pixmap = ImgManip::sobel_filter(this->img_in_path);

    pixmap = this->changePixmapScale(pixmap);

    this->ui->image_out->setPixmap(pixmap);
}


void MainWindow::on_baud_input_valueChanged(int value)
{
    int baud_rate = this->ui->baud_input->value();

    double time = ImgManip::transmission_time(this->img_in_path, baud_rate);

    this->ui->transmission_time_output->display(QString::number(time));
}
