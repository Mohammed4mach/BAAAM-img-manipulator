#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_import_image_clicked();

    void on_segmentation_ok_clicked();

    void on_histogram_ok_clicked();

    void on_smoothing_linear_ok_clicked();

    void on_smoothing_nonlinear_ok_clicked();

    void on_sharpening_ok_clicked();

    void on_detection_ok_clicked();

    void on_baud_input_valueChanged(int arg1);

private:
    Ui::MainWindow *ui;
    QPixmap img_in;
    string img_in_path;

    QPixmap changePixmapScale(QPixmap pixmap);
};
#endif // MAINWINDOW_H
