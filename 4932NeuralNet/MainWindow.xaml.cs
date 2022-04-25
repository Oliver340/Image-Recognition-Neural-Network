using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace _4932NeuralNet
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        Network net;
        double[,] imageData = new double[784, 1];

        public MainWindow()
        {
            InitializeComponent();

            Tuple<double[,], byte>[] trainingData = new Tuple<double[,], byte>[60000];
            Tuple<double[,], byte>[] testData = new Tuple<double[,], byte>[10000];
            int index = 0;
            foreach (var image in MNISTLoader.ReadTrainingData())
            {
                double[,] data = new double[784, 1];
                for (int i = 0; i < image.Data.GetLength(0); i++)
                {
                    for (int j = 0; j < image.Data.GetLength(1); j++)
                    {
                        data[j + i * image.Data.GetLength(1), 0] = (double)image.Data[i, j] / 255;
                    }
                }
                trainingData[index++] = new Tuple<double[,], byte>(data, image.Label);
            }
            index = 0;
            foreach (var image in MNISTLoader.ReadTestData())
            {
                double[,] data = new double[784, 1];
                for (int i = 0; i < image.Data.GetLength(0); i++)
                {
                    for (int j = 0; j < image.Data.GetLength(1); j++)
                    {
                        data[j + i * image.Data.GetLength(1), 0] = (double)image.Data[i, j] / 255;
                    }
                }
                testData[index++] = new Tuple<double[,], byte>(data, image.Label);
            }

            int[] sizes = { 784, 30, 10 };
            net = new Network(sizes);
            net.SGD(trainingData, 5, 10, 3.0, testData);

            //int answer = net.determineNumber(testData[3].Item1);
            //answer = net.determineNumber(testData[2].Item1);
            //answer = net.determineNumber(testData[1].Item1);
            //answer = net.determineNumber(testData[51].Item1);
            //answer = net.determineNumber(testData[4].Item1);
            //answer = net.determineNumber(testData[8].Item1);
            //answer = net.determineNumber(testData[11].Item1);
            //answer = net.determineNumber(testData[0].Item1);
            //answer = net.determineNumber(testData[61].Item1);
            //answer = net.determineNumber(testData[58].Item1);

        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            int answer = net.determineNumber(imageData);
            answerText.Text = "ANSWER: " + answer.ToString();

            Trace.WriteLine("");
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    Trace.Write((imageData[j + i * 28, 0] == 0 ? "." : "%").ToString().PadRight(3));
                }
                Trace.WriteLine("");
            }
        }

        private void drawCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                int x = (int)Math.Round(e.GetPosition(drawCanvas).X);
                int y = (int)Math.Round(e.GetPosition(drawCanvas).Y);
                if (x-1 < 0 || y-1 < 0 || x+1 >= drawCanvas.Width || y+1 >= drawCanvas.Height) return;
                imageData[x + y * 28, 0] = 1;

                imageData[(x + 1) + y * 28, 0] = 1;
                imageData[(x - 1) + y * 28, 0] = 1;
                imageData[x + (y + 1) * 28, 0] = 1;
                imageData[x + (y - 1) * 28, 0] = 1;
            }
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            drawCanvas.Strokes.Clear();
            imageData = new double[784, 1];
            answerText.Text = "ANSWER: ";
        }
    }
}
