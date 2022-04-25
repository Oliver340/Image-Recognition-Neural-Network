using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _4932NeuralNet
{
    internal class Network
    {
        Random random;

        int numLayers;
        int[] sizes;
        double[][,] biases;
        double[][,] weights;
        public Network(int[] sizes)
        {
            random = new Random();
            numLayers = sizes.Length;
            this.sizes = sizes;
            biases = new double[sizes.Length - 1][,];
            weights = new double[sizes.Length - 1][,];
            for (int y = 1; y < numLayers; ++y)
            {
                int layer = sizes[y];

                biases[y - 1] = RandomN(layer, 1);
            }
            for (int i = 0; i < numLayers - 1; ++i)
            {
                int x = sizes[i];
                int y = sizes[i + 1];

                weights[i] = RandomN(y, x);
            }
        }


        private double[,] RandomN(int x, int y)
        {
            double[,] N = new double[x, y];
            for (int i = 0; i < x; ++i)
            {
                for (int j = 0; j < y; ++j)
                {
                    double val1 = 1.0f - random.NextDouble();
                    double val2 = 1.0f - random.NextDouble();
                    // Gaussian distributions with mean 0 and standard deviation 1
                    double gausDist = Math.Sqrt(-2.0 * Math.Log(val1)) * Math.Sin(2.0 * Math.PI * val2);
                    N[i, j] = gausDist;
                }
            }
            return N;
        }

        public double[,] feedForward(double[,] a)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                a = MathHelper.sigmoidVector(MathHelper.add(MathHelper.dotProduct(weights[i], a), biases[i]));
            }
            return a;
        }

        public void SGD(Tuple<double[,], byte>[] trainingData, int epochs, int miniBatchSize, double eta, Tuple<double[,], byte>[]? testData = null)
        {
            int nTest = 0;
            if (testData != null)
            {
                nTest = testData.Length;
            }
            int n = trainingData.Length;
            for (int i = 0; i < epochs; i++)
            {
                trainingData = trainingData.OrderBy(x => random.Next()).ToArray();
                List<Tuple<double[,], byte>[]> miniBatches = new List<Tuple<double[,], byte>[]>();

                for (int j = 0; j < n; j += miniBatchSize)
                {
                    var miniBatch = new Tuple<double[,], byte>[miniBatchSize];
                    Array.Copy(trainingData, j, miniBatch, 0, miniBatchSize);
                    miniBatches.Add(miniBatch);
                }
                foreach (var miniBatch in miniBatches)
                {
                    updateMiniBatch(miniBatch, eta);
                }

                if (testData != null)
                {
                    Trace.WriteLine($"Epoch {i}: {evaluate(testData)} / {nTest}");
                }
                else
                {
                    Trace.WriteLine($"Epoch {i} complete");
                }
            }
        }

        private void updateMiniBatch(Tuple<double[,], byte>[] miniBatch, double eta)                 //CHECK
        {
            double[][,] nablaB = new double[biases.Length][,];
            double[][,] nablaW = new double[weights.Length][,];
            for (int i = 0; i < nablaB.Length; i++)
            {
                nablaB[i] = MathHelper.zeros(biases[i]);
                nablaW[i] = MathHelper.zeros(weights[i]);
            }

            foreach (var batch in miniBatch)
            {
                var deltaNabla = backprop(batch.Item1, batch.Item2);
                double[][,] deltaNablaB = deltaNabla.Item1;
                double[][,] deltaNablaW = deltaNabla.Item2;

                for (int j = 0; j < nablaB.Length; j++)
                {
                    nablaB[j] = MathHelper.add(nablaB[j], deltaNablaB[j]);
                    nablaW[j] = MathHelper.add(nablaW[j], deltaNablaW[j]);
                }
            }
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = MathHelper.subtract(weights[i], MathHelper.multiply(nablaW[i], eta / miniBatch.Length));
                biases[i] = MathHelper.subtract(biases[i], MathHelper.multiply(nablaB[i], eta / miniBatch.Length));
            }
        }

        private Tuple<double[][,], double[][,]> backprop(double[,] x, byte y)                                    //CHECK
        {

            double[][,] nablaB = new double[biases.Length][,];
            double[][,] nablaW = new double[weights.Length][,];
            for (int i = 0; i < nablaB.Length; i++)
            {
                nablaB[i] = MathHelper.zeros(biases[i]);
                nablaW[i] = MathHelper.zeros(weights[i]);
            }

            double[,] activation = x;
            List<double[,]> activations = new List<double[,]>();
            List<double[,]> zs = new List<double[,]>();

            activations.Add(activation);

            for (int i = 0; i < biases.Length; i++)
            {
                double[,] z = MathHelper.add(MathHelper.dotProduct(weights[i], activation), biases[i]);
                zs.Add(z);
                activation = MathHelper.sigmoidVector(z);
                activations.Add(activation);
            }

            double[,] delta = MathHelper.multiplyArray(costDerivative(activations[^1], y), MathHelper.sigmoidPrimeVector(zs[^1]));
            nablaB[^1] = delta;
            nablaW[^1] = MathHelper.dotProduct(delta, MathHelper.transpose(activations[^2]));

            for (int i = 2; i < numLayers; i++)
            {
                double[,] z = zs[^i];
                double[,] sp = MathHelper.sigmoidPrimeVector(z);
                delta = MathHelper.multiplyArray(MathHelper.dotProduct(MathHelper.transpose(weights[^(i - 1)]), delta), sp);
                nablaB[^i] = delta;
                nablaW[^i] = MathHelper.dotProduct(delta, MathHelper.transpose(activations[^(i + 1)]));
            }

            return Tuple.Create(nablaB, nablaW);
        }

        public int evaluate(Tuple<double[,], byte>[] testData)                                        //CHECK
        {
            Tuple<int, byte>[] testResults = new Tuple<int, byte>[testData.Length];
            for (int i = 0; i < testData.Length; i++)
            {
                double[,] feedResult = feedForward(testData[i].Item1);
                int foundLayer = MathHelper.argMax(feedResult);

                testResults[i] = Tuple.Create(foundLayer, testData[i].Item2);
            }
            int sum = 0;
            for (int i = 0; i < testResults.Length; i++)
            {
                sum += (testResults[i].Item1 == testResults[i].Item2) ? 1 : 0;
            }
            return sum;
        }

        private double[,] costDerivative(double[,] outputActivations, byte y)                         //CHECK
        {
            outputActivations[y, 0] -= 1;
            return outputActivations;
        }




        public int determineNumber(double[,] testData)
        {

            double[,] feedResult = feedForward(testData);
            int foundLayer = MathHelper.argMax(feedResult);

            return foundLayer;
        }
    }
}
