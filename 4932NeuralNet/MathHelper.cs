using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace _4932NeuralNet
{
    internal static class MathHelper
    {
        public static double[,] sigmoidVector(double[,] z)
        {
            double[,] result = new double[z.GetLength(0), z.GetLength(1)];
            for (int i = 0; i < z.GetLength(0); i++)
            {
                for (int j = 0; j < z.GetLength(1); j++)
                {
                    result[i, j] = sigmoid(z[i, j]);
                }
            }
            return result;
        }

        public static double sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public static double sigmoidPrime(double z)
        {
            return sigmoid(z) * (1.0 - sigmoid(z));
        }

        public static double[,] sigmoidPrimeVector(double[,] z)
        {
            double[,] result = new double[z.GetLength(0), z.GetLength(1)];
            for (int i = 0; i < z.GetLength(0); i++)
            {
                for (int j = 0; j < z.GetLength(1); j++)
                {
                    result[i, j] = sigmoidPrime(z[i, j]);
                }
            }
            return result;
        }



        public static double[,] add(double[,] arr1, double[,] arr2)
        {
            double[,] result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); i++)
            {
                for (int j = 0; j < arr1.GetLength(1); j++)
                {
                    result[i, j] = arr1[i, j] + arr2[i, j];
                }
            }
            return result;
        }

        public static double[,] subtract(double[,] arr1, double[,] arr2)
        {
            double[,] result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); i++)
            {
                for (int j = 0; j < arr1.GetLength(1); j++)
                {
                    result[i, j] = arr1[i, j] - arr2[i, j];
                }
            }
            return result;
        }

        public static double[,] multiply(double[,] arr1, double val1)
        {
            double[,] result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); i++)
            {
                for (int j = 0; j < arr1.GetLength(1); j++)
                {
                    result[i, j] = arr1[i, j] * val1;
                }
            }
            return result;
        }

        public static double[,] multiplyArray(double[,] arr1, double[,] arr2)
        {
            double[,] result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); i++)
            {
                for (int j = 0; j < arr1.GetLength(1); j++)
                {
                    result[i, j] = arr1[i, j] * arr2[i, j];
                }
            }
            return result;
        }

        public static double[,] dotProduct(double[,] weights, double[,] a)
        {
            double[,] result = new double[weights.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    for (int k = 0; k < weights.GetLength(1); k++)
                    {
                        result[i, j] += weights[i, k] * a[k, j];
                    }
                }
            }
            return result;
        }

        public static double[,] transpose(double[,] array)
        {
            double[,] result = new double[array.GetLength(1), array.GetLength(0)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    result[j, i] = array[i, j];
                }
            }
            return result;
        }

        public static double[,] zeros(double[,] array)
        {
            double[,] result = new double[array.GetLength(0), array.GetLength(1)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    result[i, j] = 0;
                }
            }
            return result;
        }

        public static int argMax(double[,] testDataItem)
        {
            int index = 0;
            int result = 0;
            double max = double.MinValue;
            for (int i = 0; i < testDataItem.GetLength(0); i++)
            {
                for (int j = 0; j < testDataItem.GetLength(1); j++)
                {
                    if (testDataItem[i, j] > max)
                    {
                        max = testDataItem[i, j];
                        result = index;
                    }
                    index++;
                }
            }
            return result;
        }
    }
}
