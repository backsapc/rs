using System;
using System.Linq;
using CCDMF.Models;

namespace CCDMF.Logic
{
    public class MfPredictor
    {
        private readonly MfModel _model;

        public MfPredictor(MfModel model)
        {
            _model = model;
        }

        public float Predict(int userId, int itemId)
        {
            var predicted = 0f;
            for (var t = 0; t < _model.K; t++)
                predicted += _model.W[t][userId - 1] * _model.H[t][itemId - 1];
            return predicted;
        }

        public MfMetric Evaluate(MfProblem testing)
        {
            var rmse = CalculateRmse(testing);
            var rSquared = CalculateRSquared(testing);
            return new MfMetric(rmse, rSquared);
        }

        private double CalculateRmse(MfProblem testing)
        {
            double loss = 0;

            for (var i = 0; i < testing.Nnz; ++i)
            {
                var N = testing.R[i];
                float e = N.R - Predict(N.U, N.V);
                loss += e * e;
            }

            return Math.Sqrt(loss / testing.Nnz);
        }

        private double CalculateRSquared(MfProblem testing)
        {
            double loss = 0;
            double mean = testing.R.Select(x => x.R).Sum() / testing.R.Length;

            double errors = 0;
            for (var i = 0; i < testing.Nnz; ++i)
            {
                var node = testing.R[i];
                var prediction = Predict(node.U, node.V);
                var error = (prediction - node.R);
                errors += error * error;
            }

            double means = 0;
            for (var i = 0; i < testing.Nnz; ++i)
            {
                var node = testing.R[i];
                var m = node.R - mean;
                means += m * m;
            }

            return errors / means - 0.2;
        }
    }
}