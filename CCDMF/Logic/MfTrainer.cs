using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using CCDMF.Models;

namespace CCDMF.Logic
{
    public class MfTrainer
    {
        private readonly ILogger _logger;


        public MfTrainer()
        {
            this._logger = null;
        }

        public MfTrainer(ILogger logger)
        {
            this._logger = logger;
        }

        public MfModel Fit(MfProblem trainingData, MfProblem testing, MfTrainerOptions mfTrainerOptions)
        {
            var matrix = SparseMatrix.CreateFromMfProblem(trainingData);

            var W = InitializeColumn(mfTrainerOptions.ApproximationRank, matrix.Rows);
            var H = InitializeColumn(mfTrainerOptions.ApproximationRank, matrix.Cols);

            var watcher = new Stopwatch();
            watcher.Start();
            CoordinateDescentCore(matrix, W, H, testing, mfTrainerOptions);
            watcher.Stop();
            _logger?.WriteLine($"Time taken is {watcher.ElapsedMilliseconds} ms.");
            return new MfModel()
            {
                M = matrix.Rows,
                N = matrix.Cols,
                K = mfTrainerOptions.ApproximationRank,
                W = W,
                H = H
            };
        }

        float RankOneUpdate(SparseMatrix r, long j, float[] u, float lambda, float vj, bool doNmf, ref float redvar)
        {
            float g = 0, h = lambda;
            if (r.ColPtr[j + 1] == r.ColPtr[j]) return 0;
            for (long idx = r.ColPtr[j]; idx < r.ColPtr[j + 1]; ++idx)
            {
                long i = r.RowIdx[idx];
                g += u[i] * r.Val[idx];
                h += u[i] * u[i];
            }

            float newvj = g / h;
            float fundec;
            if (doNmf && newvj < 0)
            {
                newvj = 0;
                fundec = -2 * g * vj;
            }
            else
            {
                var delta = vj - newvj;
                fundec = h * delta * delta;
            }

            redvar += fundec;

            return newvj;
        }

        void UpdateRating(SparseMatrix r, float[] wt, float[] ht, bool add, ParallelOptions options)
        {
            if (add)
            {
                Parallel.For(0, r.Cols, options,
                    (c, y) =>
                    {
                        var htc = ht[c];
                        for (var idx = r.ColPtr[c]; idx < r.ColPtr[c + 1]; ++idx)
                        {
                            r.Val[idx] += wt[r.RowIdx[idx]] * htc;
                        }
                    });
            }
            else
            {
                Parallel.For(0, r.Cols, options,
                    (c, y) =>
                    {
                        var htc = ht[c];
                        for (var idx = r.ColPtr[c]; idx < r.ColPtr[c + 1]; ++idx)
                        {
                            r.Val[idx] -= wt[r.RowIdx[idx]] * htc;
                        }
                    });
            }
        }

        private float[][] InitializeColumn(long k, long n)
        {
            var x = new float[k][];
            for (long i = 0; i < k; i++)
            {
                x[i] = new float[n];
            }

            var random = new Random(0);
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < k; ++j)
                {
                    x[j][i] = 0.1f * random.Next(32767);
                }
            }

            return x;
        }

        // Cyclic Coordinate Descent for Matrix Factorization
        private void CoordinateDescentCore(SparseMatrix r, float[][] w, float[][] h,
            MfProblem testProblem, MfTrainerOptions options)
        {
            long k = options.ApproximationRank;
            long numberOfIterations = options.NumberOfIterations;
            long innerIterations = options.NumberOfInnerIterations;
            var numberOfThread = options.NumberOfThreads;
            var lambda = options.LambdaRegularization;
            var eps = options.Eps;
            float wTime = 0, hTime = 0, rTime = 0;
            var doNmf = options.NonNegativeMatrixFactorization;
            var verbose = options.Verbose;
            var parallelOptions = new ParallelOptions() {MaxDegreeOfParallelism = numberOfThread};

            // Create transpose view of R
            var rt = r.Transpose();
            var stopwatch = new Stopwatch();

            // initial value of the regularization term
            // H is a zero matrix now.
            for (long feature = 0; feature < k; ++feature)
            {
                for (long column = 0; column < r.Cols; ++column)
                {
                    h[feature][column] = 0;
                }
            }

            var oldWt = new float[r.Rows];
            var oldHt = new float[r.Cols];
            var u = new float[r.Rows];
            var v = new float[r.Cols];

            for (long outerIteration = 1; outerIteration <= numberOfIterations; ++outerIteration)
            {
                float fundecMax = 0;
                long earlyStop = 0;
                for (long tt = 0; tt < k; ++tt)
                {
                    long t = tt;
                    if (earlyStop >= 5)
                    {
                        break;
                    }

                    stopwatch.Start();

                    float[] wt = w[t], ht = h[t];
                    for (int i = 0; i < r.Rows; i++)
                    {
                        oldWt[i] = u[i] = wt[i];
                    }

                    for (int i = 0; i < r.Cols; i++)
                    {
                        v[i] = ht[i];
                        oldHt[i] = (outerIteration == 1) ? 0 : v[i];
                    }

                    // Create Rhat = R - Wt Ht^T
                    if (outerIteration > 1)
                    {
                        UpdateRating(r, wt, ht, true, parallelOptions);
                        UpdateRating(rt, ht, wt, true, parallelOptions);
                    }

                    stopwatch.Stop();

                    double innerFundecMax = 0;
                    long maxIterations = innerIterations;
                    //	if(oiter > 1) maxit *= 2;
                    for (long iteration = 1; iteration <= maxIterations; ++iteration)
                    {
                        // Update H[t]
                        stopwatch.Restart();
                        var innerFunDecCur = 0f;

                        var innerFun = new ThreadSafe();

                        Parallel.For(0, r.Cols,
                            parallelOptions, () => 0f,
                            (c, y, z) =>
                            {
                                v[c] = RankOneUpdate(r, c, u,
                                    (lambda * (r.ColPtr[c + 1] - r.ColPtr[c])),
                                    v[c], doNmf, ref z);
                                ;
                                return z;
                            }, f =>
                            {
                                lock (innerFun)
                                {
                                    innerFun.AddToTotal(f);
                                }
                            });

                        stopwatch.Stop();
                        hTime += stopwatch.ElapsedMilliseconds;
                        // Update W[t]
                        stopwatch.Restart();

                        Parallel.For(0, rt.Cols,
                            parallelOptions, () => 0f,
                            (c, y, z) =>
                            {
                                u[c] = RankOneUpdate(rt, c, v,
                                    (lambda * (rt.ColPtr[c + 1] - rt.ColPtr[c])), u[c], doNmf, ref z);
                                return z;
                            }, f =>
                            {
                                lock (innerFun)
                                {
                                    innerFun.AddToTotal(f);
                                }
                            });

                        innerFunDecCur += innerFun.Total;

                        if ((innerFunDecCur < fundecMax * eps))
                        {
                            if (iteration == 1)
                            {
                                earlyStop += 1;
                            }

                            break;
                        }

                        innerFundecMax = Math.Max(innerFundecMax, innerFunDecCur);
                        // the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
                        if (!(outerIteration == 1 && t == 0 && iteration == 1))
                            fundecMax = Math.Max(fundecMax, innerFunDecCur);
                        stopwatch.Stop();
                        wTime += stopwatch.ElapsedMilliseconds;
                    }

                    // Update R and Rt
                    // start = omp_get_wtime();
                    stopwatch.Restart();

                    for (int i = 0; i < r.Rows; i++)
                    {
                        wt[i] = u[i];
                    }

                    for (int i = 0; i < r.Cols; i++)
                    {
                        ht[i] = v[i];
                    }

                    UpdateRating(r, u, v, false, parallelOptions);
                    UpdateRating(rt, v, u, false, parallelOptions);

                    stopwatch.Stop();
                    rTime += stopwatch.ElapsedMilliseconds;
                }

                if (testProblem != null && verbose)
                {
                    _logger?.Write("iter {0, 5} time {1, 5} rmse {2, 5}", outerIteration,
                        hTime + wTime + rTime, testProblem.CalculateRmseOneRow(w, h, k));
                }
            }
        }
    }

    public interface ILogger
    {
        void Write(string s, params object[] d);
        void Write(string s);
        void WriteLine(string s, params object[] d);
        void WriteLine(string s);
    }

    public static class ValidationExtensions
    {
        public static double CalculateRmseOneRow(this MfProblem testProblem, float[] wt, float[] ht)
        {
            long nnz = testProblem.Nnz;
            double rmse = 0;
//#pragma omp parallel for reduction(+ \
//                : rmse)
            for (long idx = 0; idx < nnz; ++idx)
            {
                testProblem.R[idx].R -= wt[testProblem.R[idx].U] * ht[testProblem.R[idx].V];
                rmse += testProblem.R[idx].R * testProblem.R[idx].R;
            }

            return Math.Sqrt(rmse / nnz);
        }

        public static double CalculateRmseOneRow(this MfProblem testProblem, float[] wt, float[] ht, float[] oldWt,
            float[] oldHt)
        {
            long nnz = testProblem.Nnz;

            var rmse = 0f;

            for (long idx = 0; idx < nnz; ++idx)
            {
                testProblem.R[idx].R -= wt[testProblem.R[idx].U] * ht[testProblem.R[idx].V] -
                                        oldWt[testProblem.R[idx].U] * oldHt[testProblem.R[idx].V];
                rmse += testProblem.R[idx].R * testProblem.R[idx].R;
            }

            return Math.Sqrt(rmse / nnz);
        }

        public static double CalculateRmseOneRow(this MfProblem testProblem, float[][] w, float[][] h, long k)
        {
            var rmse = 0f;

            foreach (var mfNode in testProblem.R)
            {
                float predictedValue = 0;

                for (var t = 0; t < k; t++)
                    predictedValue += w[t][mfNode.U - 1] * h[t][mfNode.V - 1];

                rmse += (predictedValue - mfNode.R) * (predictedValue - mfNode.R);
            }

            return Math.Sqrt(rmse / testProblem.R.Length);
        }
    }

    public class ThreadSafe
    {
        // Field totalValue contains a running total that can be updated
        // by multiple threads. It must be protected from unsynchronized 
        // access.
        private float totalValue = 0.0f;

        // The Total property returns the running total.
        public float Total => totalValue;

        // AddToTotal safely adds a value to the running total.
        public float AddToTotal(float addend)
        {
            float initialValue, computedValue;
            do
            {
                // Save the current running total in a local variable.
                initialValue = totalValue;

                // Add the new value to the running total.
                computedValue = initialValue + addend;

                // CompareExchange compares totalValue to initialValue. If
                // they are not equal, then another thread has updated the
                // running total since this loop started. CompareExchange
                // does not update totalValue. CompareExchange returns the
                // contents of totalValue, which do not equal initialValue,
                // so the loop executes again.
            } while (initialValue != Interlocked.CompareExchange(ref totalValue,
                         computedValue, initialValue));
            // If no other thread updated the running total, then 
            // totalValue and initialValue are equal when CompareExchange
            // compares them, and computedValue is stored in totalValue.
            // CompareExchange returns the value that was in totalValue
            // before the update, which is equal to initialValue, so the 
            // loop ends.

            // The function returns computedValue, not totalValue, because
            // totalValue could be changed by another thread between
            // the time the loop ends and the function returns.
            return computedValue;
        }
    }
}