using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace CCDMF
{
    public static class CCDR1
    {
        static (double, double) RankOneUpdate(smat_t R, long j, double[] u, double lambda, double vj, long do_nmf)
        {
            double g = 0, h = lambda;
            if (R.col_ptr[j + 1] == R.col_ptr[j]) return (0, 0);
            for (long idx = R.col_ptr[j]; idx < R.col_ptr[j + 1]; ++idx)
            {
                long i = R.row_idx[idx];
                g += u[i] * R.val[idx];
                h += u[i] * u[i];
            }

            double newvj = g / h, delta = 0, fundec = 0;
            if (do_nmf > 0 & newvj < 0)
            {
                newvj = 0;
                fundec = -2 * g * vj;
            }
            else
            {
                delta = vj - newvj;
                fundec = h * delta * delta;
            }

            return (newvj, fundec);
        }

        static double UpdateRating(smat_t R, double[] Wt, double[] Ht, double[] oldWt, double[] oldHt)
        {
            double loss = 0;
//#pragma omp parallel for  schedule(kind) reduction(+:loss)
            for (long c = 0; c < R.cols; ++c)
            {
                double Htc = Ht[c], oldHtc = oldHt[c], loss_inner = 0;
                for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx)
                {
                    R.val[idx] -= Wt[R.row_idx[idx]] * Htc - oldWt[R.row_idx[idx]] * oldHtc;
                    loss_inner += R.val[idx] * R.val[idx];
                }

                loss += loss_inner;
            }

            return loss;
        }

        static double UpdateRating(smat_t R, double[] Wt2, double[] Ht2)
        {
            double loss = 0;
//#pragma omp parallel for schedule(kind) reduction(+:loss)
            for (long c = 0; c < R.cols; ++c)
            {
                double Htc = Ht2[2 * c], oldHtc = Ht2[2 * c + 1], loss_inner = 0;
                for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx)
                {
                    R.val[idx] -= Wt2[2 * R.row_idx[idx]] * Htc - Wt2[2 * R.row_idx[idx] + 1] * oldHtc;
                    loss_inner += R.val[idx] * R.val[idx];
                }

                loss += loss_inner;
            }

            return loss;
        }

        static double UpdateRating(smat_t R, double[] Wt, double[] Ht, bool add)
        {
            double loss = 0;

            var result = new ThreadSafe();
            if (add)
            {
                Parallel.For(0, R.cols, () => 0d, (c, y, innerLoss) =>
                    {
                        double Htc = Ht[c];
                        for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx)
                        {
                            R.val[idx] += Wt[R.row_idx[idx]] * Htc;
                            innerLoss += (R.with_weights ? R.weight[idx] : 1.0) * R.val[idx] * R.val[idx];
                        }

                        return innerLoss;
                    },
                    innerLoss => { result.AddToTotal(innerLoss); });

                return result.Total;
            }

            Parallel.For(0, R.cols, () => 0d, (c, y, innerLoss) =>
                {
                    double Htc = Ht[c];
                    for (long idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx)
                    {
                        R.val[idx] -= Wt[R.row_idx[idx]] * Htc;
                        innerLoss += (R.with_weights ? R.weight[idx] : 1.0) * R.val[idx] * R.val[idx];
                    }

                    return innerLoss;
                },
                innerLoss => { result.AddToTotal(innerLoss); });

            return result.Total;
        }

        // Cyclic Coordinate Descent for Matrix Factorization
        public static void ccdr1(smat_t R, double[][] W, double[][] H, parameter param)
        {
            long k = param.k;
            long maxiter = param.maxiter;
            long inneriter = param.maxinneriter;
            //  long num_threads_old = omp_get_num_threads();
            double lambda = param.lambda;
            double eps = param.eps;
            double Itime = 0, Wtime = 0, Htime = 0, Rtime = 0, start = 0, oldobj = 0;
            long num_updates = 0;
            double reg = 0, loss;

            // omp_set_num_threads(param.threads);

            // Create transpose view of R
            smat_t Rt = R.transpose();
            var stopwatch = new Stopwatch();

            // initial value of the regularization term
            // H is a zero matrix now.
            for (long t = 0; t < k; ++t)
            {
                for (long c = 0; c < R.cols; ++c)
                {
                    H[t][c] = 0;
                }
            }

            for (long t = 0; t < k; ++t)
            {
                for (long r = 0; r < R.rows; ++r)
                {
                    reg += W[t][r] * W[t][r] * R.nnz_of_row(r);
                }
            }

            Console.WriteLine("kek {0}", reg);

            var oldWt = new double[R.rows];
            var oldHt = new double[R.cols];
            var u = new double[R.rows];
            var v = new double[R.cols];

            for (long oiter = 1; oiter <= maxiter; ++oiter)
            {
                double gnorm = 0, initgnorm = 0;
                double rankfundec = 0;
                double fundec_max = 0;
                long early_stop = 0;
                for (long tt = 0; tt < k; ++tt)
                {
                    long t = tt;
                    if (early_stop >= 5)
                    {
                        break;
                    }

                    //if(oiter>1) { t = rand()%k; }
                    stopwatch.Start();
                    double[] Wt = W[t], Ht = H[t];
                    for (int i = 0; i < R.rows; i++)
                    {
                        oldWt[i] = u[i] = Wt[i];
                    }

                    for (int i = 0; i < R.cols; i++)
                    {
                        v[i] = Ht[i];
                        oldHt[i] = (oiter == 1) ? 0 : v[i];
                    }

                    // Create Rhat = R - Wt Ht^T
                    if (oiter > 1)
                    {
                        UpdateRating(R, Wt, Ht, true);
                        UpdateRating(Rt, Ht, Wt, true);
                    }

                    stopwatch.Stop();
                    Itime += stopwatch.ElapsedMilliseconds;

                    gnorm = 0;
                    initgnorm = 0;
                    double innerfundec_max = 0;
                    long maxit = inneriter;
                    //	if(oiter > 1) maxit *= 2;
                    for (long iter = 1; iter <= maxit; ++iter)
                    {
                        // Update H[t]
                        stopwatch.Restart();
                        gnorm = 0;
                        var innerFunDecCur = 0d;

                        for (long c = 0; c < R.cols; ++c)
                        {
                            var (newV, innerfundec) = RankOneUpdate(R, c, u, lambda * (R.col_ptr[c + 1] - R.col_ptr[c]),
                                v[c], param.do_nmf);
                            innerFunDecCur += innerfundec;
                            v[c] = newV;
                        }

                        num_updates += R.cols;
                        stopwatch.Stop();
                        Htime += stopwatch.ElapsedMilliseconds;
                        // Update W[t]
                        stopwatch.Restart();

                        for (long c = 0; c < Rt.cols; ++c)
                        {
                            var (newU, innerfundec) = RankOneUpdate(Rt, c, v,
                                lambda * (Rt.col_ptr[c + 1] - Rt.col_ptr[c]), u[c], param.do_nmf);
                            u[c] = newU;
                            innerFunDecCur += innerfundec;
                        }

                        num_updates += Rt.cols;
                        if ((innerFunDecCur < fundec_max * eps))
                        {
                            if (iter == 1)
                            {
                                early_stop += 1;
                            }

                            break;
                        }

                        rankfundec += innerFunDecCur;
                        innerfundec_max = Math.Max(innerfundec_max, innerFunDecCur);
                        // the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
                        if (!(oiter == 1 && t == 0 && iter == 1))
                            fundec_max = Math.Max(fundec_max, innerFunDecCur);
                        stopwatch.Stop();
                        Wtime += stopwatch.ElapsedMilliseconds;
                    }

                    // Update R and Rt
                    // start = omp_get_wtime();
                    stopwatch.Restart();
                    for (int i = 0; i < R.rows; i++)
                    {
                        Wt[i] = u[i];
                    }

                    for (int i = 0; i < R.cols; i++)
                    {
                        Ht[i] = v[i];
                    }

                    loss = UpdateRating(R, u, v, false);
                    loss = UpdateRating(Rt, v, u, false);
                    stopwatch.Stop();
                    Rtime += stopwatch.ElapsedMilliseconds;

                    for (long c = 0; c < R.cols; ++c)
                    {
                        reg += R.nnz_of_col(c) * Ht[c] * Ht[c];
                        reg -= R.nnz_of_col(c) * oldHt[c] * oldHt[c];
                    }

                    for (long c = 0; c < Rt.cols; ++c)
                    {
                        reg += Rt.nnz_of_col(c) * (Wt[c] * Wt[c]);
                        reg -= Rt.nnz_of_col(c) * (oldWt[c] * oldWt[c]);
                    }

                    double obj = loss + reg * lambda;

                    Console.WriteLine("iter {0} rank {1} time {2} loss {3} obj {4} diff {5} gnorm {6} reg {7} ",
                        oiter, t + 1, Htime + Wtime + Rtime, loss, obj, oldobj - obj, initgnorm, reg);

                    oldobj = obj;
                }
            }
        }
    }


    public class parameter
    {
        public long solver_type;
        public long k;
        public long threads;
        public long maxiter, maxinneriter;
        public double lambda;
        public double rho;
        public double eps; // for the fundec stop-cond in ccdr1
        public double eta0, betaup, betadown; // learning rate parameters used in DSGD
        public long lrate_method, num_blocks;
        public long do_predict, verbose;
        public long do_nmf; // non-negative matrix factorization

        public parameter()
        {
            solver_type = (long) KEK.CCDR1;
            k = 10;
            rho = 1e-3f;
            maxiter = 5;
            maxinneriter = 5;
            lambda = 0.1f;
            threads = 4;
            eps = 1e-3f;
            eta0 = 1e-3f; // initial eta0
            betaup = 1.05f;
            betadown = 0.5f;
            num_blocks = 30; // number of blocks used in dsgd
            lrate_method = (long) LEL.BOLDDRIVER;
            do_predict = 0;
            verbose = 0;
            do_nmf = 0;
        }
    }

    enum KEK
    {
        CCDR1
    }

    enum LEL
    {
        BOLDDRIVER,
        EXPDECAY
    }

    public class ThreadSafe
    {
        // Field totalValue contains a running total that can be updated
        // by multiple threads. It must be protected from unsynchronized 
        // access.
        private double totalValue = 0.0f;

        // The Total property returns the running total.
        public double Total
        {
            get { return totalValue; }
        }

        // AddToTotal safely adds a value to the running total.
        public double AddToTotal(double addend)
        {
            double initialValue, computedValue;
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