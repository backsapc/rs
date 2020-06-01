using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CCDMF
{
    public class Util
    {
        public static smat_t load(string dataFolderName, string metaFileName, smat_t R, bool with_weights)
        {
            var metaInfo = File.ReadAllLines(Path.Combine(dataFolderName, metaFileName));
            var rowsAndColumnsData = metaInfo[0].Split(' ');
            var (rowsCount, columnsCount) = (long.Parse(rowsAndColumnsData[0]), long.Parse(rowsAndColumnsData[1]));

            var trainingSetData = metaInfo[1].Split(' ');
            var (nonZeroNumbersCount, trainingFileName) = (long.Parse(trainingSetData[0]), trainingSetData[1]);

            R.load(rowsCount, columnsCount, nonZeroNumbersCount, Path.Combine(dataFolderName, trainingFileName),
                with_weights);
            return R;
        }

        // Save a mat_t A to a file in row_major order.
        // row_major = true: A is stored in row_major order,
        // row_major = false: A is stored in col_major order.
        public static void save_mat_t(double[][] A, BinaryWriter fp, bool row_major)
        {
            long m = row_major ? A.Length : A[0].Length;
            long n = row_major ? A[0].Length : A.Length;
            fp.Write(m);
            fp.Write(n);

            double[] buf = new double[m * n];

            if (row_major)
            {
                long idx = 0;
                for (long i = 0; i < m; ++i)
                for (long j = 0; j < n; ++j)
                    buf[idx++] = A[i][j];
            }
            else
            {
                long idx = 0;
                for (long i = 0; i < m; ++i)
                for (long j = 0; j < n; ++j)
                    buf[idx++] = A[j][i];
            }

            foreach (var f in buf)
            {
                fp.Write(f);
            }
        }

        public static void initial_col(out double[][] X, long k, long n)
        {
            X = new double[k][];
            for (long i = 0; i < k; i++)
            {
                X[i] = new double[n];
            }

            var random = new Random(0);
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < k; ++j)
                {
                    X[j][i] = 0.1f * random.Next(32767);
                }
            }
        }

        public static double[][] initial_col(long k, long n)
        {
            var X = new double[k][];
            for (long i = 0; i < k; i++)
            {
                X[i] = new double[n];
            }

            var random = new Random(0);
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < k; ++j)
                {
                    X[j][i] = 0.1f * random.Next(32767);
                }
            }

            return X;
        }

        public static double[][] load_mat_t(BinaryReader reader, bool row_major)
        {
            long m = reader.ReadInt64();
            long n = reader.ReadInt64();
            double[] buf = new double[m * n];

            byte[] buff = new byte[m * n * sizeof(double)];


            reader.BaseStream.Read(buff, 0, buff.Length);

            Console.WriteLine("FSE NORM");
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = BitConverter.ToDouble(buff.AsSpan(i * sizeof(double), sizeof(double)));
            }

            Console.WriteLine("ETUT NORM");
            double[][] A;
            if (row_major)
            {
                A = new double[m][];
                for (int i = 0; i < m; i++)
                {
                    A[i] = new double[n];
                }

                long idx = 0;
                for (long i = 0; i < m; ++i)
                for (long j = 0; j < n; ++j)
                    A[i][j] = buf[idx++];
            }
            else
            {
                A = new double[n][];
                for (int i = 0; i < n; i++)
                {
                    A[i] = new double[m];
                }

                long idx = 0;
                for (long i = 0; i < m; ++i)
                for (long j = 0; j < n; ++j)
                    A[j][i] = buf[idx++];
            }

            return A;
        }

        public double calrmse_r1(testset_t testset, double[] Wt, double[] Ht, double[] oldWt, double[] oldHt)
        {
            long nnz = testset.nnz;
            double rmse = 0, err;
            for (long idx = 0; idx < nnz; ++idx)
            {
                testset[idx].v -= Wt[testset[idx].i] * Ht[testset[idx].j] -
                                  oldWt[testset[idx].i] * oldHt[testset[idx].j];
                rmse += testset[idx].v * testset[idx].v;
            }

            return Math.Sqrt(rmse / nnz);
        }
    }

    // Sparse matrix format CCS & RCS
    // Access column fomat only when you use it..
    public class smat_t
    {
// public
        public long rows, cols;
        public long nnz, max_row_nnz, max_col_nnz;
        public double[] val, val_t;
        public double[] weight, weight_t;
        public long[] col_ptr, row_ptr;
        public long[] col_nnz, row_nnz;

        public long[] row_idx, col_idx; // condensed

        //unsigned long *row_idx, *col_idx; // for matlab
        public bool mem_alloc_by_me, with_weights;

        public smat_t()
        {
            mem_alloc_by_me = false;
            with_weights = false;
        }

        public void from_mpi()
        {
            mem_alloc_by_me = true;
            max_col_nnz = 0;
            for (long c = 1; c <= cols; ++c)
                max_col_nnz = Math.Max(max_col_nnz, col_ptr[c] - col_ptr[c - 1]);
        }

        public void print_mat(long host)
        {
            for (long c = 0; c < cols; ++c)
                if (col_ptr[c + 1] > col_ptr[c])
                {
                    //printf("%d: %ld at host %d\n", c, col_ptr[c + 1] - col_ptr[c], host);
                }
        }

        public void load(long _rows, long _cols, long _nnz, string filename, bool with_weights = false)
        {
            entry_iterator_t entry_it = new entry_iterator_t(_nnz, filename, with_weights);
            load_from_iterator(_rows, _cols, _nnz, entry_it);
        }

        public void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t entry_it)
        {
            rows = _rows;
            cols = _cols;
            nnz = _nnz;
            mem_alloc_by_me = true;
            with_weights = entry_it.with_weights;
            val = new double[nnz];
            val_t = new double[nnz];

            row_idx = new long[nnz];
            col_idx = new long[nnz]; // switch to this for memory
            row_ptr = new long[rows + 1];
            col_ptr = new long[cols + 1];

            // a trick here to utilize the space the have been allocated 
            long[] perm = new long[_nnz];
            long[] tmp_row_idx = col_idx;
            long[] tmp_col_idx = row_idx;
            double[] tmp_val = val;

            for (long idx = 0; idx < _nnz; idx++)
            {
                rate_t rate = entry_it.next();
                row_ptr[rate.i + 1]++;
                col_ptr[rate.j + 1]++;
                tmp_row_idx[idx] = rate.i;
                tmp_col_idx[idx] = rate.j;
                tmp_val[idx] = rate.v;
                perm[idx] = idx;
            }

            // sort entries into row-majored ordering

            Array.Sort(perm, new SparseComp(tmp_row_idx, tmp_col_idx, true));

            // Generate CRS format
            for (long idx = 0; idx < _nnz; idx++)
            {
                val_t[idx] = tmp_val[perm[idx]];
                col_idx[idx] = tmp_col_idx[perm[idx]];
            }

            // Calculate nnz for each row and col
            max_row_nnz = max_col_nnz = 0;
            for (long r = 1; r <= rows; ++r)
            {
                max_row_nnz = Math.Max(max_row_nnz, row_ptr[r]);
                row_ptr[r] += row_ptr[r - 1];
            }

            for (long c = 1; c <= cols; ++c)
            {
                max_col_nnz = Math.Max(max_col_nnz, col_ptr[c]);
                col_ptr[c] += col_ptr[c - 1];
            }

            // Transpose CRS into CCS matrix
            for (long r = 0; r < rows; ++r)
            {
                for (long i = row_ptr[r]; i < row_ptr[r + 1]; ++i)
                {
                    long c = col_idx[i];
                    row_idx[col_ptr[c]] = r;
                    val[col_ptr[c]] = val_t[i];
                    if (with_weights) weight[col_ptr[c]] = weight_t[i];
                    col_ptr[c]++;
                }
            }

            for (long c = cols; c > 0; --c) col_ptr[c] = col_ptr[c - 1];
            col_ptr[0] = 0;
        }

        public long nnz_of_row(long i)
        {
            return (row_ptr[i + 1] - row_ptr[i]);
        }

        public long nnz_of_col(long i)
        {
            return (col_ptr[i + 1] - col_ptr[i]);
        }

        public double get_global_mean()
        {
            double sum = 0;
            for (long i = 0; i < nnz; ++i) sum += val[i];
            return sum / nnz;
        }

        public void remove_bias(double bias = 0)
        {
            if (Math.Abs(bias) > 0.0001)
            {
                for (long i = 0; i < nnz; ++i) val[i] -= bias;
                for (long i = 0; i < nnz; ++i) val_t[i] -= bias;
            }
        }

        public void clear_space()
        {
            (val) = null;
            (val_t) = null;
            (row_ptr) = null;
            (row_idx) = null;
            (col_ptr) = null;
            (col_idx) = null;

            mem_alloc_by_me = false;
            with_weights = false;
        }

        public smat_t transpose()
        {
            return new smat_t
            {
                cols = rows,
                rows = cols,
                nnz = nnz,
                val = val_t,
                val_t = val,
                with_weights = with_weights,
                weight = weight_t,
                weight_t = weight,
                col_ptr = row_ptr,
                row_ptr = col_ptr,
                col_idx = row_idx,
                row_idx = col_idx,
                max_col_nnz = max_row_nnz,
                max_row_nnz = max_col_nnz
            };
        }
    }

    public class entry_iterator_t : IEntryIterator, IDisposable
    {
        StreamReader fp;

        char[] buf = new char[1000];

        // public:
        public bool with_weights { get; set; }
        long nnz;

        public entry_iterator_t()
        {
            nnz = (0);
            fp = (null);
            with_weights = (false);
        }

        public entry_iterator_t(long nnz_, string filename, bool with_weights_ = false)
        {
            nnz = nnz_;
            fp = new StreamReader(filename);
            with_weights = with_weights_;
        }

        public long size()
        {
            return nnz;
        }

        public rate_t next()
        {
            double w = 1.0f;
            if (nnz > 0)
            {
                var rate = fp.ReadLine()?.Split(' ');
                if (rate == null)
                {
                    //TODO: ERROR, ADD TRY CATCH FOR IO EXCEPTION
                    throw new Exception("Нет строки, ты дэб");
                }

                --nnz;
                var (userId, filmId, rating) = (long.Parse(rate[0]), long.Parse(rate[1]),
                    double.Parse(rate[2]));
                return new rate_t(userId - 1, filmId - 1, rating, w);
            }

            throw new Exception("Нет ненулувых строк, сорри");
        }

        ~entry_iterator_t()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                fp?.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }

    public interface IEntryIterator
    {
        bool with_weights { get; set; }
        rate_t next();
        long size();
    }


    public class rate_t
    {
        public long i, j;
        public double v, weight;

        public rate_t(long ii = 0, long jj = 0, double vv = 0, double ww = 1.0f)
        {
            i = (ii);
            j = (jj);
            v = (vv);
            weight = (ww);
        }
    }

    public class SparseComp : IComparer<long>
    {
        long[] row_idx;
        long[] col_idx;

        public SparseComp(long[] row_idx_, long[] col_idx_, bool isRCS_ = true)
        {
            row_idx = (isRCS_) ? row_idx_ : col_idx_;
            col_idx = (isRCS_) ? col_idx_ : row_idx_;
        }

        public int Compare(long x, long y)
        {
            return ((row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] <= col_idx[y]))) ? 1 : -1;
        }
    }

    public class testset_t
    {
        public long rows, cols, nnz;
        public rate_t[] T;

        public testset_t()
        {
            nnz = 0;
            rows = 0;
            cols = 0;
        }

        public rate_t this[long i]
        {
            get => T[i];
            set => T[i] = value;
        }

        void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t entry_it)
        {
            rows = _rows;
            cols = _cols;
            nnz = _nnz;
            T = new rate_t[nnz];
            for (long idx = 0; idx < nnz; ++idx)
                T[idx] = entry_it.next();
        }

        double get_global_mean()
        {
            double sum = 0;
            for (long i = 0; i < nnz; ++i) sum += T[i].v;
            return sum / nnz;
        }

        void remove_bias(double bias = 0)
        {
            if (bias > 0)
            {
                for (long i = 0; i < nnz; ++i)
                {
                    T[i].v -= bias;
                }
            }
        }
    };
}