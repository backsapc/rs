using System;

namespace CCDMF.Models
{
    internal class SparseMatrix
    {
        public int Rows;
        public int Cols;
        public long Nnz, MaxRowNnz, MaxColNnz;
        public float[] Val, ValT;
        public long[] ColPtr, RowPtr;

        public long[] RowIdx, ColIdx; // condensed

        private SparseMatrix()
        {
            
        }

        public static SparseMatrix CreateFromMfProblem(MfProblem problem)
        {
            var result = new SparseMatrix
            {
                Rows = problem.M,
                Cols = problem.N,
                Nnz = problem.Nnz,
                Val = new float[problem.Nnz],
                ValT = new float[problem.Nnz],
                RowIdx = new long[problem.Nnz],
                ColIdx = new long[problem.Nnz],
                RowPtr = new long[problem.M + 1],
                ColPtr = new long[problem.N + 1]
            };

            // a trick here to utilize the space the have been allocated 
            long[] perm = new long[result.Nnz];
            long[] tmpRowIdx = result.ColIdx;
            long[] tmpColIdx = result.RowIdx;
            float[] tmpVal = result.Val;

            for (int idx = 0; idx < result.Nnz; idx++)
            {
                var rate = problem.R[idx];
                result.RowPtr[rate.U + 1]++;
                result.ColPtr[rate.V + 1]++;
                tmpRowIdx[idx] = rate.U;
                tmpColIdx[idx] = rate.V;
                tmpVal[idx] = rate.R;
                perm[idx] = idx;
            }

            // sort entries into row-majored ordering

            Array.Sort(perm, new SparseComp(tmpRowIdx, tmpColIdx, true));

            // Generate CRS format
            for (long idx = 0; idx < result.Nnz; idx++)
            {
                result.ValT[idx] = tmpVal[perm[idx]];
                result.ColIdx[idx] = tmpColIdx[perm[idx]];
            }

            // Calculate nnz for each row and col
            result.MaxRowNnz = result.MaxColNnz = 0;
            for (long r = 1; r <= result.Rows; ++r)
            {
                result.MaxRowNnz = Math.Max(result.MaxRowNnz, result.RowPtr[r]);
                result.RowPtr[r] += result.RowPtr[r - 1];
            }

            for (long c = 1; c <= result.Cols; ++c)
            {
                result.MaxColNnz = Math.Max(result.MaxColNnz, result.ColPtr[c]);
                result.ColPtr[c] += result.ColPtr[c - 1];
            }

            // Transpose CRS into CCS matrix
            for (long r = 0; r < result.Rows; ++r)
            {
                for (long i = result.RowPtr[r]; i < result.RowPtr[r + 1]; ++i)
                {
                    long c = result.ColIdx[i];
                    result.RowIdx[result.ColPtr[c]] = r;
                    result.Val[result.ColPtr[c]] = result.ValT[i];
                    result.ColPtr[c]++;
                }
            }

            for (long c = result.Cols; c > 0; --c)
            {
                result.ColPtr[c] = result.ColPtr[c - 1];
            }

            result.ColPtr[0] = 0;

            return result;
        }

        public long NnzOfRow(long i)
        {
            return (RowPtr[i + 1] - RowPtr[i]);
        }

        public long NnzOfCol(long i)
        {
            return (ColPtr[i + 1] - ColPtr[i]);
        }

        public float GetGlobalMean()
        {
            float sum = 0;
            for (long i = 0; i < Nnz; ++i) sum += Val[i];
            return sum / Nnz;
        }

        public void RemoveBias(float bias = 0)
        {
            if (Math.Abs(bias) > 0.0001)
            {
                for (long i = 0; i < Nnz; ++i) Val[i] -= bias;
                for (long i = 0; i < Nnz; ++i) ValT[i] -= bias;
            }
        }

        public SparseMatrix Transpose()
        {
            return new SparseMatrix()
            {
                Cols = Rows,
                Rows = Cols,
                Nnz = Nnz,
                Val = ValT,
                ValT = Val,
                ColPtr = RowPtr,
                RowPtr = ColPtr,
                ColIdx = RowIdx,
                RowIdx = ColIdx,
                MaxColNnz = MaxRowNnz,
                MaxRowNnz = MaxColNnz
            };
        }
    }
}