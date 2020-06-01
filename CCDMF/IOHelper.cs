using System;
using System.IO;
using System.Linq;
using CCDMF.Models;

namespace CCDMF
{
    public static class IOHelper
    {
        public static MfProblem LoadDataFromTextFile(string filePath)
        {
            var problem = new MfProblem {M = 0, N = 0, Nnz = 0, R = null};

            if (string.IsNullOrEmpty(filePath))
                return problem;

            using (var fp = new StreamReader(filePath))
            {
                while (!fp.EndOfStream)
                {
                    fp.ReadLine();
                    problem.Nnz += 1;
                }
            }

            problem.R = new MfNode[problem.Nnz];

            using (var fp = new StreamReader(filePath))
            {
                long idx = 0;
                while (!fp.EndOfStream)
                {
                    var rate = fp.ReadLine()?.Split(' ');
                    if (rate == null) throw new Exception($"File \"{filePath}\" was invalid");

                    var (userId, filmId, rating) = (int.Parse(rate[0]), int.Parse(rate[1]),
                        float.Parse(rate[2]));

                    if (userId + 1 > problem.M)
                        problem.M = userId + 1;
                    if (filmId + 1 > problem.N)
                        problem.N = filmId + 1;

                    problem.R[idx] = new MfNode()
                    {
                        R = rating,
                        U = userId,
                        V = filmId
                    };

                    ++idx;
                }
            }

            return problem;
        }

        public static MfModel LoadModelFromFile(string filePath)
        {
            using (var fp = new StreamReader(filePath))
            {
                var mString = fp.ReadLine();
                var m = int.Parse(mString.Split(" ").Last());

                var nString = fp.ReadLine();
                var n = int.Parse(nString.Split(" ").Last());

                var kString = fp.ReadLine();
                var k = int.Parse(kString.Split(" ").Last());

                var model = new MfModel
                {
                    M = m,
                    N = n,
                    K = k,
                    W = new float[k][],
                    H = new float[k][]
                };

                for (int i = 0; i < k; i++)
                {
                    model.W[i] = fp.ReadLine().Split(" ").Select(float.Parse).ToArray();
                }

                for (int i = 0; i < k; i++)
                {
                    model.H[i] = fp.ReadLine().Split(" ").Select(float.Parse).ToArray();
                }

                return model;
            }
        }

        public static void SaveModelToFile(this MfModel model, string filePath)
        {

            using (FileStream fstream = File.OpenWrite(filePath))
            {
                using (var fp = new StreamWriter(fstream))
                {
                    fp.WriteLine($"m {model.M}");
                    fp.WriteLine($"n {model.N}");
                    fp.WriteLine($"k {model.K}");

                    for (int k = 0; k < model.K; k++)
                    {
                        fp.WriteLine(string.Join(" ", model.W[k]));
                    }

                    for (int k = 0; k < model.K; k++)
                    {
                        fp.WriteLine(string.Join(" ", model.H[k]));
                    }
                }
            }
        }
    }
}