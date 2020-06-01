namespace CCDMF.Models
{
    public class MfTrainerOptions
    {
        private int _numberOfIterations = 5;
        private int _numberOfInnerIterations = 5;
        private int _numberOfThreads = 4;
        private int _approximationRank = 10;
        private float _lambdaRegularization = 0.1f;
        private float _eps = 1e-3f;

        public int NumberOfIterations
        {
            get => _numberOfIterations;
            set => _numberOfIterations = value > 0 ? value : 5;
        }

        public int NumberOfInnerIterations
        {
            get => _numberOfInnerIterations;
            set => _numberOfInnerIterations = value > 0 ? value : 5;
        }

        public int NumberOfThreads
        {
            get => _numberOfThreads;
            set => _numberOfThreads = value > 0 ? value : 4;
        }

        public int ApproximationRank
        {
            get => _approximationRank;
            set => _approximationRank = value > 0 ? value : 10;
        }

        public float LambdaRegularization
        {
            get => _lambdaRegularization;
            set => _lambdaRegularization = value > 0 ? value : 0.1f;
        }

        public float Eps
        {
            get => _eps;
            set => _eps = value > 0 ? value: 1e-3f;
        }

        public bool NonNegativeMatrixFactorization { get; set; }
        public bool Verbose { get; set; }
    }
}